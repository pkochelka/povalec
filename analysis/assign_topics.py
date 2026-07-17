"""Assign a topic to every EuroParl speech so we can build topic-conditional
(party x topic) prototypes downstream.

Three interchangeable topic sources (pick with --topics):

  statements  -> nearest of the 30 EU&I VAA statements, matched IN THE SAME
                 LANGUAGE as the speech (the language offset is shared by all 30
                 candidates, so it cancels in the argmax -> no transform needed).
                 This is the lens that aligns with the LLM target speeches, which
                 are themselves generated per statement.
  cluster     -> KMeans on language-centered embeddings (derived topics; keeps the
                 "any opinionated text" generality, used to drive training batches).
  date        -> the session date as a debate proxy (cheap, noisy baseline).

Reuses the frozen Harrier embeddings already cached by party_prototype_embeddings;
only the 30 statements (x per-language) are embedded fresh here.

Output: data/EuroParl Custom/topics_{split}_{mode}.parquet keyed by speech id.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import v_measure_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import (
    CACHE_DIR,
    DATA_DIR,
    PARTY_COLUMN,
    EmbeddingCache,
    FeatureTransform,
    Harrier,
    chunks,
    ensure_embedded,
    load_split,
    with_instruction,
)

STATEMENTS_PATH = Path("data/euandi_2024_data/statements.jsonl")

# EuroParl/ISO language code -> EU&I statement-file language key.
# Mirror of GEMMA_TO_EP in embed_llm_speeches.py, inverted; identity for the rest.
EP_TO_EUANDI = {
    "cs": "cz", "da": "dk", "et": "ee", "el": "gr", "sl": "si", "sv": "se",
}


def load_statements(path=STATEMENTS_PATH):
    """Returns {statement_idx: {lang_key: text}} in file order (idx matches euandi)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line)["statement"])
    return dict(enumerate(rows))


def statement_text(statements, idx, ep_lang):
    """Statement text in the speech's own language, falling back to English."""
    key = EP_TO_EUANDI.get(ep_lang, ep_lang)
    texts = statements[idx]
    return texts.get(key) or texts.get("en") or next(iter(texts.values()))


def embed_statements(embedder, statements, ep_langs):
    """(n_statements, n_langs, dim) L2-normalized matrix, one column per language."""
    idxs = sorted(statements)
    matrix = {}
    for lang in tqdm(ep_langs, desc="embedding statements"):
        texts = [statement_text(statements, idx, lang) for idx in idxs]
        vecs = []
        for batch in chunks(texts, embedder.batch_size):
            vecs.append(embedder.encode_batch(batch))  # already L2-normalized
        matrix[lang] = np.concatenate(vecs).astype(np.float32)
    return idxs, matrix


def assign_statements(cache, df, embedder, statements):
    idxs, stmt_by_lang = embed_statements(embedder, statements, sorted(df["language"].unique()))
    idx_arr = np.array(idxs)
    cols = ["statement_idx", "statement_sim", "statement_sim2", "statement_margin"]
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    for lang, block in df.groupby("language"):
        emb = cache.get(block["id"].tolist()).astype(np.float32)   # cached, L2-normalized
        sims = emb @ stmt_by_lang[lang].T                          # (n, 30) cosine
        # top-2 per row; margin = sim1 - sim2 cancels the shared anisotropy offset,
        # so it (not the absolute sim) measures how distinctively a speech fits its statement.
        top2 = np.argsort(-sims, axis=1)[:, :2]
        rows = np.arange(len(sims))
        s1 = sims[rows, top2[:, 0]]
        s2 = sims[rows, top2[:, 1]]
        out.loc[block.index, "statement_idx"] = idx_arr[top2[:, 0]]
        out.loc[block.index, "statement_sim"] = s1
        out.loc[block.index, "statement_sim2"] = s2
        out.loc[block.index, "statement_margin"] = s1 - s2
    out["statement_idx"] = out["statement_idx"].astype(int)
    return out


def assign_clusters(cache, df, k, seed=0, fit_sample=300000):
    """Language-center first (so clusters key on topic, not language), then KMeans.
    Fit on a subsample + predict in chunks to stay memory-safe on the 2M-row train split."""
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    ids = df["id"].to_numpy()
    langs = df["language"].to_numpy()
    n = len(df)

    sel = rng.choice(n, min(fit_sample, n), replace=False)
    fit_emb = cache.get(ids[sel].tolist())
    transform = FeatureTransform(kind="lang_center").fit(fit_emb, groups=langs[sel])
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5)
    km.fit(transform.apply(fit_emb, langs[sel]))

    labels = np.empty(n, dtype=int)
    for start in tqdm(range(0, n, 50000), desc="cluster predict"):
        idx = slice(start, min(start + 50000, n))
        block = cache.get(ids[idx].tolist())
        labels[idx] = km.predict(transform.apply(block, langs[idx]))
    return pd.DataFrame({"topic_cluster": labels}, index=df.index)


def report_statements(df, assigned):
    print("\n=== statement-similarity distribution (cosine, same-language) ===")
    print(assigned["statement_sim"].describe().to_string())
    print("  NOTE absolute sim is anisotropy-compressed (tight high band) -> use the MARGIN below")
    print("\n=== margin (top1 - top2) distribution -> the discriminative ruler ===")
    print(assigned["statement_margin"].describe().to_string())
    counts = assigned["statement_idx"].value_counts().sort_index()
    print(f"\n=== speeches per statement ({len(counts)}/30 statements populated) ===")
    print(counts.to_string())
    merged = df.join(assigned)
    med = merged.groupby("statement_idx")["statement_margin"].median()
    print("\n=== per-statement median margin (low -> that statement is topically ambiguous) ===")
    print(med.round(4).to_string())
    cell = merged.groupby(["statement_idx", PARTY_COLUMN]).size().unstack(fill_value=0)
    empty = (cell == 0).sum().sum()
    print(f"\n(party x statement) cells with zero speeches: {empty} of {cell.size} "
          f"-> these need prototype shrinkage downstream")


def report_clusters(df, assigned):
    merged = df.join(assigned)
    sizes = merged["topic_cluster"].value_counts().sort_index()
    print(f"\n=== cluster sizes (k={len(sizes)}) ===")
    print(sizes.describe().to_string())
    v_lang = v_measure_score(merged["language"], merged["topic_cluster"])
    v_party = v_measure_score(merged[PARTY_COLUMN], merged["topic_cluster"])
    print(f"\nV-measure cluster vs language: {v_lang:.4f}  (want LOW: clusters != languages)")
    print(f"V-measure cluster vs party   : {v_party:.4f}  (want LOW: clusters != parties)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,dev", help="comma-separated split names")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--topics", choices=["statements", "cluster", "date"], default="statements")
    parser.add_argument("--clusters", type=int, default=60, help="k for --topics cluster")
    parser.add_argument("--cluster-fit-sample", type=int, default=300000,
                        help="rows used to fit lang-center + KMeans (rest predicted in chunks)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="head N per party (debug)")
    args = parser.parse_args()

    embedder = None
    if args.topics in ("statements", "cluster"):
        embedder = Harrier(batch_size=args.batch_size, max_tokens=args.max_tokens)
    cache = EmbeddingCache(args.cache)
    statements = load_statements() if args.topics == "statements" else None

    for split in args.splits.split(","):
        df = load_split(split, args.data_dir)
        if args.limit is not None:
            df = df.groupby(PARTY_COLUMN, group_keys=False).head(args.limit).reset_index(drop=True)
        print(f"\n########## {split}: {len(df):,} speeches ##########")

        if args.topics == "statements":
            ensure_embedded(cache, embedder, df["text"].tolist())
            assigned = assign_statements(cache, df, embedder, statements)
            report_statements(df, assigned)
        elif args.topics == "cluster":
            ensure_embedded(cache, embedder, df["text"].tolist())
            assigned = assign_clusters(cache, df, args.clusters, fit_sample=args.cluster_fit_sample)
            report_clusters(df, assigned)
        else:  # date
            assigned = pd.DataFrame({"topic_date": df["date"].astype(str)}, index=df.index)
            print(f"\n=== distinct debate-days: {assigned['topic_date'].nunique()} ===")

        keep = ["id", "language", PARTY_COLUMN]
        out = df[keep].join(assigned)
        out_path = Path(args.data_dir) / f"topics_{split}_{args.topics}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

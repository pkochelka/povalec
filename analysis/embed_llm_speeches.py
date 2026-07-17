import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import (
    CACHE_DIR,
    REPO_ROOT,
    EmbeddingCache,
    FeatureTransform,
    Harrier,
    ensure_embedded,
    text_id,
    transform_split,
)

RESULTS_ROOT = REPO_ROOT / "data" / "euandi_2024_results"
GEMMA_TO_EP = {
    "bg": "bg", "cz": "cs", "dk": "da", "ee": "et", "fi": "fi", "fr": "fr",
    "de": "de", "gr": "el", "hu": "hu", "it": "it", "lv": "lv", "lt": "lt",
    "nl": "nl", "pl": "pl", "pt": "pt", "ro": "ro", "sk": "sk", "si": "sl",
    "es": "es", "se": "sv", "en": "en",
}
ANSWER_RE = re.compile(r"^answer_([a-z]{2})(?:_(?:negated|question))?_v(\d+)$")

def per_statement_report(df, parties, out_path=None):
    sim_cols = [f"sim_{party_token(p)}" for p in parties]

    # mean closeness to each party, per statement (pooled over language/variant/track)
    by_stmt = df.groupby("statement_idx")[sim_cols].mean()
    arr = by_stmt.to_numpy()
    by_stmt = by_stmt.assign(
        nearest=[parties[i] for i in arr.argmax(1)],
        margin=np.sort(arr, axis=1)[:, -1] - np.sort(arr, axis=1)[:, -2],  # top1 - top2
    )

    print("\n=== per-statement nearest EP prototype (pooled) ===")
    print(by_stmt[["nearest", "margin"]].to_string())
    print("\ndistinct nearest parties across statements:",
          by_stmt["nearest"].nunique(), "of", len(parties))
    print(by_stmt["nearest"].value_counts().to_string())

    # STANCE TEST: does negating the statement change its nearest party?
    if {"base", "negated"} <= set(df["track"].unique()):
        nb = (df[df.track.isin(["base", "negated"])]
              .groupby(["statement_idx", "track"])[sim_cols].mean())
        nb = nb.assign(nearest=[parties[i] for i in nb.to_numpy().argmax(1)]).reset_index()
        wide = nb.pivot(index="statement_idx", columns="track", values="nearest")
        same = (wide["base"] == wide["negated"]).mean()
        print("\n=== stance sensitivity (base vs negated nearest party) ===")
        print(wide.to_string())
        print(f"\nsame nearest party for base & negated: {same:.0%} of statements")

    if out_path:
        by_stmt.to_csv(out_path)
        print(f"\nSaved per-statement table to {out_path}")

def party_token(party):
    return party.replace("/", "_").replace("&", "_").replace(" ", "")


def discover_speech_files(model_dir):
    files = []
    for path in sorted(model_dir.glob("speeches_*.csv")):
        name = path.name
        if "_classified" in name or "_scored" in name:
            continue
        track = "negated" if "_negated" in name else "question" if "_question" in name else "base"
        files.append((track, path))
    return files


def extract_speeches(track, path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    rows = []
    for column in df.columns:
        match = ANSWER_RE.match(column)
        if not match:
            continue
        lang, variant = match.group(1), int(match.group(2))
        if lang not in GEMMA_TO_EP:
            continue
        for statement_idx, value in df[column].items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text or text.upper().startswith("FAILED"):
                continue
            rows.append({
                "track": track,
                "language": lang,
                "ep_language": GEMMA_TO_EP[lang],
                "statement_idx": int(statement_idx),
                "variant": f"v{variant}",
                "text": text,
                "id": text_id(text),
            })
    return rows


def load_reference(ep_cache, tag):
    prototypes_file = ep_cache / f"prototypes_{tag}.npy"
    parties_file = ep_cache / f"prototypes_parties_{tag}.json"
    transform_file = ep_cache / f"transform_{tag}.pkl"
    for path in (prototypes_file, parties_file, transform_file):
        if not path.exists():
            raise SystemExit(
                f"missing {path.name} in {ep_cache}; run party_prototype_embeddings.py "
                f"with --transform {tag} first"
            )
    prototypes = np.load(prototypes_file)
    parties = json.loads(parties_file.read_text(encoding="utf-8"))
    with open(transform_file, "rb") as handle:
        transform = pickle.load(handle)
    return prototypes, parties, transform


def negation_quality_check(df, projected):
    """
    For each statement, measure cosine distance between base and negated embeddings.
    
    High distance (close to 1.0) = good negation (LLM expressed opposite stance)
    Low distance (close to 0.0) = poor negation (LLM didn't flip semantically)
    """
    import numpy as np
    import pandas as pd
    
    if "base" not in df["track"].values or "negated" not in df["track"].values:
        print("\n[negation check] skipped: base and/or negated tracks missing")
        return None
    
    # Group by statement_idx, language, variant and pair base with negated
    pairs = []
    for (stmt, lang, var), group_base in df[df["track"] == "base"].groupby(["statement_idx", "language", "variant"]):
        group_neg = df[(df["statement_idx"] == stmt) & (df["language"] == lang) & 
                       (df["variant"] == var) & (df["track"] == "negated")]
        if len(group_base) > 0 and len(group_neg) > 0:
            pairs.append((stmt, lang, var, group_base.index[0], group_neg.index[0]))
    
    if not pairs:
        print("\n[negation check] no matching base/negated pairs")
        return None
    
    cosine_dists = []
    for stmt, lang, var, base_idx, neg_idx in pairs:
        base_emb = projected[base_idx]
        neg_emb = projected[neg_idx]
        # Cosine distance = 1 - cosine similarity (both are L2-normalized, so dot product = cosine)
        sim = np.dot(base_emb, neg_emb)
        dist = 1.0 - sim
        cosine_dists.append({
            "statement_idx": stmt,
            "language": lang,
            "variant": var,
            "cosine_similarity": float(sim),
            "cosine_distance": float(dist),
        })
    
    result_df = pd.DataFrame(cosine_dists)
    
    print("\n=== NEGATION QUALITY (cosine distance: base vs negated) ===")
    print(f"Total pairs: {len(result_df)}")
    print(f"Mean cosine distance: {result_df['cosine_distance'].mean():.4f}")
    print(f"  (0.0 = identical, 1.0 = opposite; expect >0.3 for good negation)")
    print(f"Median cosine distance: {result_df['cosine_distance'].median():.4f}")
    print(f"Std dev: {result_df['cosine_distance'].std():.4f}")
    print(f"\nQuartiles:")
    print(result_df['cosine_distance'].quantile([0.25, 0.5, 0.75]).to_string())
    
    print(f"\nBy statement:")
    by_stmt = result_df.groupby("statement_idx")["cosine_distance"].agg(["mean", "std", "count"])
    print(by_stmt.to_string())
    
    print(f"\nBy language:")
    by_lang = result_df.groupby("language")["cosine_distance"].agg(["mean", "std", "count"])
    print(by_lang.to_string())
    
    # Flag statements with very low negation distance
    low_neg = result_df[result_df["cosine_distance"] < 0.1]
    if len(low_neg) > 0:
        print(f"\n⚠️  {len(low_neg)} pairs with distance < 0.1 (poor negation):")
        print(low_neg[["statement_idx", "language", "cosine_distance"]].to_string())
    
    return result_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(RESULTS_ROOT / "kimi-k2.6"))
    parser.add_argument("--ep-cache", default=str(CACHE_DIR))
    parser.add_argument("--cache", default=None)
    parser.add_argument("--tag", default="lang_center")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tracks", default="base,negated")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name = model_dir.name
    cache_dir = Path(args.cache) if args.cache else CACHE_DIR / "llm" / model_name
    out_path = Path(args.out) if args.out else model_dir / f"speech_geometry_harrier_{args.tag}.parquet"

    prototypes, parties, transform = load_reference(Path(args.ep_cache), args.tag)

    speeches = []

    keep = set(args.tracks.split(","))
    for track, path in discover_speech_files(model_dir):
        if track not in keep:
            continue
        rows = extract_speeches(track, path)
        speeches.extend(rows)
        print(f"[{model_name}] {track:<8} {path.name}: {len(rows):,} speeches")
    if not speeches:
        raise SystemExit(f"no speeches found in {model_dir}")
    df = pd.DataFrame(speeches).reset_index(drop=True)
    print(f"[{model_name}] total {len(df):,} speeches over "
          f"{df['language'].nunique()} languages, {df['track'].nunique()} tracks")

    embedder = Harrier(batch_size=args.batch_size, max_tokens=args.max_tokens)
    cache = EmbeddingCache(cache_dir)
    ensure_embedded(cache, embedder, df["text"].tolist())

    
    llm_emb = cache.get(df["id"].tolist())
    # center LLM texts by their OWN per-language means -> removes language AND the LLM-domain offset
    llm_transform = FeatureTransform(kind="lang_center").fit(llm_emb, groups=df["ep_language"].to_numpy())
    projected = llm_transform.apply(llm_emb, df["ep_language"].to_numpy())
    similarities = projected @ prototypes.T   # prototypes stay as the EuroParl reference axes

    for index, party in enumerate(parties):
        df[f"sim_{party_token(party)}"] = similarities[:, index]

    
    negation_df = negation_quality_check(df, projected)
    if negation_df is not None:
        negation_df.to_csv(out_path.with_name(out_path.stem + "_negation_quality.csv"), index=False)
    nearest = similarities.argmax(1)
    ordered = np.sort(similarities, axis=1)
    df["nearest_party"] = [party_token(parties[i]) for i in nearest]
    df["nearest_sim"] = similarities[np.arange(len(df)), nearest]
    df["margin"] = ordered[:, -1] - ordered[:, -2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows to {out_path}")

    print("\n=== nearest EP party prototype (count) ===")
    print(df["nearest_party"].value_counts().to_string())
    print("\n=== mean cosine similarity to each prototype ===")
    for party in parties:
        print(f"  {party:<12} {df[f'sim_{party_token(party)}'].mean():.4f}")
    
    per_statement_report(df, parties, out_path=out_path.with_name(out_path.stem + "_per_statement.csv"))


if __name__ == "__main__":
    main()

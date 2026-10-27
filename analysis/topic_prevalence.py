#!/usr/bin/env python3
"""Per-cluster prevalence of the questionnaire topics in the classifier's training data.

Answers "has the classifier just learned which cluster talks about which topic?": if one
cluster owned a topic in the training speeches, an LLM text on that topic would lean to it
whatever it argues. Method (the paper's appendix on per-group topic prevalence):

  1. Embed every training speech with a multilingual sentence encoder (mean pooling,
     the encoder's own 128-token window, so a speech is its opening).
  2. Embed the 30 EU&I statements in every language; a speech is compared with the
     statement in its own language.
  3. Each statement retrieves its --k nearest speeches (cosine). A topic's set is the union
     over the statements coded non-zero on that axis in the questionnaire.
  4. Report each topic set's composition by cluster next to what the training file alone
     would give. The training file is not balanced by cluster, so the reference is the
     corpus share, taken within language and weighted by the set's language mix
     ("expected"); lift = share / expected, TVD = half the summed |share - expected|.

Embeddings are cached in --out_dir, so reruns with another --k skip the GPU.

Usage, from the repository root (GPU; ~1.3M speeches take a few minutes):
  python -m analysis.topic_prevalence
  python -m analysis.topic_prevalence --train "data/EuroParl Custom/clusters_k4_national_ni/train.parquet"
Output: results_topic_prevalence.txt (tables + LaTeX), <out_dir>/topic_prevalence_k<K>.csv,
        <out_dir>/topic_prevalence_statements_k<K>.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from analysis.core.questionnaire import load_axis_directions
from utils import AXES, PARTY_DISPLAY_ORDER, PARTY_SHORT, configure_stdout

configure_stdout()

PARTY = "EU Party"
DEFAULT_TRAIN = Path("data") / "EuroParl Custom" / "clusters_k4_national_ni_eval25k" / "train_langmatched_mt.parquet"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# EuroParl/ISO language code -> EU&I statement-file key; identity for the rest.
EP_TO_EUANDI = {"cs": "cz", "da": "dk", "el": "gr", "sv": "se", "sl": "si", "et": "ee"}
AXIS_SHORT = {"Ukraine": "UA", "Ecology": "Eco", "Immigration": "Imm", "Values": "Val",
              "Economy": "Econ", "Europe": "Eu", "Left-Right": "L-R"}
out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, max_length, batch_size, desc="embedding"):
    """L2-normalized mean-pooled embeddings, (len(texts), hidden) float32."""
    embeddings = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, mininterval=30):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            hidden = model(**inputs).last_hidden_state
        pooled = F.normalize(mean_pool(hidden, inputs["attention_mask"]), p=2, dim=-1)
        embeddings[start:start + len(batch)] = pooled.cpu().numpy()
    return embeddings


def load_speeches(path, max_chars, limit):
    """language, cluster and the opening of every speech (the encoder reads no further)."""
    table = pq.read_table(path, columns=["language", PARTY, "text"])
    if limit:  # the split files are sorted by cluster, so a head would hold one cluster
        table = table.take(np.sort(np.random.default_rng(0).choice(len(table), limit, replace=False)))
    text = pc.utf8_slice_codeunits(table["text"], 0, max_chars)
    df = table.drop_columns(["text"]).to_pandas()
    df["text"] = text.to_pandas().fillna("")
    return df


def load_statements(dataset):
    """Statements in statements.jsonl order ({language key: text} each) and their axis coding."""
    data_dir = Path("data") / f"{dataset}_data"
    with open(data_dir / "statements.jsonl", encoding="utf-8") as f:
        statements = [json.loads(line)["statement"] for line in f if line.strip()]
    # The questionnaire is in codebook order, statements.jsonl is not: join on the text.
    coding = load_axis_directions(str(data_dir / f"{dataset}_questionnaire.jsonl"),
                                  pd.Series([s["en"] for s in statements]))
    return statements, coding


def statement_similarities(emb, langs, statements, tokenizer, model, args):
    """(speeches, statements) cosine, each speech against the statement in its own language."""
    sims = np.zeros((len(emb), len(statements)), dtype=np.float32)
    for lang in sorted(set(langs)):
        key = EP_TO_EUANDI.get(lang, lang)
        missing = sum(key not in s for s in statements)
        if missing:
            say(f"  {lang}: {missing} statement(s) have no '{key}' text, English used instead")
        texts = [s.get(key) or s["en"] for s in statements]
        stmt_emb = embed_texts(texts, tokenizer, model, args.device, args.max_length, args.batch_size,
                               desc=f"statements {lang}")
        rows = np.flatnonzero(langs == lang)
        sims[rows] = emb[rows] @ stmt_emb.T
    return sims


def prevalence(sims, df, coding, clusters, by_language, k):
    """Topic-level and statement-level composition of the retrieved sets at one K."""
    k = min(k, len(sims))
    labels, langs = df[PARTY].to_numpy(), df["language"].to_numpy()
    top = [np.argpartition(-sims[:, j], k - 1)[:k] for j in range(sims.shape[1])]

    statement_rows = []
    for j, rows in enumerate(top):
        counts = pd.Series(labels[rows]).value_counts().reindex(clusters, fill_value=0)
        statement_rows.append({"statement": j, "kth_similarity": sims[rows, j].min(), **counts})

    topic_rows = []
    for axis in AXES:
        members = np.flatnonzero(coding[axis].to_numpy() != 0)
        rows = np.unique(np.concatenate([top[j] for j in members]))
        share = pd.Series(labels[rows]).value_counts(normalize=True).reindex(clusters, fill_value=0.0)
        lang_mix = pd.Series(langs[rows]).value_counts(normalize=True)
        expected = by_language.loc[lang_mix.index].mul(lang_mix, axis=0).sum()[clusters]
        for cluster in clusters:
            topic_rows.append({"k": k, "topic": axis, "statements": len(members), "speeches": len(rows),
                               "cluster": cluster, "share": share[cluster], "expected": expected[cluster],
                               "lift": share[cluster] / expected[cluster],
                               "tvd": 0.5 * (share - expected).abs().sum()})
    return pd.DataFrame(topic_rows), pd.DataFrame(statement_rows)


def wide(topics, value):
    return topics.pivot(index="cluster", columns="topic", values=value)[AXES]


def latex_table(topics, corpus, clusters, k, n_speeches, embed_model):
    share, lift = 100 * wide(topics, "share"), wide(topics, "lift")
    n_statements = topics.drop_duplicates("topic").set_index("topic")["statements"]
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{l" + "r" * (len(AXES) + 1) + "}", r"\toprule",
             "Cluster & Train & " + " & ".join(f"{AXIS_SHORT[a]}$^{{{n_statements[a]}}}$" for a in AXES) + r" \\",
             r"\midrule"]
    for cluster in clusters:
        cells = [f"\\textbf{{{share.loc[cluster, a]:.2f}}}" if lift[a].idxmax() == cluster
                 else f"{share.loc[cluster, a]:.2f}" for a in AXES]
        lines.append(f"{PARTY_SHORT.get(cluster, cluster)} & {100 * corpus[cluster]:.2f} & "
                     + " & ".join(cells) + r" \\")
    tvd = topics.drop_duplicates("topic").set_index("topic")["tvd"]
    n_tex = f"{n_speeches:,}".replace(",", "{,}")
    lines += [r"\midrule", "TVD &  & " + " & ".join(f"{tvd[a]:.3f}" for a in AXES) + r" \\",
              r"\bottomrule", r"\end{tabular}",
              r"\caption{Composition of each topic's retrieved speech set by cluster (\%, columns sum to 100). "
              f"Each EU\\&I statement retrieves its $K={k}$ nearest of the ${n_tex}$ training speeches "
              f"by cosine similarity of \\texttt{{{embed_model.split('/')[-1]}}} embeddings, in the speech's own "
              r"language; a topic's set is the union over its statements. Superscripts give the number of "
              r"statements per topic. \emph{Train} is each cluster's share of the training data; the cluster "
              r"most over-represented relative to the training data in the set's languages is in bold. TVD is "
              r"the total variation distance between a topic's composition and that reference.}",
              r"\label{tab:topic-shares}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="training parquet the classifier saw")
    parser.add_argument("--dataset", default="euandi_2024")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--k", type=int, default=2000, help="speeches retrieved per statement")
    parser.add_argument("--k_sweep", default="1000,5000", help="other K values, for the stability check")
    parser.add_argument("--max_length", type=int, default=128, help="tokens read per speech")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=Path, default=Path("data") / "topic_prevalence")
    parser.add_argument("--report", type=Path, default=Path("results_topic_prevalence.txt"))
    parser.add_argument("--limit", type=int, default=None, help="random N rows only (smoke test)")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # No sentencepiece token is longer than 16 characters, so nothing past this is ever read.
    df = load_speeches(args.train, 20 * args.max_length, args.limit)
    langs = df["language"].to_numpy()
    clusters = [c for c in PARTY_DISPLAY_ORDER if c in set(df[PARTY])] or sorted(df[PARTY].unique())
    corpus = df[PARTY].value_counts(normalize=True)[clusters]
    by_language = pd.crosstab(df["language"], df[PARTY], normalize="index")[clusters]
    say(f"Training file: {args.train}  ({len(df):,} speeches, {df['language'].nunique()} languages)")
    say(f"Encoder: {args.embed_model}, first {args.max_length} tokens of each speech")
    say("\nCluster shares of the training file (%), and their range over languages:")
    say(pd.DataFrame({"share": 100 * corpus, "min language": 100 * by_language.min(),
                      "max language": 100 * by_language.max()}).round(2).to_string())

    tokenizer = AutoTokenizer.from_pretrained(args.embed_model)
    model = AutoModel.from_pretrained(args.embed_model).to(args.device).eval()
    cache = args.out_dir / (f"emb_{args.train.parent.name}_{args.train.stem}_{len(df)}_"
                            f"{args.embed_model.split('/')[-1]}_len{args.max_length}.npy")
    if cache.exists():
        say(f"\nEmbeddings from cache: {cache}")
        emb = np.load(cache)
    else:
        emb = embed_texts(df["text"].tolist(), tokenizer, model, args.device, args.max_length, args.batch_size)
        np.save(cache, emb)
    df = df.drop(columns="text")

    statements, coding = load_statements(args.dataset)
    sims = statement_similarities(emb, langs, statements, tokenizer, model, args)

    ks = [args.k] + [int(k) for k in args.k_sweep.split(",") if k.strip() and int(k) != args.k]
    leaders = {}
    for k in ks:
        topics, per_statement = prevalence(sims, df, coding, clusters, by_language, k)
        per_statement.insert(1, "text", [s["en"] for s in statements])
        topics.to_csv(args.out_dir / f"topic_prevalence_k{k}.csv", index=False)
        per_statement.to_csv(args.out_dir / f"topic_prevalence_statements_k{k}.csv", index=False)
        lift = wide(topics, "lift").loc[clusters]
        leaders[k] = lift.idxmax()
        if k != args.k:
            continue

        sizes = topics.drop_duplicates("topic").set_index("topic").loc[AXES]
        say(f"\n=== K = {k} speeches per statement ===")
        say("\nTopic sets (statements, speeches in the union, TVD from expected):")
        say(sizes[["statements", "speeches", "tvd"]].round(3).T.to_string())
        say("\nComposition of each topic set by cluster (%, columns sum to 100):")
        say((100 * wide(topics, "share").loc[clusters]).round(2).to_string())
        say("\nExpected from the training file at the set's language mix (%):")
        say((100 * wide(topics, "expected").loc[clusters]).round(2).to_string())
        say("\nLift (share / expected; 1 = the topic is spread like the training data):")
        say(lift.round(2).to_string())
        say(f"\nLift range over all topic x cluster cells: {lift.min().min():.2f} - {lift.max().max():.2f}; "
            f"largest TVD {sizes['tvd'].max():.3f} ({sizes['tvd'].idxmax()})")
        counts = per_statement[clusters]
        statement_lift = counts.div(counts.sum(axis=1), axis=0) / corpus
        say("\nPer statement, lift against the training shares (K-th similarity = the weakest retrieved match):")
        say(pd.concat([per_statement[["kth_similarity"]].round(3), statement_lift.round(2),
                       per_statement["text"].str[:60]], axis=1).to_string())
        say("\nLaTeX:\n" + latex_table(topics, corpus, clusters, k, len(df), args.embed_model))

    say("\n=== Stability of the most over-represented cluster per topic ===")
    say(pd.DataFrame({f"K={k}": leaders[k].map(lambda c: PARTY_SHORT.get(c, c)) for k in ks}).to_string())
    changed = [a for a in AXES if len({leaders[k][a] for k in ks}) > 1]
    say("Leaders unchanged across K." if not changed else f"Leader changes with K on: {', '.join(changed)}")

    args.report.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nSaved report to {args.report}, tables to {args.out_dir}")


if __name__ == "__main__":
    main()

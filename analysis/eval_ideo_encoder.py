"""Use a fine-tuned IdeoEncoder: rebuild (party x statement) prototypes in the
learned space and measure the conditioned separation on dev, in BOTH distance
spaces, against the frozen baseline.

Baseline to beat (frozen Harrier, lang-centered, statement-conditioned):
    gap = own - other ~ 0.028 ,  top1 ~ 0.26   (chance ~ 0.14)

  metric A = backbone pooled, plain cosine
  metric B = projection head, learned metric

We don't re-encode all 1.7M speeches: prototypes are built from a per-cell sample
of train, evaluation from a sample of dev. Both run through the fine-tuned model.

Usage:
  python analysis/eval_ideo_encoder.py --model-dir models/harrier-ideo --tag final
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import DATA_DIR, PARTY_COLUMN, FeatureTransform, load_split
from statement_cutoff_sweep import build_prototypes, evaluate
from train_ideo_encoder import IdeoEncoder


class InMemoryCache:
    """cache.has/.get/.vectors over an id->row table, so we can reuse the sweep's
    build_prototypes/evaluate on fine-tuned (not on-disk) embeddings."""

    def __init__(self, ids, vectors):
        self.vectors = vectors
        self._pos = {i: k for k, i in enumerate(ids)}

    def has(self, i):
        return i in self._pos

    def get(self, ids):
        return self.vectors[[self._pos[i] for i in ids]]


def attach_text(topics_df, split, data_dir):
    text = load_split(split, data_dir)[["id", "text"]]
    return topics_df.merge(text, on="id", how="inner")


def sample_per_cell(df, per_cell, seed):
    # iterate groups (keeps all columns across pandas versions; groupby.apply on
    # newer pandas drops the grouping columns from the passed frame)
    parts = []
    for _, g in df.groupby(["statement_idx", PARTY_COLUMN]):
        parts.append(g.sample(per_cell, random_state=seed) if len(g) > per_cell else g)
    return pd.concat(parts)


def encode(encoder, texts, batch_size):
    a_all, b_all = [], []
    for start in tqdm(range(0, len(texts), batch_size), desc="encoding"):
        a, b = encoder.embed(texts[start:start + batch_size])
        a_all.append(a)
        b_all.append(b)
    return np.concatenate(a_all), np.concatenate(b_all)


def report_space(name, cache, proto_df, dev_df, parties):
    transform = FeatureTransform(kind="none").fit(np.zeros((1, cache.vectors.shape[1])))
    protos = build_prototypes(cache, proto_df, transform)
    m = evaluate(cache, dev_df, transform, protos, parties)
    print(f"  {name:<10} own={m['own']:.4f}  other={m['other']:.4f}  "
          f"gap={m['gap']:.4f}  top1={m['top1']:.4f}  (n_dev={m['n_dev']:,})")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/harrier-ideo")
    p.add_argument("--tag", default="final")
    p.add_argument("--train", default=str(DATA_DIR / "topics_train_statements.parquet"))
    p.add_argument("--dev", default=str(DATA_DIR / "topics_dev_statements.parquet"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--dev-split", default="dev")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--per-cell", type=int, default=300, help="train speeches per (statement,party) for prototypes")
    p.add_argument("--dev-sample", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    encoder = IdeoEncoder.load(args.model_dir, args.tag)
    parties = sorted(encoder.meta["parties"])

    proto_df = attach_text(pd.read_parquet(args.train), args.train_split, args.data_dir)
    dev_df = attach_text(pd.read_parquet(args.dev), args.dev_split, args.data_dir)
    proto_df = sample_per_cell(proto_df, args.per_cell, args.seed).reset_index(drop=True)
    if len(dev_df) > args.dev_sample:
        dev_df = dev_df.sample(args.dev_sample, random_state=args.seed).reset_index(drop=True)
    print(f"prototypes from {len(proto_df):,} train speeches | eval on {len(dev_df):,} dev speeches")

    # encode the union once, in both spaces
    union = pd.concat([proto_df[["id", "text"]], dev_df[["id", "text"]]]).drop_duplicates("id")
    a_vecs, b_vecs = encode(encoder, union["text"].tolist(), args.batch_size)
    ids = union["id"].tolist()
    cache_a = InMemoryCache(ids, a_vecs)
    cache_b = InMemoryCache(ids, b_vecs)

    print("\n=== fine-tuned, statement-conditioned separation on dev ===")
    print(f"  {'baseline':<10} own=0.2891  other=0.2609  gap=0.0282  top1=0.2623  (frozen, lang_center)")
    report_space("metric A", cache_a, proto_df, dev_df, parties)
    report_space("metric B", cache_b, proto_df, dev_df, parties)


if __name__ == "__main__":
    main()

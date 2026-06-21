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
    "de": "de", "gr": "el", "hu": "hu", "ie": "en", "it": "it", "lv": "lv",
    "lt": "lt", "nl": "nl", "pl": "pl", "pt": "pt", "ro": "ro", "sk": "sk",
    "si": "sl", "es": "es", "se": "sv", "en": "en",
}
ANSWER_RE = re.compile(r"^answer_([a-z]{2})(?:_(?:negated|question))?_v(\d+)$")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(RESULTS_ROOT / "gemma-4-31b-it"))
    parser.add_argument("--ep-cache", default=str(CACHE_DIR))
    parser.add_argument("--cache", default=None)
    parser.add_argument("--tag", default="lang_center")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_name = model_dir.name
    cache_dir = Path(args.cache) if args.cache else CACHE_DIR / "llm" / model_name
    out_path = Path(args.out) if args.out else model_dir / f"speech_geometry_harrier_{args.tag}.parquet"

    prototypes, parties, transform = load_reference(Path(args.ep_cache), args.tag)

    speeches = []
    for track, path in discover_speech_files(model_dir):
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

    projected = transform_split(cache, df, transform, "ep_language")
    similarities = projected @ prototypes.T
    for index, party in enumerate(parties):
        df[f"sim_{party_token(party)}"] = similarities[:, index]
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


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score, v_measure_score
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "EuroParl Custom"
CACHE_DIR = REPO_ROOT / "data" / "embeddings" / "harrier"
PARTY_COLUMN = "EU Party"
EMBED_DIM = 1024
MAX_TOKENS = 1024
BATCH_SIZE = 8
FLUSH_EVERY = 50000
SILHOUETTE_SAMPLE = 10000
INSTRUCTION = "Represent this European Parliament speech for retrieving its political party group."
PARTY_COLOR = {
    "GUE/NGL": "#8b0000", "S&D": "#e8112d", "Greens/EFA": "#3eb049", "ALDE": "#f6b40e",
    "PPE": "#3a86c8", "ECR": "#0a4ea3", "ID": "#1b1f3b",
}


def text_id(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def with_instruction(text):
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def load_split(split, data_dir=DATA_DIR):
    df = pd.read_parquet(Path(data_dir) / f"{split}.parquet")
    df = df.dropna(subset=[PARTY_COLUMN, "text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["id"] = df["text"].map(text_id)
    return df.reset_index(drop=True)


class Harrier:
    def __init__(self, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS):
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()

    @torch.no_grad()
    def encode_batch(self, texts):
        tokens = self.tokenizer(
            [with_instruction(t) for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.model(**tokens).last_hidden_state
        last = tokens["attention_mask"].sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=self.device), last]
        pooled = normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float16)


class EmbeddingCache:
    def __init__(self, directory=CACHE_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        vectors = []
        self.n_shards = 0
        for ids_file in sorted(self.dir.glob("shard_*_ids.npy")):
            shard_ids = np.load(ids_file, allow_pickle=True)
            shard_vecs = np.load(str(ids_file).replace("_ids.npy", "_vecs.npy"))
            base = len(self.index)
            for offset, sid in enumerate(shard_ids):
                self.index[sid] = base + offset
            vectors.append(shard_vecs)
            self.n_shards += 1
        self.vectors = np.concatenate(vectors) if vectors else np.zeros((0, EMBED_DIM), np.float16)

    def has(self, sid):
        return sid in self.index

    def get(self, ids):
        return self.vectors[[self.index[i] for i in ids]]

    def write_shard(self, ids, vecs):
        if not ids:
            return
        path = self.dir / f"shard_{self.n_shards:05d}"
        np.save(f"{path}_ids.npy", np.array(ids, dtype=object))
        np.save(f"{path}_vecs.npy", vecs)
        base = len(self.index)
        for offset, sid in enumerate(ids):
            self.index[sid] = base + offset
        self.vectors = np.concatenate([self.vectors, vecs])
        self.n_shards += 1


def ensure_embedded(cache, embedder, texts):
    seen = set()
    todo = []
    for text in texts:
        sid = text_id(text)
        if cache.has(sid) or sid in seen:
            continue
        seen.add(sid)
        todo.append((sid, text))
    if not todo:
        return
    buffer_ids, buffer_vecs = [], []
    total_batches = (len(todo) + embedder.batch_size - 1) // embedder.batch_size
    for batch in tqdm(chunks(todo, embedder.batch_size), total=total_batches, desc="embedding"):
        ids = [sid for sid, _ in batch]
        vecs = embedder.encode_batch([text for _, text in batch])
        buffer_ids.extend(ids)
        buffer_vecs.append(vecs)
        if len(buffer_ids) >= FLUSH_EVERY:
            cache.write_shard(buffer_ids, np.concatenate(buffer_vecs))
            buffer_ids, buffer_vecs = [], []
    if buffer_ids:
        cache.write_shard(buffer_ids, np.concatenate(buffer_vecs))


def build_prototypes(cache, df):
    parties = sorted(df[PARTY_COLUMN].unique())
    sums = {p: np.zeros(EMBED_DIM, np.float64) for p in parties}
    counts = {p: 0 for p in parties}
    for rows in tqdm(list(chunks(df.index.tolist(), 8192)), desc="prototypes"):
        block = df.loc[rows]
        vecs = cache.get(block["id"].tolist()).astype(np.float64)
        labels = block[PARTY_COLUMN].to_numpy()
        for party in parties:
            mask = labels == party
            if mask.any():
                sums[party] += vecs[mask].sum(0)
                counts[party] += int(mask.sum())
    matrix = np.stack([sums[p] / counts[p] for p in parties])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return parties, matrix.astype(np.float32)


def evaluate(cache, df, parties, prototypes):
    party_to_idx = {p: i for i, p in enumerate(parties)}
    embeddings = cache.get(df["id"].tolist()).astype(np.float32)
    truth = df[PARTY_COLUMN].map(party_to_idx).to_numpy()
    predictions = (embeddings @ prototypes.T).argmax(1)
    overall = float((predictions == truth).mean())
    per_class = {}
    for index, party in enumerate(parties):
        mask = truth == index
        per_class[party] = float((predictions[mask] == index).mean()) if mask.any() else float("nan")
    macro = float(np.nanmean(list(per_class.values())))
    vmeasure = float(v_measure_score(truth, predictions))
    sample = min(SILHOUETTE_SAMPLE, len(embeddings))
    silhouette = float(silhouette_score(embeddings, truth, metric="cosine", sample_size=sample, random_state=0))

    distances = 1.0 - (embeddings @ prototypes.T)
    distance_matrix = np.zeros((len(parties), len(parties)))
    for index in range(len(parties)):
        mask = truth == index
        distance_matrix[index] = distances[mask].mean(0) if mask.any() else np.nan

    return {
        "overall_top1": overall,
        "macro_top1": macro,
        "per_class_top1": per_class,
        "v_measure": vmeasure,
        "silhouette": silhouette,
        "distance_to_prototype": {
            parties[i]: {parties[j]: float(distance_matrix[i, j]) for j in range(len(parties))}
            for i in range(len(parties))
        },
    }


def report(parties, prototypes, metrics):
    print("\n=== NEAREST-PROTOTYPE GEOMETRY (dev, cosine) ===")
    print(f"{'party':<12}{'support_top1':>14}")
    print("-" * 26)
    for party in parties:
        print(f"{party:<12}{metrics['per_class_top1'][party]:>14.4f}")
    print("-" * 26)
    print(f"{'MACRO':<12}{metrics['macro_top1']:>14.4f}")
    print(f"{'OVERALL':<12}{metrics['overall_top1']:>14.4f}")
    print(f"\nV-measure : {metrics['v_measure']:.4f}")
    print(f"Silhouette: {metrics['silhouette']:.4f}  (cosine, sampled)")

    print("\n=== MEAN COSINE DISTANCE: party speeches (rows) -> prototypes (cols) ===")
    distance = metrics["distance_to_prototype"]
    header = "".join(f"{p[:9]:>10}" for p in parties)
    print(f"{'true \\ proto':<12}{header}")
    for true_party in parties:
        own = distance[true_party][true_party]
        others = {p: d for p, d in distance[true_party].items() if p != true_party}
        nearest_other = min(others, key=others.get)
        row = "".join(f"{distance[true_party][p]:>10.4f}" for p in parties)
        flag = "  <- own NOT nearest" if own > others[nearest_other] else ""
        print(f"{true_party:<12}{row}    own={own:.4f} nearest_other={nearest_other}({others[nearest_other]:.4f}){flag}")


def reduce_2d(method, vectors):
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(vectors)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(5, len(vectors) // 4))
        return TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity, random_state=0).fit_transform(vectors)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, metric="cosine", random_state=0).fit_transform(vectors)
    raise ValueError(f"unknown method: {method}")


def plot_embeddings_2d(cache, df, parties, prototypes, method, per_party, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    rows = []
    for party in parties:
        party_rows = df.index[df[PARTY_COLUMN] == party].to_numpy()
        if len(party_rows) > per_party:
            party_rows = rng.choice(party_rows, per_party, replace=False)
        rows.extend(party_rows.tolist())
    sample = df.loc[rows]
    embeddings = cache.get(sample["id"].tolist()).astype(np.float32)
    coords = reduce_2d(method, np.vstack([embeddings, prototypes]))
    points, proto_points = coords[: len(embeddings)], coords[len(embeddings):]

    labels = sample[PARTY_COLUMN].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    for party in parties:
        mask = labels == party
        ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.45,
                   color=PARTY_COLOR.get(party, "#888888"), label=party, linewidths=0)
    for index, party in enumerate(parties):
        ax.scatter(proto_points[index, 0], proto_points[index, 1], marker="X", s=280,
                   color=PARTY_COLOR.get(party, "#888888"), edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(party, proto_points[index], fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)
    ax.set_title(f"Harrier dev embeddings ({method.upper()}, n={len(embeddings)}, X = prototype)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved 2D projection to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plot-method", choices=["pca", "tsne", "umap"], default="tsne")
    parser.add_argument("--plot-per-party", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    embedder = Harrier(batch_size=args.batch_size, max_tokens=args.max_tokens)
    cache = EmbeddingCache(args.cache)

    train = load_split("train", args.data_dir)
    dev = load_split("dev", args.data_dir)
    if args.limit is not None:
        train = train.groupby(PARTY_COLUMN, group_keys=False).head(args.limit).reset_index(drop=True)
        dev = dev.groupby(PARTY_COLUMN, group_keys=False).head(args.limit).reset_index(drop=True)

    ensure_embedded(cache, embedder, train["text"].tolist())
    ensure_embedded(cache, embedder, dev["text"].tolist())

    parties, prototypes = build_prototypes(cache, train)
    np.save(cache.dir / "prototypes.npy", prototypes)
    (cache.dir / "prototypes_parties.json").write_text(json.dumps(parties, ensure_ascii=False, indent=2))

    metrics = evaluate(cache, dev, parties, prototypes)
    (cache.dir / "dev_geometry.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    report(parties, prototypes, metrics)

    if not args.no_plot:
        out_path = cache.dir / f"dev_embeddings_{args.plot_method}.png"
        plot_embeddings_2d(cache, dev, parties, prototypes, args.plot_method, args.plot_per_party, out_path)


if __name__ == "__main__":
    main()

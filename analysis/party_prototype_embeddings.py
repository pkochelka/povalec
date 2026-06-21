import argparse
import hashlib
import json
import pickle
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
BATCH_SIZE = 32
FLUSH_EVERY = 50000
SILHOUETTE_SAMPLE = 10000
INSTRUCTION = "Represent this European Parliament speech for retrieving its political party group."
PARTY_COLOR = {
    "GUE/NGL": "#8b0000", "S&D": "#e8112d", "Greens/EFA": "#3eb049", "ALDE": "#f6b40e",
    "PPE": "#3a86c8", "ECR": "#0a4ea3", "ID": "#1b1f3b",
}
LANG_COLUMN_CANDIDATES = ["language"]


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
        mask = tokens["attention_mask"]
        # Robust last-token pool: works for either padding side.
        if int(mask[:, -1].sum()) == mask.shape[0]:          # left-padded
            pooled = hidden[:, -1]
        else:                                                # right-padded
            idx = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=self.device), idx]
        pooled = normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


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
        self.vectors = (
            np.concatenate(vectors).astype(np.float32)
            if vectors else np.zeros((0, EMBED_DIM), np.float32)
        )

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


# --------------------------------------------------------------------------- #
# Feature transforms: fit on TRAIN only, apply identically to dev / LLM speech #
# --------------------------------------------------------------------------- #
class FeatureTransform:
    """Post-hoc transform of frozen embeddings.

    kind:
      none        -> raw embeddings (re-normalized; reproduces the baseline)
      center       -> subtract train mean, L2-normalize (kills the shared component)
      center_pca   -> center, then remove the top `remove_pcs` principal directions
                      ("all-but-the-top"), L2-normalize
      lda          -> center, then project onto the (n_parties-1) Fisher discriminant
                      axes that maximize between-party / within-party variance

    Fit once on train; persist; reuse the SAME object on dev and on the LLM speeches.
    """

    def __init__(self, kind="none", remove_pcs=1):
        self.kind = kind
        self.remove_pcs = remove_pcs
        self.mu = None
        self.pcs = None
        self.lda = None
        self.group_means = None      # for lang_center: per-language mean vector
        self.out_dim = EMBED_DIM

    def fit(self, X, y=None, groups=None):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(0, keepdims=True)
        Xc = X - self.mu
        if self.kind == "none":
            self.out_dim = X.shape[1]
        elif self.kind == "center":
            self.out_dim = X.shape[1]
        elif self.kind == "lang_center":
            if groups is None:
                raise ValueError("transform=lang_center requires a language column (--lang-column)")
            groups = np.asarray(groups)
            self.group_means = {
                g: X[groups == g].mean(0, keepdims=True) for g in np.unique(groups)
            }
            self.out_dim = X.shape[1]
        elif self.kind == "center_pca":
            from sklearn.decomposition import PCA
            k = max(1, self.remove_pcs)
            self.pcs = PCA(n_components=k, random_state=0).fit(Xc).components_  # (k, d)
            self.out_dim = X.shape[1]
        elif self.kind == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            if y is None:
                raise ValueError("transform=lda requires labels")
            n_comp = len(np.unique(y)) - 1
            self.lda = LinearDiscriminantAnalysis(n_components=n_comp).fit(Xc, y)
            self.out_dim = n_comp
        else:
            raise ValueError(f"unknown transform: {self.kind}")
        return self

    def apply(self, X, groups=None):
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "none":
            Z = X
        elif self.kind == "lang_center":
            if groups is None:
                raise ValueError("transform=lang_center requires groups at apply time")
            groups = np.asarray(groups)
            Z = X.copy()
            for g in np.unique(groups):
                m = self.group_means.get(g, self.mu)   # unseen language -> global mean
                Z[groups == g] -= m
        else:
            Z = X - self.mu
            if self.kind == "center_pca" and self.pcs is not None:
                Z = Z - (Z @ self.pcs.T) @ self.pcs       # strip top PCs
            elif self.kind == "lda":
                Z = self.lda.transform(Z)                  # -> (n, n_parties-1)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (Z / norms).astype(np.float32)              # unit norm -> dot product == cosine


def fit_transform_on_train(cache, train_df, kind, remove_pcs, fit_sample, lang_column=None, seed=0):
    sub = train_df
    if fit_sample and len(train_df) > fit_sample:
        # stratified-ish subsample to keep small parties represented
        per = max(1, fit_sample // train_df[PARTY_COLUMN].nunique())
        sel = (
            train_df.groupby(PARTY_COLUMN, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per), random_state=seed))
            .index.to_numpy()
        )
        sub = train_df.loc[sel]
    X = cache.get(sub["id"].tolist())
    labels = sub[PARTY_COLUMN].to_numpy()
    groups = sub[lang_column].to_numpy() if (kind == "lang_center" and lang_column) else None
    print(f"[transform] fitting '{kind}' on {len(sub):,} train rows")
    return FeatureTransform(kind=kind, remove_pcs=remove_pcs).fit(X, labels, groups)


def transform_split(cache, df, transform, lang_column=None):
    out = []
    for rows in chunks(df.index.tolist(), 8192):
        block = df.loc[rows]
        groups = block[lang_column].to_numpy() if (transform.kind == "lang_center" and lang_column) else None
        out.append(transform.apply(cache.get(block["id"].tolist()), groups))
    return np.concatenate(out)


def build_prototypes(cache, df, transform, lang_column=None):
    parties = sorted(df[PARTY_COLUMN].unique())
    sums = {p: np.zeros(transform.out_dim, np.float64) for p in parties}
    counts = {p: 0 for p in parties}
    for rows in tqdm(list(chunks(df.index.tolist(), 8192)), desc="prototypes"):
        block = df.loc[rows]
        groups = block[lang_column].to_numpy() if (transform.kind == "lang_center" and lang_column) else None
        proj = transform.apply(cache.get(block["id"].tolist()), groups).astype(np.float64)
        labels = block[PARTY_COLUMN].to_numpy()
        for party in parties:
            mask = labels == party
            if mask.any():
                sums[party] += proj[mask].sum(0)
                counts[party] += int(mask.sum())
    matrix = np.stack([sums[p] / counts[p] for p in parties])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return parties, matrix.astype(np.float32)


def evaluate(embeddings, df, parties, prototypes):
    party_to_idx = {p: i for i, p in enumerate(parties)}
    embeddings = embeddings.astype(np.float32)
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


def language_diagnostic(embeddings, df, parties, lang_column, sample=20000, seed=0):
    """Is the space organized by language or by party? Cluster, then score against each."""
    if lang_column is None or lang_column not in df.columns:
        print("\n[lang diagnostic] no language column found -> skipped "
              "(pass --lang-column to enable)")
        return None
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx = rng.choice(n, min(sample, n), replace=False)
    X = embeddings[idx].astype(np.float32)
    langs = df.iloc[idx][lang_column].astype(str).to_numpy()
    party = df.iloc[idx][PARTY_COLUMN].astype(str).to_numpy()

    n_lang = len(np.unique(langs))
    n_party = len(parties)
    km_lang = MiniBatchKMeans(n_clusters=max(2, n_lang), random_state=seed, n_init=3).fit_predict(X)
    km_party = MiniBatchKMeans(n_clusters=max(2, n_party), random_state=seed, n_init=3).fit_predict(X)
    v_lang = float(v_measure_score(langs, km_lang))
    v_party = float(v_measure_score(party, km_party))

    print("\n=== LANGUAGE vs PARTY STRUCTURE (transformed dev, KMeans V-measure) ===")
    print(f"  V-measure vs LANGUAGE (k={max(2, n_lang)}): {v_lang:.4f}   over {n_lang} languages")
    print(f"  V-measure vs PARTY    (k={max(2, n_party)}): {v_party:.4f}   over {n_party} parties")
    if v_lang > 2 * max(v_party, 1e-6):
        print("  -> language dominates the geometry. Build per-language prototypes, or "
              "restrict to the language of your LLM speeches before comparing.")
    elif v_lang > v_party:
        print("  -> language is the stronger axis; per-language prototypes likely help.")
    else:
        print("  -> party is at least as strong as language; a global prototype is reasonable.")
    return {"v_language": v_lang, "v_party": v_party, "n_languages": n_lang}


def report(parties, prototypes, metrics, transform_tag):
    print(f"\n=== NEAREST-PROTOTYPE GEOMETRY (dev, cosine, transform={transform_tag}) ===")
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
    print(f"{'true // proto':<12}{header}")
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


def plot_embeddings_2d(embeddings, df, parties, prototypes, method, per_party, out_path, transform_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    pos = {sid: i for i, sid in enumerate(df["id"].tolist())}
    rows = []
    for party in parties:
        party_ids = df["id"][df[PARTY_COLUMN] == party].to_numpy()
        if len(party_ids) > per_party:
            party_ids = rng.choice(party_ids, per_party, replace=False)
        rows.extend(party_ids.tolist())
    sel = [pos[s] for s in rows]
    emb = embeddings[sel]
    labels = df.set_index("id").loc[rows, PARTY_COLUMN].to_numpy()

    coords = reduce_2d(method, np.vstack([emb, prototypes]))
    points, proto_points = coords[: len(emb)], coords[len(emb):]

    fig, ax = plt.subplots(figsize=(11, 9))
    for party in parties:
        mask = labels == party
        ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.45,
                   color=PARTY_COLOR.get(party, "#888888"), label=party, linewidths=0)
    for index, party in enumerate(parties):
        ax.scatter(proto_points[index, 0], proto_points[index, 1], marker="X", s=280,
                   color=PARTY_COLOR.get(party, "#888888"), edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(party, proto_points[index], fontsize=9, fontweight="bold", ha="center", va="center", zorder=6)
    ax.set_title(f"Harrier dev ({method.upper()}, transform={transform_tag}, n={len(emb)}, X = prototype)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved 2D projection to {out_path}")


def detect_lang_column(df, explicit):
    if explicit:
        return explicit
    for candidate in LANG_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--transform", choices=["none", "center", "center_pca", "lda", "lang_center"], default="none")
    parser.add_argument("--remove-pcs", type=int, default=1, help="PCs to strip for center_pca")
    parser.add_argument("--transform-fit-sample", type=int, default=200000,
                        help="cap rows used to fit the transform (0 = use all)")
    parser.add_argument("--lang-column", default=None, help="column with language label (diagnostic / lang_center / --language)")
    parser.add_argument("--language", default=None, help="restrict train+dev to a single language value in the lang column")
    parser.add_argument("--plot-method", choices=["pca", "tsne", "umap"], default="tsne")
    parser.add_argument("--plot-per-party", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    embedder = Harrier(batch_size=args.batch_size, max_tokens=args.max_tokens)
    print(f"[tokenizer] padding_side = {embedder.tokenizer.padding_side}")
    cache = EmbeddingCache(args.cache)

    train = load_split("train_post2009", args.data_dir)
    dev = load_split("dev_post2009", args.data_dir)
    if args.limit is not None:
        train = train.groupby(PARTY_COLUMN, group_keys=False).head(args.limit).reset_index(drop=True)
        dev = dev.groupby(PARTY_COLUMN, group_keys=False).head(args.limit).reset_index(drop=True)

    lang_column = detect_lang_column(dev, args.lang_column)
    if args.language is not None:
        if lang_column is None:
            raise SystemExit("--language given but no language column found; pass --lang-column")
        train = train[train[lang_column] == args.language].reset_index(drop=True)
        dev = dev[dev[lang_column] == args.language].reset_index(drop=True)
        print(f"[language] restricted to '{args.language}': train={len(train):,} dev={len(dev):,}")
        if len(train) == 0 or len(dev) == 0:
            raise SystemExit(f"no rows for language '{args.language}' (check the value in column '{lang_column}')")
    if args.transform == "lang_center" and lang_column is None:
        raise SystemExit("--transform lang_center needs a language column; pass --lang-column")

    ensure_embedded(cache, embedder, train["text"].tolist())
    ensure_embedded(cache, embedder, dev["text"].tolist())

    # Fit the chosen transform on TRAIN, persist it for reuse on the LLM speeches.
    transform = fit_transform_on_train(
        cache, train, args.transform, args.remove_pcs,
        None if args.transform_fit_sample == 0 else args.transform_fit_sample,
        lang_column=lang_column,
    )
    tag = args.transform if args.transform != "center_pca" else f"center_pca{args.remove_pcs}"
    if args.language:
        tag = f"{tag}_{args.language}"
    with open(cache.dir / f"transform_{tag}.pkl", "wb") as fh:
        pickle.dump(transform, fh)

    parties, prototypes = build_prototypes(cache, train, transform, lang_column)
    np.save(cache.dir / f"prototypes_{tag}.npy", prototypes)
    (cache.dir / f"prototypes_parties_{tag}.json").write_text(json.dumps(parties, ensure_ascii=False, indent=2))

    dev_proj = transform_split(cache, dev, transform, lang_column)
    metrics = evaluate(dev_proj, dev, parties, prototypes)
    (cache.dir / f"dev_geometry_{tag}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    report(parties, prototypes, metrics, tag)

    language_diagnostic(dev_proj, dev, parties, None if args.language else lang_column)

    if not args.no_plot:
        out_path = cache.dir / f"dev_embeddings_{args.plot_method}_{tag}.png"
        plot_embeddings_2d(dev_proj, dev, parties, prototypes, args.plot_method, args.plot_per_party, out_path, tag)


if __name__ == "__main__":
    main()
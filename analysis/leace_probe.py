#!/usr/bin/env python3
"""Does linearly erasing native-vs-translated origin stop the Slovak Sov-right shortcut?

The national-k4 classifier labels AI-written Slovak ~99% Sovereigntist right. The cause is
traced to origin, not language: original Slovak EP text comes only from Slovak MEPs (99%
Sov-right), translated Slovak carries the EU-wide mix, and AI or machine-translated text
reads as original (analysis/translation_probe.py). This probe asks, without retraining,
whether that cue is linear in the representation the classifier reads.

LEACE (Belrose et al. 2023, closed form, numpy) is fit at the input of the final linear
layer (after ModernBERT's mean pooling and dense/GELU/LayerNorm head), so a linear head
refit on the erased features provably cannot read the erased concept. Concepts:

  language            one-hot language
  origin              original vs translated, pooled over languages
  origin x language   one indicator per language: original text in that language
  cells               one-hot (language, origin)
  origin within lang  a separate origin eraser per language (needs the text's language)

Each eraser is scored with the current head applied to erased features, and with a
class-balanced logistic head refit on them. "none / refit" is the refit control.
Labels are raw argmax (no manifest biases) except the "current + biases" reference row.

Stages:
  embed  (GPU, analysis/slurm_leace_probe.sh) writes features + raw logits to
         data/leace_probe/<run>/ for:
           fit   train sample, --n_fit per (language, origin, cluster); MT-augmentation
                 source rows left out
           test  test sample, --n_test per (language, origin, cluster)
           cf    data/translation_probe/counterfactual.csv, original and MT text
           mt    MT-augmentation translations so far (sources are train rows: the model
                 saw the source text, but never in this language)
           ai    LLM texts from --results_dir, the same --n_ai (statement, framing,
                 paraphrase) keys per (model, track) in every --ai_languages language,
                 so each text has an English partner
  fit    (CPU, seconds to minutes) fits the erasers and heads, writes
         results_leace_probe.txt / .csv

Metrics: leak = balanced accuracy of a per-language origin probe, 5-fold cross-validated on
the erased test set (0.5 = erased), linear and one-hidden-layer MLP (LEACE only promises the
linear one); ai->orig = share of AI text that probe calls original;
F1 on test (all / translated / original cells); native sk and ro accuracy; counterfactual
cs/pl -> sk and it/es/fr/pt -> ro accuracy after MT; MT-augmentation accuracy; AI-text
shares and agreement with the English partner text.

Usage, from the repository root:
  sbatch analysis/slurm_leace_probe.sh                 # embed + fit on the cluster
  python analysis/leace_probe.py --stage fit           # refit locally after copying data/leace_probe/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.translation_probe import GROUPS, SHORT, load_ai
from preprocessing.build_language_matched_split import origin_of
from preprocessing.build_mt_augmentation import read_texts
from utils import configure_stdout

configure_stdout()

DEFAULT_CLASSIFIER = "models/national-k4-logitadj_ni/model"
DEFAULT_DATA = "data/EuroParl Custom/clusters_k4_national_ni"
DEFAULT_RESULTS = "data/euandi_2024_results_new_ni_fixed_center"
DEFAULT_CF = "data/translation_probe/counterfactual.csv"
OUT_ROOT = Path("data/leace_probe")
PARTY = "EU Party"
SETS = ["fit", "test", "cf", "mt", "ai"]
AI_KEY = ["model", "track", "framing", "paraphrase", "statement_idx"]
MIN_ORIGIN_ROWS = 30   # per language and origin, to fit or score an origin eraser / probe

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


# --------------------------------------------------------------------------- #
# embed (GPU)
# --------------------------------------------------------------------------- #

def sample_split(path, n, seed, exclude=()):
    """n rows per (language, origin, cluster), with text; origin as in the matched split."""
    keys = pd.read_parquet(path, columns=["date", PARTY, "language", "speaker"])
    keys["origin"] = origin_of(keys, 2)
    keys.index.name = "row"
    keys = keys.reset_index()
    keys = keys[keys[PARTY].isin(SHORT) & ~keys.row.isin(set(exclude))]
    pick = keys.sample(frac=1, random_state=seed).groupby(["language", "origin", PARTY]).head(n).copy()
    pick["text"] = pick.row.map(read_texts(path, pick.row))
    pick = pick[pick.text.fillna("").str.strip().str.len() > 0]
    pick["gold"] = pick[PARTY].map(SHORT)
    return pick.drop(columns=["date", "speaker", PARTY]).reset_index(drop=True)


def load_mt(data_dir, n, seed):
    """MT-augmentation translations so far, n per (target, cluster); also the source rows."""
    out_dir = Path(data_dir) / "mt_augmentation"
    if not (out_dir / "translations.jsonl").exists():
        print(f"  no {out_dir / 'translations.jsonl'}: skipping the mt set")
        return None, []
    plan = pd.read_parquet(out_dir / "plan.parquet", columns=["target", PARTY, "row", "source_language"])
    recs = [json.loads(line) for line in (out_dir / "translations.jsonl").read_text(encoding="utf-8").splitlines()]
    done = pd.DataFrame(recs).drop_duplicates(["target", "row"])
    mt = plan.merge(done, on=["target", "row"])
    exclude = mt.row.unique()
    mt = mt.sample(frac=1, random_state=seed).groupby(["target", PARTY]).head(n)
    mt = pd.DataFrame({"row": mt.row, "language": mt.target, "source_language": mt.source_language,
                       "origin": "mt", "gold": mt[PARTY].map(SHORT), "text": mt.text})
    return mt.reset_index(drop=True), exclude


def load_cf(path):
    cf = pd.read_csv(path)
    orig = pd.DataFrame({"set": cf.set, "version": "orig", "language": cf.language, "gold": cf.gold,
                         "origin": "original", "text": cf.text})
    mt = pd.DataFrame({"set": cf.set, "version": "mt", "language": cf.tgt, "gold": cf.gold,
                       "origin": "mt", "text": cf.text_mt})
    out = pd.concat([orig, mt], ignore_index=True)
    return out[out.text.notna()].reset_index(drop=True)


def load_ai_sample(args, labels):
    from analysis.analyze_all import MODEL_DIRS
    ai = load_ai(Path(args.results_dir), list(MODEL_DIRS), args.ai_languages, labels)
    keys = (ai[AI_KEY].drop_duplicates().sample(frac=1, random_state=args.seed)
            .groupby(["model", "track"]).head(args.n_ai))
    ai = ai.merge(keys, on=AI_KEY).rename(columns={"pred": "stored_pred"})
    ai["origin"], ai["gold"] = "ai", None
    return ai.reset_index(drop=True)


def embed_texts(texts, clf, device, batch):
    """Input of the final linear layer, and the raw logits, for each text."""
    import torch
    from tqdm import tqdm
    from analysis.classify_speeches import predict_class_logits
    layer = clf.model.classifier
    captured = []
    hook = layer.register_forward_hook(lambda module, inp, out: captured.append(inp[0].float().cpu().numpy()))
    feats = np.zeros((len(texts), layer.in_features), dtype=np.float32)
    logits = np.zeros((len(texts), layer.out_features), dtype=np.float32)
    order = np.argsort([-len(t) for t in texts], kind="stable")  # longest first: an OOM shows up at once
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch)):
                rows = order[start:start + batch]
                captured.clear()
                logits[rows] = predict_class_logits([texts[i] for i in rows], clf, device)
                feats[rows] = captured[0]
    finally:
        hook.remove()
    return feats, logits


def embed(args, out_dir):
    import torch
    from analysis.classify_speeches import load_classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = load_classifier(args.classifier, device)
    print(f"classifier {args.classifier} on {device}; labels {clf.labels}")
    layer = clf.model.classifier
    np.savez(out_dir / "head.npz", W=layer.weight.detach().float().cpu().numpy(),
             b=(layer.bias.detach().float().cpu().numpy() if layer.bias is not None
                else np.zeros(layer.out_features, dtype=np.float32)),
             biases=clf.biases if clf.biases is not None else np.zeros(layer.out_features, dtype=np.float32),
             labels=np.array([SHORT[l] for l in clf.labels]), classifier=args.classifier)

    mt, mt_rows = load_mt(args.data_dir, args.n_mt, args.seed)
    frames = {
        "fit": lambda: sample_split(Path(args.data_dir) / "train.parquet", args.n_fit, args.seed, exclude=mt_rows),
        "test": lambda: sample_split(Path(args.data_dir) / "test.parquet", args.n_test, args.seed),
        "cf": lambda: load_cf(args.cf) if Path(args.cf).exists() else None,
        "mt": lambda: mt,
        "ai": lambda: load_ai_sample(args, clf.labels) if Path(args.results_dir).exists() else None,
    }
    for name in args.sets:
        df = frames[name]()
        if df is None or df.empty:
            print(f"[{name}] no rows, skipped")
            continue
        print(f"[{name}] {len(df)} texts")
        feats, logits = embed_texts(df.text.astype(str).tolist(), clf, device, args.batch_size)
        np.savez_compressed(out_dir / f"{name}.npz", features=feats, logits=logits)
        df.drop(columns=["text"]).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# erasers
# --------------------------------------------------------------------------- #

def fit_leace(X, Z, tol=1e-6):
    """LEACE: r(x) = x - W+ P_U W (x - mu), U = colspace(W Sigma_xz), W = Sigma_xx^(-1/2).
    Returns (mu, M) with r(x) = x - (x - mu) @ M."""
    X = X.astype(np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    vals, vecs = np.linalg.eigh(Xc.T @ Xc / len(X))
    keep = vals > tol * vals.max()
    V, s = vecs[:, keep], np.sqrt(vals[keep])
    W, W_inv = (V / s) @ V.T, (V * s) @ V.T
    u, sv, _ = np.linalg.svd(W @ (Xc.T @ Zc / len(X)), full_matrices=False)
    U = u[:, sv > tol * max(sv.max(initial=0.0), 1e-12)]
    return mu, (W_inv @ U @ U.T @ W).T


class Eraser:
    """One LEACE eraser for all rows, or one per language (rows of other languages pass through)."""

    def __init__(self, fit_meta, X, concept):
        self.global_fit, self.per_lang = None, {}
        if concept == "none":
            return
        if concept == "origin within lang":
            for lang, idx in fit_meta.groupby("language").indices.items():
                orig = (fit_meta.origin.to_numpy()[idx] == "original")
                if min(orig.sum(), (~orig).sum()) >= MIN_ORIGIN_ROWS:
                    self.per_lang[lang] = fit_leace(X[idx], orig[:, None])
            return
        self.global_fit = fit_leace(X, concept_matrix(fit_meta, concept))

    def __call__(self, X, languages):
        X = X.astype(np.float64)
        if self.global_fit is not None:
            mu, M = self.global_fit
            return X - (X - mu) @ M
        out = X.copy()
        for lang, (mu, M) in self.per_lang.items():
            rows = np.asarray(languages) == lang
            out[rows] = X[rows] - (X[rows] - mu) @ M
        return out


def concept_matrix(meta, concept):
    lang, orig = meta.language.to_numpy(), (meta.origin.to_numpy() == "original")
    if concept == "language":
        return pd.get_dummies(lang).to_numpy(float)
    if concept == "origin":
        return orig[:, None].astype(float)
    if concept == "origin x language":
        return np.column_stack([(lang == l) & orig for l in sorted(set(lang))]).astype(float)
    if concept == "cells":
        return pd.get_dummies(pd.Series(lang) + "|" + np.where(orig, "original", "translated")).to_numpy(float)
    raise ValueError(concept)


CONCEPTS = ["none", "language", "origin", "origin x language", "cells", "origin within lang"]


# --------------------------------------------------------------------------- #
# fit + score (CPU)
# --------------------------------------------------------------------------- #

def lr(C, class_weight="balanced"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=3000))


def mlp(seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300,
                                                         early_stopping=True, random_state=seed))


def leakage(test_meta, Xt, ai_meta, Xa, make_probe, seed):
    """Per-language origin probe (original vs translated), 5-fold cross-validated on the
    erased TEST set. Not trained on the fit rows: there the eraser leaves exactly zero
    covariance, so a linear probe learns a constant and scores 0.5 whatever is left on
    other rows. ai->orig: the probe refit on all test rows of the language, applied to AI text."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    scores, ai_orig = {}, {}
    for lang, idx in test_meta.groupby("language").indices.items():
        y = test_meta.origin.to_numpy()[idx] == "original"
        if min(y.sum(), (~y).sum()) < MIN_ORIGIN_ROWS:
            continue
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        scores[lang] = balanced_accuracy_score(y, cross_val_predict(make_probe(), Xt[idx], y, cv=cv))
        rows = (ai_meta.language == lang).to_numpy() if ai_meta is not None else np.zeros(0, bool)
        if rows.any() and lang != "en":
            ai_orig[lang] = make_probe().fit(Xt[idx], y).predict(Xa[rows]).mean()
    return scores, ai_orig


def load_sets(run_dir):
    sets = {}
    for name in SETS:
        if (run_dir / f"{name}.npz").exists():
            arr = np.load(run_dir / f"{name}.npz")
            sets[name] = (pd.read_csv(run_dir / f"{name}.csv"), arr["features"], arr["logits"])
    head = np.load(run_dir / "head.npz")
    return sets, head


def share(pred, group):
    return float(np.mean(np.asarray(pred) == group)) if len(pred) else np.nan


def score(pred, sets, ai_pairs):
    """pred: {set: predicted short labels}. Returns one flat metrics dict."""
    from sklearn.metrics import f1_score
    m = {}
    t, tp = sets["test"][0], pred["test"]
    m["test F1"] = f1_score(t.gold, tp, labels=GROUPS, average="macro")
    for origin in ["translated", "original"]:
        rows = (t.origin == origin).to_numpy()
        m[f"F1 {origin[:5]}"] = f1_score(t.gold[rows], tp[rows], labels=GROUPS, average="macro")
    for lang in ["sk", "ro"]:
        for origin in ["original", "translated"]:
            rows = ((t.language == lang) & (t.origin == origin)).to_numpy()
            m[f"{lang} {origin[:5]} acc"] = float(np.mean(tp[rows] == t.gold.to_numpy()[rows])) if rows.any() else np.nan
    if "cf" in pred:
        c, cp = sets["cf"][0], pred["cf"]
        for name, tag, cue in [("native cs/pl -> sk", "cf->sk", "Sov-right"),
                               ("native it/es/fr/pt -> ro", "cf->ro", "Lib-con")]:
            rows = ((c.set == name) & (c.version == "mt")).to_numpy()
            m[f"{tag} acc"] = float(np.mean(cp[rows] == c.gold.to_numpy()[rows])) if rows.any() else np.nan
            m[f"{tag} {cue}"] = share(cp[rows], cue)
    if "mt" in pred:
        mt, mp = sets["mt"][0], pred["mt"]
        per_lang = pd.Series(mp == mt.gold.to_numpy()).groupby(mt.language.to_numpy()).mean()
        m["mt acc"] = float(per_lang.mean())
        m["mt sk acc"] = float(per_lang.get("sk", np.nan))
        m["mt sk Sov-right"] = share(mp[(mt.language == "sk").to_numpy()], "Sov-right")
    if "ai" in pred:
        a, ap = sets["ai"][0], pred["ai"]
        for lang, cue in [("sk", "Sov-right"), ("ro", "Lib-con"), ("en", "Rad-left")]:
            m[f"ai {lang} {cue}"] = share(ap[(a.language == lang).to_numpy()], cue)
        # top-class share per language, averaged: 0.25 = no language prior at all
        top = pd.Series(ap).groupby(a.language.to_numpy()).agg(lambda s: s.value_counts(normalize=True).iloc[0])
        m["ai top share"] = float(top.mean())
        if ai_pairs is not None:
            left, right = ai_pairs
            agree = pd.Series(ap[left] == ap[right]).groupby(a.language.to_numpy()[left]).mean()
            m["ai = en"] = float(agree.mean())
            m["ai sk = en"] = float(agree.get("sk", np.nan))
    return m


def english_pairs(ai_meta):
    """Row indices (non-en text, its en partner) for the same model/track/statement/framing/paraphrase."""
    if ai_meta is None:
        return None
    idx = ai_meta.reset_index()
    en = idx[idx.language == "en"].set_index(AI_KEY)["index"]
    other = idx[idx.language != "en"].join(en.rename("en_index"), on=AI_KEY).dropna(subset=["en_index"])
    return other["index"].to_numpy(), other.en_index.astype(int).to_numpy()


def fit_stage(args, run_dir):
    sets, head = load_sets(run_dir)
    for need in ["fit", "test"]:
        if need not in sets:
            raise SystemExit(f"missing {run_dir / need}.npz -- run --stage embed first")
    W, b, biases, names = head["W"], head["b"], head["biases"], head["labels"]
    fit_meta, Xfit, Lfit = sets["fit"]
    gap = np.abs(Xfit @ W.T + b - Lfit).max()
    say(f"classifier {head['classifier']}; run dir {run_dir}")
    say(f"captured features reproduce the logits: max |XW'+b - logits| = {gap:.2e}")
    say("rows per set: " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()))
    say(f"fit rows by origin: {fit_meta.origin.value_counts().to_dict()}")
    ai_meta = sets["ai"][0] if "ai" in sets else None
    if ai_meta is not None:
        stored = (names[(sets["ai"][2] + biases).argmax(1)] == ai_meta.stored_pred.to_numpy()).mean()
        say(f"AI texts: recomputed biased label = stored label for {stored:.1%}")
    pairs = english_pairs(ai_meta)

    rows, ai_tables = [], {}
    current = {k: names[(v[2] + biases).argmax(1)] for k, v in sets.items()}
    rows.append({"concept": "current + biases", "head": "orig", **score(current, sets, pairs)})
    for concept in CONCEPTS:
        print(f"--- {concept}")
        eraser = Eraser(fit_meta, Xfit, concept)
        erased = {k: eraser(v[1], v[0].language.to_numpy()) for k, v in sets.items()}
        extra = {}
        probes = [("", lambda: lr(args.C))] + ([("mlp ", lambda: mlp(args.seed))] if args.mlp_probe else [])
        for tag, make_probe in probes:
            leak, ai_orig = leakage(sets["test"][0], erased["test"], ai_meta, erased.get("ai"), make_probe, args.seed)
            extra |= {f"leak {tag}sk": leak.get("sk", np.nan),
                      f"leak {tag}mean": float(np.mean(list(leak.values()))) if leak else np.nan,
                      f"ai->orig {tag}sk": ai_orig.get("sk", np.nan),
                      f"ai->orig {tag}mean": float(np.mean(list(ai_orig.values()))) if ai_orig else np.nan}
        if concept == "origin within lang":
            say(f"origin within lang: erased in {len(eraser.per_lang)} languages: {sorted(eraser.per_lang)}")
        refit = lr(args.C).fit(erased["fit"], fit_meta.gold.to_numpy())
        for head_name, predict in [("orig", lambda X: names[(X @ W.T + b).argmax(1)]),
                                   ("refit", refit.predict)]:
            pred = {k: np.asarray(predict(X)) for k, X in erased.items()}
            rows.append({"concept": concept, "head": head_name, **extra, **score(pred, sets, pairs)})
            if "ai" in pred and head_name == "refit":
                a = sets["ai"][0]
                ai_tables[concept] = (pd.crosstab(a.language, pred["ai"], normalize="index")
                                      .reindex(columns=GROUPS).fillna(0).mul(100).round(0).astype(int))

    table = pd.DataFrame(rows).set_index(["concept", "head"])
    pd.set_option("display.width", 250)
    say(f"\n{'=' * 100}\nLEACE at the classifier input; raw argmax except 'current + biases'\n{'=' * 100}")
    blocks = [[c for c in table.columns if c.startswith(("leak", "ai->orig"))],
              ["test F1", "F1 trans", "F1 origi", "sk origi acc", "sk trans acc", "ro origi acc", "ro trans acc"],
              [c for c in table.columns if c.startswith(("cf", "mt"))],
              [c for c in table.columns if c.startswith("ai ")]]
    for cols in blocks:
        cols = [c for c in cols if c in table.columns]
        if cols:
            say("\n" + table[cols].round(3).to_string())
    for concept, t in ai_tables.items():
        say(f"\nAI text, refit head, {concept}: predicted share by language (%)")
        say(t.to_string())
    table.to_csv(Path(args.report).with_suffix(".csv"), encoding="utf-8-sig")


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["embed", "fit", "all"], default="all")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--data_dir", default=DEFAULT_DATA)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS)
    parser.add_argument("--cf", default=DEFAULT_CF, help="counterfactual.csv from analysis/translation_probe.py")
    parser.add_argument("--sets", default=",".join(SETS), type=lambda s: s.split(","))
    parser.add_argument("--ai_languages", default="sk,ro,cs,pl,hu,de,es,en", type=lambda s: s.split(","))
    parser.add_argument("--n_fit", type=int, default=250, help="train rows per (language, origin, cluster)")
    parser.add_argument("--n_test", type=int, default=100, help="test rows per (language, origin, cluster)")
    parser.add_argument("--n_mt", type=int, default=100, help="MT-augmentation rows per (target, cluster)")
    parser.add_argument("--n_ai", type=int, default=15, help="AI keys per (model, track), all languages each")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the refit head and probes")
    parser.add_argument("--no_mlp_probe", dest="mlp_probe", action="store_false",
                        help="skip the nonlinear (one hidden layer) origin probe next to the linear one")
    parser.add_argument("--min_origin_rows", type=int, default=MIN_ORIGIN_ROWS,
                        help="rows per (language, origin) needed to fit an origin eraser or probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="", help="default: data/leace_probe/<run name>")
    parser.add_argument("--report", default="results_leace_probe.txt")
    args = parser.parse_args()
    if "en" not in args.ai_languages:
        args.ai_languages.append("en")
    return args


def main():
    global MIN_ORIGIN_ROWS
    args = parse_args()
    MIN_ORIGIN_ROWS = args.min_origin_rows
    clf = Path(args.classifier)  # runs/<run>/model or runs/<run>/model_epoch<N>
    run_dir = Path(args.out_dir or OUT_ROOT / (clf.parent.name if clf.name == "model" else f"{clf.parent.name}_{clf.name}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("embed", "all"):
        embed(args, run_dir)
    if args.stage in ("fit", "all"):
        fit_stage(args, run_dir)
        Path(args.report).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()

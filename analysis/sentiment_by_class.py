#!/usr/bin/env python3
"""Sentiment of the test-split speeches, broken down by the class the classifier assigns.

Question: does the cluster classifier route texts by tone as well as by politics? If
speeches it labels Sovereigntist right are more negative than speeches whose *speaker*
is Sovereigntist right, the classifier has (partly) learned "negative = Sov-right".

Three cached stages:
  1. classify   test.parquet through the cluster classifier, argmax(logits + manifest
                biases) as deployed (analysis/classify_speeches.py). Also reports accuracy
                and macro F1, which must match the checkpoint's results file.
  2. sentiment  ParlaSent (classla/xlm-r-parlasent; XLM-R-large pre-trained on 27
                parliaments, fine-tuned on sentences from cs/sk/sl/BCS/en proceedings) is a
                regressor on 0 (negative) .. 5 (positive). It was trained on sentences, so
                each speech is split into sentences, every sentence is scored, and the speech
                gets the mean score plus its shares of negative / positive sentences
                (3-way mapping from the model card: round, clip to 0..5, // 2).
                `--unit text` scores the whole speech truncated to 512 tokens instead.
                Any HF sequence classifier with negative/neutral/positive labels also works
                (score = P(pos) - P(neg), category = argmax).
                `--sentiment_model llm:<model>` instead asks an LLM (utils/api_caller) to rate
                each whole speech on the same 6-level ParlaSent scale. No encoder sentiment
                model is trained on all 21 test languages; an LLM reads all of them. Answers
                are cached in llm_cache_<model>.json, so an interrupted run resumes.
  3. analysis   mean sentiment by predicted class and by gold class; the gold x predicted
                matrix; OLS with language fixed effects
                  M_pred:  score ~ pred + language
                  M_gold:  score ~ gold + language
                  M_both:  score ~ pred + gold + language
                Class effects are effect-coded (deviation from the unweighted mean of the four
                classes), so each reads as "this class is X points more positive than the
                average class". The M_both pred effect is the tone the classifier adds beyond
                who actually spoke and in which language. Robustness: original vs translated
                speeches, ParlaSent training languages only, correct vs wrong predictions,
                and Spearman between score and P(class). Translation invariance: the same
                speech in up to 21 EP translations should get the same score (ICC, per-
                language offsets), which tests whether a rater reads every language alike.

All intervals are 95% percentile intervals from a cluster bootstrap over speeches
(speaker + date): pre-2013 EP translations put one speech in up to 21 languages, so
rows are not independent.

Outputs (default --out_dir data/sentiment_by_class):
  classifier_<clf>_<split>.parquet           stage-1 cache (one row per test text)
  sentiment_<model>_<unit>_<split>.parquet   stage-2 cache
  sentiment_by_class_<tag>.csv               per-class summary (pred and gold)
  sentiment_effects_<tag>.csv                OLS class effects with CIs
  sentiment_by_class_<tag>.tex               LaTeX tabular for the paper
  sentiment_by_class_<tag>.png               figure
and the report results_sentiment_by_class[_<tag>].txt in the repo root.

Usage, from the repository root (GPU strongly recommended; see
analysis/slurm_sentiment_by_class.sh):
  python analysis/sentiment_by_class.py
  python analysis/sentiment_by_class.py --limit 200 --bootstrap 200       # smoke test
  python analysis/sentiment_by_class.py --sentiment_model llm:qwen3.8-flash-next          # all languages
  python analysis/sentiment_by_class.py --sentiment_model cardiffnlp/twitter-xlm-roberta-base-sentiment
"""
import argparse
import hashlib
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.core.labels import party_sort_key, short_party
from utils import PARTY_COLORS, FALLBACK_PARTY_COLOR, configure_stdout

configure_stdout()

DEFAULT_CLASSIFIER = "models/national-k4-logitadj_ni/model"
DEFAULT_DATA_DIR = "data/EuroParl Custom/clusters_k4_national_ni"
DEFAULT_SENTIMENT = "classla/xlm-r-parlasent"
PARTY = "EU Party"
# ParlaSent's training languages that occur in the EP test split (BCS does not).
PARLASENT_LANGUAGES = {"en", "cs", "sk", "sl"}
CATEGORIES = ["negative", "neutral", "positive"]
# A sentence ends at . ! ? ... followed by whitespace; fragments shorter than this are
# glued to the previous sentence ("Mr. President", "Art. 5", "No.").
MIN_SENTENCE_CHARS = 20
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


def indent(text, by="  "):
    return by + str(text).replace("\n", "\n" + by)


def slug(name):
    return re.sub(r"[^0-9A-Za-z]+", "-", str(name)).strip("-").lower()


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# stage 1: classifier
# --------------------------------------------------------------------------- #

def load_test(args):
    df = pd.read_parquet(Path(args.data_dir) / f"{args.split}.parquet")
    df = df[df["text"].notna() & df[PARTY].notna()].reset_index(drop=True)
    # Origin as in preprocessing/build_language_matched_split.py: a speech present in
    # <= 2 languages of the split is an original, otherwise an EP translation. Computed on
    # the full split, before any --limit subsample.
    n_lang = df.groupby(["speaker", "date"])["language"].transform("nunique")
    df["origin"] = np.where(n_lang <= 2, "original", "translated")
    if args.pilot:
        # N random speeches present in >= --invariance_min_langs languages, every translation
        # kept: the same rows for every rater, to compare their translation invariance.
        n_lang = df.groupby(["speaker", "date"])["language"].transform("nunique")
        keys = df.loc[n_lang >= args.invariance_min_langs, ["speaker", "date"]].drop_duplicates()
        keys = keys.sample(n=min(args.pilot, len(keys)), random_state=args.seed)
        df = df.merge(keys, on=["speaker", "date"]).drop_duplicates(["speaker", "date", "language"])
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    df["text_sha"] = df["text"].map(sha)
    df["speech"] = df.groupby(["speaker", "date"], sort=False).ngroup()
    return df


def classify(df, args, device, path):
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(df) and (cached["text_sha"].to_numpy() == df["text_sha"].to_numpy()).all():
            print(f"classifier predictions from cache {path}")
            return cached
        print(f"stale cache {path} (rows differ), reclassifying")
    from analysis.classify_speeches import load_classifier, predict_class_logits, softmax

    clf = load_classifier(args.classifier, device)
    texts = df["text"].tolist()
    order = np.argsort([len(t) for t in texts])[::-1]  # long first: OOM shows up at once
    logits = np.zeros((len(texts), len(clf.labels)), dtype=np.float32)
    for start in tqdm(range(0, len(order), args.clf_batch), desc="classifier"):
        idx = order[start:start + args.clf_batch]
        logits[idx] = predict_class_logits([texts[i] for i in idx], clf, device)
    adjusted = logits if clf.biases is None else logits + clf.biases
    probs = softmax(adjusted / clf.temperature)
    out = df[["text_sha", "language", "speaker", "date", "origin", "speech", PARTY]].rename(columns={PARTY: "gold"})
    out["pred"] = np.array(clf.labels)[probs.argmax(1)]
    for j, label in enumerate(clf.labels):
        out[f"p_{slug(label)}"] = probs[:, j]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    del clf
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# stage 2: sentiment
# --------------------------------------------------------------------------- #

def split_sentences(text):
    sentences = []
    for piece in SENTENCE_BREAK.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and (len(piece) < MIN_SENTENCE_CHARS or len(sentences[-1]) < MIN_SENTENCE_CHARS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences or [text.strip()]


class SentimentModel:
    def __init__(self, name, device, max_len):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device, self.max_len = device, max_len
        self.regression = self.model.config.num_labels == 1
        if self.regression:
            self.scale = "ParlaSent regression, 0 = negative .. 5 = positive"
        else:
            names = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
            self.cols = {}
            for cat in CATEGORIES:
                hits = [i for i, l in names.items() if l.startswith(cat[:3])]
                if len(hits) != 1:
                    raise SystemExit(f"{name}: cannot find a '{cat}' label in {names}")
                self.cols[cat] = hits[0]
            self.scale = "P(positive) - P(negative), -1 .. 1"

    @torch.no_grad()
    def __call__(self, texts, batch):
        """(score, category index 0/1/2) per text."""
        order = np.argsort([len(t) for t in texts])[::-1]
        score = np.zeros(len(texts), dtype=np.float32)
        cat = np.zeros(len(texts), dtype=np.int8)
        autocast = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else torch.autocast("cpu", enabled=False)
        for start in tqdm(range(0, len(order), batch), desc="sentiment"):
            idx = order[start:start + batch]
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", truncation=True,
                                 padding=True, max_length=self.max_len).to(self.device)
            with autocast:
                logits = self.model(**enc).logits.float().cpu().numpy()
            if self.regression:
                s = logits[:, 0]
                score[idx] = s
                cat[idx] = np.clip(np.round(s), 0, 5).astype(int) // 2
            else:
                p = np.exp(logits - logits.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                score[idx] = p[:, self.cols["positive"]] - p[:, self.cols["negative"]]
                cat[idx] = np.stack([p[:, self.cols[c]] for c in CATEGORIES], 1).argmax(1)
        return score, cat


class LLMRater:
    """Zero-shot rating of a whole speech by an LLM on ParlaSent's own 6-level scale, so the
    score is on the same 0..5 axis as the ParlaSent regressor. Answers are cached per text
    in a JSON file (resumable: an interrupted run only re-asks what is missing)."""

    PROMPT = (
        "You rate the sentiment of a speech from the European Parliament. Sentiment means the overall "
        "emotional tone the speaker expresses (how they feel about what they talk about), not the topic "
        "and not whether you agree.\n\n"
        "Use this 6-point scale (from the ParlaSent annotation guidelines):\n"
        "0 = Negative: clearly negative tone throughout\n"
        "1 = Mixed negative: both positive and negative, negative prevails\n"
        "2 = Neutral, leaning negative: mostly neutral or factual, slightly negative\n"
        "3 = Neutral, leaning positive: mostly neutral or factual, slightly positive\n"
        "4 = Mixed positive: both positive and negative, positive prevails\n"
        "5 = Positive: clearly positive tone throughout\n\n"
        "The speech may be in any EU language; judge it in its own language, do not translate it.\n"
        "Answer with a single digit (0-5) and nothing else.\n\nSpeech:\n{text}")
    ANSWER = re.compile(r"\D*([0-5])\D*")

    def __init__(self, model, workers, cache_path):
        import json
        import threading
        self.model, self.workers, self.cache_path = model, workers, cache_path
        self.cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.lock = threading.Lock()
        self.scale = f"LLM ({model}) on the ParlaSent 6-level scale, 0 = negative .. 5 = positive"

    def key(self, text):
        return hashlib.sha1(f"{self.PROMPT}\n{text}".encode("utf-8")).hexdigest()

    def save(self):
        import json
        with self.lock:
            snapshot = json.dumps(self.cache, ensure_ascii=False)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(snapshot, encoding="utf-8")
        tmp.replace(self.cache_path)

    def _one(self, text):
        import time
        from utils.api_caller import call_api
        for attempt in range(10):
            try:
                reply = call_api(self.PROMPT.format(text=text), self.model, max_tokens=20,
                                 enable_thinking=False, reasoning_effort="none")
            except RuntimeError as err:  # 429 = over the key's parallel-request limit
                time.sleep(min(60, 2 * (attempt + 1)) if "429" in str(err) else 5)
                continue
            out = (reply["choices"][0]["message"].get("content") or "").strip()
            # Parse the answer only: an error string ("HTTP 429") must never read as a score.
            match = self.ANSWER.fullmatch(out)
            return int(match.group(1)) if match else -1  # -1 = unparseable, kept in the cache
        return None  # not cached: retried on the next run

    def __call__(self, texts, batch=None):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        keys = [self.key(t) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        print(f"LLM {self.model}: {len(texts) - len(todo):,} cached, {len(todo):,} to rate")
        if todo:
            with ThreadPoolExecutor(self.workers) as pool:
                futures = {pool.submit(self._one, t): k for k, t in todo.items()}
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=self.model), 1):
                    if future.result() is not None:
                        with self.lock:
                            self.cache[futures[future]] = future.result()
                    if i % 500 == 0:
                        self.save()
            self.save()
        raw = np.array([self.cache.get(k, -1) for k in keys], dtype=float)
        failed = int((raw < 0).sum())
        if failed:
            print(f"  {failed} texts without a usable answer (NaN, dropped from the analysis)")
        score = np.where(raw < 0, np.nan, raw)
        cat = np.where(raw < 0, -1, np.clip(raw, 0, 5) // 2).astype(np.int8)
        return score, cat


def is_llm(name):
    return name.startswith("llm:")


def sentiment(df, args, device, path):
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(df) and (cached["text_sha"].to_numpy() == df["text_sha"].to_numpy()).all():
            print(f"sentiment scores from cache {path}")
            return cached
        print(f"stale cache {path} (rows differ), rescoring")
    if is_llm(args.sentiment_model):
        name = args.sentiment_model[len("llm:"):]
        model = LLMRater(name, args.llm_workers, path.parent / f"llm_cache_{slug(name)}.json")
    else:
        model = SentimentModel(args.sentiment_model, device, 512 if args.unit == "text" else args.sentence_max_len)
    if args.unit == "text":
        units = df["text"].tolist()
        owner = np.arange(len(df))
    else:
        pieces = [split_sentences(t) for t in df["text"]]
        units = [s for p in pieces for s in p]
        owner = np.repeat(np.arange(len(df)), [len(p) for p in pieces])
        print(f"{len(units):,} sentences from {len(df):,} texts "
              f"(median {int(np.median([len(p) for p in pieces]))} per text)")
    score, cat = model(units, args.sentiment_batch)
    n = np.bincount(owner, minlength=len(df))
    out = pd.DataFrame({"text_sha": df["text_sha"], "n_units": n,
                        "score": np.bincount(owner, weights=score, minlength=len(df)) / n})
    for k, name in enumerate(CATEGORIES):
        out[f"share_{name}"] = np.bincount(owner, weights=(cat == k), minlength=len(df)) / n
    out.loc[out["score"].isna(), [f"share_{name}" for name in CATEGORIES]] = np.nan
    out["scale"] = model.scale
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# stage 3: analysis
# --------------------------------------------------------------------------- #

def bootstrap_weights(clusters, n_boot, seed):
    """(n_boot, n_rows) integer weights: clusters drawn with replacement."""
    rng = np.random.default_rng(seed)
    n_clusters = clusters.max() + 1
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        yield counts[clusters].astype(float)


def ci(reps):
    reps = np.asarray(reps)
    return np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)


def group_means(values, groups, n_groups, w):
    """Weighted mean of each column of `values` (n, m) per group -> (n_groups, m)."""
    tot = np.bincount(groups, weights=w, minlength=n_groups)
    return np.stack([np.bincount(groups, weights=w * v, minlength=n_groups) for v in values.T], 1) / tot[:, None]


def effect_code(codes, k):
    """Deviation coding: k-1 columns, the last level is -1 everywhere."""
    x = np.zeros((len(codes), k - 1))
    for j in range(k - 1):
        x[:, j] = (codes == j)
    x[codes == k - 1] = -1
    return x


def design(d, terms, classes, languages):
    blocks = [np.ones((len(d), 1))]
    for term in terms:
        blocks.append(effect_code(d[f"{term}_code"].to_numpy(), len(classes)))
    lang = d["lang_code"].to_numpy()
    if len(languages) > 1:
        blocks.append(np.stack([(lang == j) for j in range(1, len(languages))], 1).astype(float))
    return np.hstack(blocks)


def class_effects(beta, terms, k):
    """All k effects per term (the last one is minus the sum of the others)."""
    out = {}
    for t, term in enumerate(terms):
        b = beta[1 + t * (k - 1): 1 + (t + 1) * (k - 1)]
        out[term] = np.append(b, -b.sum())
    return out


def wls(x, y, w):
    xtw = x.T * w
    return np.linalg.lstsq(xtw @ x, xtw @ y, rcond=None)[0]


def analyse(d, args, scale, tag, out_dir):
    classes = sorted(d["gold"].unique(), key=party_sort_key)
    k = len(classes)
    languages = sorted(d["language"].unique(), key=lambda l: (l != "en", l))  # en = FE reference
    d = d.copy()
    d["pred_code"] = d["pred"].map({c: i for i, c in enumerate(classes)})
    d["gold_code"] = d["gold"].map({c: i for i, c in enumerate(classes)})
    d["lang_code"] = d["language"].map({l: i for i, l in enumerate(languages)})
    measures = ["score", "share_negative", "share_positive"]
    vals = d[measures].to_numpy()
    clusters = d["speech"].to_numpy()
    n = len(d)
    one = np.ones(n)

    acc = (d.pred == d.gold).mean()
    f1 = []
    for c in classes:
        tp = ((d.pred == c) & (d.gold == c)).sum()
        f1.append(2 * tp / ((d.pred == c).sum() + (d.gold == c).sum()))
    say(f"classifier on {args.split}: n={n:,}, accuracy {acc:.4f}, macro F1 {np.mean(f1):.4f} "
        f"(compare the checkpoint's results file)")
    say(f"sentiment model: {args.sentiment_model} | unit: {args.unit} | scale: {scale}")
    say(f"{d['speech'].nunique():,} speech clusters (speaker + date) for the bootstrap, B={args.bootstrap}")
    say(f"origin: {(d.origin == 'original').mean():.0%} original, {(d.origin == 'translated').mean():.0%} translated; "
        f"mean {d.n_units.mean():.1f} {args.unit}s per text")

    # --- observed + bootstrap: per-class means (pred, gold), the matrix, the OLS effects
    terms_by_model = {"M_pred": ["pred"], "M_gold": ["gold"], "M_both": ["pred", "gold"]}
    xs = {m: design(d, t, classes, languages) for m, t in terms_by_model.items()}
    y = d["score"].to_numpy()
    cell = d["gold_code"].to_numpy() * k + d["pred_code"].to_numpy()
    grand = vals.mean(0)

    def stats(w):
        res = {"pred": group_means(vals, d["pred_code"].to_numpy(), k, w),
               "gold": group_means(vals, d["gold_code"].to_numpy(), k, w),
               "cell": group_means(vals[:, :1], cell, k * k, w)[:, 0].reshape(k, k)}
        for m, terms in terms_by_model.items():
            for term, eff in class_effects(wls(xs[m], y, w), terms, k).items():
                res[f"{m}:{term}"] = eff
        # within each gold class: predicted-as-other minus predicted-correctly
        res["wrong_minus_right"] = np.array([
            np.nanmean(res["cell"][g][np.arange(k) != g]) - res["cell"][g, g] for g in range(k)])
        return res

    obs = stats(one)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = [stats(w) for w in tqdm(bootstrap_weights(clusters, args.bootstrap, args.seed),
                                       total=args.bootstrap, desc="bootstrap")]
    lo, hi = {}, {}
    for key in obs:
        lo[key], hi[key] = ci([r[key] for r in reps])

    def fmt(key, idx, digits=2, pct=False):
        f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.{digits}f}")
        return f"{f(obs[key][idx])} [{f(lo[key][idx])}, {f(hi[key][idx])}]"

    # --- 1. by predicted and gold class
    say(f"\n{'=' * 100}\n[1] SENTIMENT BY CLASS (mean score; % negative / positive {args.unit}s)\n{'=' * 100}")
    say(f"all texts: score {grand[0]:.3f}, {100 * grand[1]:.1f}% negative, {100 * grand[2]:.1f}% positive")
    rows = []
    for basis in ["pred", "gold"]:
        counts = d[f"{basis}_code"].value_counts().reindex(range(k), fill_value=0).to_numpy()
        say(f"\nby {'PREDICTED' if basis == 'pred' else 'GOLD (speaker)'} class")
        say(f"  {'class':<10}{'n':>7}   {'score [95% CI]':<24}{'% neg [CI]':<22}{'% pos [CI]':<22}")
        for i, c in enumerate(classes):
            say(f"  {short_party(c):<10}{counts[i]:>7}   {fmt(basis, (i, 0), 3):<24}"
                f"{fmt(basis, (i, 1), pct=True):<22}{fmt(basis, (i, 2), pct=True):<22}")
            rows.append({"basis": basis, "class": c, "n": counts[i],
                         **{f"{m}{suf}": arr[basis][i, j] for j, m in enumerate(measures)
                            for suf, arr in [("", obs), ("_lo", lo), ("_hi", hi)]}})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"sentiment_by_class_{tag}.csv", index=False)

    # --- 2. gold x predicted
    say(f"\n{'=' * 100}\n[2] GOLD (rows) x PREDICTED (cols): mean score, n below\n{'=' * 100}")
    ct = pd.crosstab(d.gold_code, d.pred_code).reindex(index=range(k), columns=range(k), fill_value=0)
    names = [short_party(c) for c in classes]
    say(indent(pd.DataFrame(obs["cell"], index=names, columns=names).round(3).to_string()))
    say(indent(pd.DataFrame(ct.to_numpy(), index=names, columns=names).to_string()))
    say("\nwithin each gold class: mean score of texts predicted as another class (unweighted over the\n"
        "other three) minus texts predicted correctly")
    for g, c in enumerate(classes):
        say(f"  {short_party(c):<10}{fmt('wrong_minus_right', g, 3)}")
    say("\nwhere do misclassified texts of each gold class go, and are they more negative there?")
    for g, c in enumerate(classes):
        parts = [f"->{names[p]} {obs['cell'][g, p] - obs['cell'][g, g]:+.3f} (n={ct.iat[g, p]})"
                 for p in range(k) if p != g]
        say(f"  {names[g]:<10}" + "  ".join(parts))

    # --- 3. OLS
    say(f"\n{'=' * 100}\n[3] OLS WITH LANGUAGE FIXED EFFECTS: class effects on the score\n{'=' * 100}")
    say("effect-coded: deviation from the unweighted mean of the classes; 95% cluster-bootstrap CI")
    eff_rows = []
    for m, terms in terms_by_model.items():
        say(f"\n{m}: score ~ {' + '.join(terms)} + language")
        for term in terms:
            key = f"{m}:{term}"
            for i, c in enumerate(classes):
                sig = "*" if lo[key][i] > 0 or hi[key][i] < 0 else " "
                say(f"  {term:<5}{short_party(c):<10}{fmt(key, i, 3)} {sig}")
                eff_rows.append({"model": m, "term": term, "class": c, "effect": obs[key][i],
                                 "lo": lo[key][i], "hi": hi[key][i]})
    effects = pd.DataFrame(eff_rows)
    effects.to_csv(out_dir / f"sentiment_effects_{tag}.csv", index=False)
    spread = {m: np.ptp(obs[f"{m}:pred"]) for m in ["M_pred", "M_both"]}
    say(f"\nspread of pred effects (max - min): {spread['M_pred']:.3f} without gold, "
        f"{spread['M_both']:.3f} holding gold fixed; gold spread {np.ptp(obs['M_gold:gold']):.3f}")
    say(f"score SD across texts: {y.std():.3f}; within-language SD: "
        f"{(y - d.groupby('language')['score'].transform('mean')).std():.3f}")

    # --- 4. robustness subsets (point estimates of M_both pred effects + pred means)
    say(f"\n{'=' * 100}\n[4] ROBUSTNESS: M_both pred effects (and pred-class means) in subsets\n{'=' * 100}")
    subsets = {
        "all": np.ones(n, bool),
        "original only": (d.origin == "original").to_numpy(),
        "translated only": (d.origin == "translated").to_numpy(),
        "ParlaSent train langs (en/cs/sk/sl)": d.language.isin(PARLASENT_LANGUAGES).to_numpy(),
        "other languages": (~d.language.isin(PARLASENT_LANGUAGES)).to_numpy(),
        "English only": (d.language == "en").to_numpy(),
    }
    say(f"  {'subset':<38}{'n':>7}  " + "".join(f"{nm:>11}" for nm in names) + "   | pred means")
    for label, mask in subsets.items():
        s = d[mask]
        if s.pred_code.nunique() < k or s.gold_code.nunique() < k or len(s) < 50:
            say(f"  {label:<38}{len(s):>7}  (too few rows / classes)")
            continue
        langs = sorted(s.language.unique(), key=lambda l: (l != "en", l))
        s = s.assign(lang_code=s.language.map({l: i for i, l in enumerate(langs)}))
        eff = class_effects(wls(design(s, ["pred", "gold"], classes, langs), s.score.to_numpy(),
                                np.ones(len(s))), ["pred", "gold"], k)["pred"]
        means = s.groupby("pred_code")["score"].mean().reindex(range(k))
        say(f"  {label:<38}{len(s):>7}  " + "".join(f"{e:>+11.3f}" for e in eff)
            + "   | " + " ".join(f"{m:.2f}" for m in means))

    say("\ncorrect vs wrong predictions: mean score by predicted class")
    for label, mask in {"correct": d.pred == d.gold, "wrong": d.pred != d.gold}.items():
        means = d[mask].groupby("pred_code")["score"].agg(["mean", "size"]).reindex(range(k))
        say(f"  {label:<8}" + "  ".join(f"{names[i]} {means['mean'][i]:.3f} (n={int(means['size'][i])})"
                                        for i in range(k)))

    say("\nSpearman rho between score and P(class) (calibrated probability), all texts / within language mean-centred")
    centred = y - d.groupby("language")["score"].transform("mean").to_numpy()
    for c in classes:
        p = d[f"p_{slug(c)}"].to_numpy()
        pc = p - d.groupby("language")[f"p_{slug(c)}"].transform("mean").to_numpy()
        say(f"  {short_party(c):<10}{spearmanr(y, p)[0]:+.3f}  /  {spearmanr(centred, pc)[0]:+.3f}")

    say("\nmean score by language (n), with each language's top predicted class")
    by_lang = d.groupby("language").agg(n=("score", "size"), score=("score", "mean"),
                                        neg=("share_negative", "mean"))
    by_lang["top_pred"] = d.groupby("language")["pred"].agg(lambda s: short_party(s.value_counts().index[0]))
    say(indent(by_lang.sort_values("score").round(3).to_string()))
    d.groupby(["language", "pred"])[measures].mean().assign(
        n=d.groupby(["language", "pred"]).size()).to_csv(out_dir / f"sentiment_by_language_class_{tag}.csv")

    translation_invariance(d, args)

    write_latex(summary, effects, classes, scale, args, out_dir / f"sentiment_by_class_{tag}.tex", n)
    plot(summary, effects, classes, scale, out_dir / f"sentiment_by_class_{tag}.png")


def translation_invariance(d, args):
    """The same EP speech in several languages should get the same score from a rater that
    reads every language alike. On speeches present in >= --invariance_min_langs languages:
    ICC(1) (share of score variance between speeches, 1 = translation-invariant), the mean
    absolute deviation of a translation from its speech's mean, agreement of the 3-way
    category with the speech's modal category, and each language's mean deviation from the
    speech mean (a per-language offset the rater adds; language fixed effects absorb a
    constant offset, not a noisier reading)."""
    d = d[d.groupby("speech")["language"].transform("nunique") >= args.invariance_min_langs]
    d = d.drop_duplicates(["speech", "language"])
    say(f"\n{'=' * 100}\n[5] TRANSLATION INVARIANCE: one speech, many EP translations\n{'=' * 100}")
    if d["speech"].nunique() < 5:
        say(f"  fewer than 5 speeches with >= {args.invariance_min_langs} languages in this sample; skipped")
        return
    y, g = d["score"].to_numpy(), d["speech"].to_numpy()
    sizes = d.groupby("speech").size()
    n, k = len(d), len(sizes)
    means = d.groupby("speech")["score"].transform("mean").to_numpy()
    ms_between = (sizes * (d.groupby("speech")["score"].mean() - y.mean()) ** 2).sum() / (k - 1)
    ms_within = ((y - means) ** 2).sum() / (n - k)
    k0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (k0 - 1) * ms_within)
    cat = np.clip(np.round(y), 0, 5).astype(int) // 2  # ParlaSent's 3-way mapping of the text score
    modal = pd.Series(cat).groupby(g).transform(lambda c: c.mode().iloc[0]).to_numpy()
    say(f"  {k} speeches x {n / k:.1f} languages on average ({n:,} texts)")
    say(f"  ICC(1) {icc:.3f}  |  mean |translation - speech mean| {np.abs(y - means).mean():.3f}  "
        f"(score SD {y.std():.3f})  |  3-way category = speech's modal category {np.mean(cat == modal):.1%}")
    offset = pd.DataFrame({"language": d["language"].to_numpy(), "dev": y - means}).groupby("language")["dev"]
    table = offset.agg(["mean", "std", "size"]).sort_values("mean")
    say(f"  per-language offset from the speech mean (SD of offsets {table['mean'].std():.3f}):")
    say(indent(table.round(3).T.to_string(), "    "))


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

def write_latex(summary, effects, classes, scale, args, path, n):
    def cell(v, l, h, digits=2, pct=False):
        raw = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.{digits}f}")
        f = lambda x: raw(x).replace("-", "$-$")  # a typeset minus, not a hyphen
        return f"{f(v)} {{\\scriptsize [{f(l)}, {f(h)}]}}"

    lines = [
        "% generated by analysis/sentiment_by_class.py -- do not edit by hand",
        f"% classifier {args.classifier}; split {args.split}; sentiment {args.sentiment_model} "
        f"({args.unit}-level); {scale}; n={n}; B={args.bootstrap} speech-cluster bootstrap",
        r"\begin{tabular}{l r l l l l l}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Predicted class} & Gold class & \multicolumn{2}{c}{Class effect (OLS)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
        r"Class & $n$ & Score & \% negative & Score & Pred. & Pred.\,|\,gold \\",
        r"\midrule",
    ]
    for c in classes:
        p = summary[(summary.basis == "pred") & (summary["class"] == c)].iloc[0]
        g = summary[(summary.basis == "gold") & (summary["class"] == c)].iloc[0]
        e1 = effects[(effects.model == "M_pred") & (effects.term == "pred") & (effects["class"] == c)].iloc[0]
        e2 = effects[(effects.model == "M_both") & (effects.term == "pred") & (effects["class"] == c)].iloc[0]
        lines.append(" & ".join([
            short_party(c), f"{int(p.n):,}".replace(",", "{,}"),
            cell(p.score, p.score_lo, p.score_hi),
            cell(p.share_negative, p.share_negative_lo, p.share_negative_hi, pct=True),
            cell(g.score, g.score_lo, g.score_hi),
            cell(e1.effect, e1.lo, e1.hi), cell(e2.effect, e2.lo, e2.hi)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX table -> {path}")


def plot(summary, effects, classes, scale, path):
    from analysis.core.labels import use_short_party_labels
    use_short_party_labels()
    ink, muted, grid = "#1A1A1A", "#5A5A5A", "#E3E3E3"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    pos = np.arange(len(classes))

    for basis, offset, filled, label in [("pred", -0.13, True, "Predicted class"),
                                         ("gold", 0.13, False, "Gold class (speaker)")]:
        s = summary[summary.basis == basis].set_index("class").loc[classes]
        for i, c in enumerate(classes):
            color = PARTY_COLORS.get(c, FALLBACK_PARTY_COLOR)
            a1.errorbar(i + offset, s.score.iloc[i], yerr=[[s.score.iloc[i] - s.score_lo.iloc[i]],
                                                          [s.score_hi.iloc[i] - s.score.iloc[i]]],
                        fmt="o", ms=8, color=color, mfc=color if filled else "white", mew=2,
                        elinewidth=2, capsize=0, zorder=3)
        a1.plot([], [], "o", ms=8, color=muted, mfc=muted if filled else "white", mew=2, label=label)
    a1.set_title("Mean sentiment", fontsize=11, color=ink, loc="left")
    a1.set_ylabel(scale.split(",")[-1].strip() if "ParlaSent" in scale else scale, fontsize=9, color=muted)
    a1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    for model, offset, filled, label in [("M_pred", -0.13, True, "Predicted class"),
                                         ("M_both", 0.13, False, "Predicted | gold")]:
        e = effects[(effects.model == model) & (effects.term == "pred")].set_index("class").loc[classes]
        for i, c in enumerate(classes):
            color = PARTY_COLORS.get(c, FALLBACK_PARTY_COLOR)
            a2.errorbar(i + offset, e.effect.iloc[i], yerr=[[e.effect.iloc[i] - e.lo.iloc[i]],
                                                           [e.hi.iloc[i] - e.effect.iloc[i]]],
                        fmt="o", ms=8, color=color, mfc=color if filled else "white", mew=2,
                        elinewidth=2, capsize=0, zorder=3)
        a2.plot([], [], "o", ms=8, color=muted, mfc=muted if filled else "white", mew=2, label=label)
    a2.axhline(0, color=muted, lw=1, zorder=1)
    a2.set_title("Class effect, language fixed effects", fontsize=11, color=ink, loc="left")
    a2.set_ylabel("deviation from mean class", fontsize=9, color=muted)
    a2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    for ax in (a1, a2):
        ax.set_xticks(pos, classes, fontsize=10)
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"figure -> {path}")


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sentiment_model", default=DEFAULT_SENTIMENT,
                        help="HF model id, or llm:<model> for a zero-shot LLM rater through "
                             "utils/api_caller (whole speeches; --unit is forced to text)")
    parser.add_argument("--llm_workers", type=int, default=4, help="parallel API requests (the key allows 4)")
    parser.add_argument("--invariance_min_langs", type=int, default=10,
                        help="translation-invariance check: speeches present in at least this many languages")
    parser.add_argument("--unit", choices=["sentence", "text"], default="sentence")
    parser.add_argument("--sentence_max_len", type=int, default=256)
    parser.add_argument("--clf_batch", type=int, default=32)
    parser.add_argument("--sentiment_batch", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="random subsample of the split (smoke tests)")
    parser.add_argument("--pilot", type=int, default=0,
                        help="only N random multi-language speeches with all their translations "
                             "(rater comparison on translation invariance)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="data/sentiment_by_class")
    parser.add_argument("--report", default=None, help="default results_sentiment_by_class[_<tag>].txt")
    return parser.parse_args()


def main():
    args = parse_args()
    if is_llm(args.sentiment_model):
        args.unit = "text"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clf_name = slug(Path(args.classifier).parent.name if Path(args.classifier).name == "model"
                    else Path(args.classifier).name)
    lim = (f"_pilot{args.pilot}" if args.pilot else "") + (f"_lim{args.limit}" if args.limit else "")
    sent_name = slug(args.sentiment_model.split("/")[-1])
    tag = f"{clf_name}_{sent_name}_{args.unit}_{args.split}{lim}"
    print(f"device {args.device}; tag {tag}")

    df = load_test(args)
    preds = classify(df, args, args.device, out_dir / f"classifier_{clf_name}_{args.split}{lim}.parquet")
    sent = sentiment(df, args, args.device, out_dir / f"sentiment_{sent_name}_{args.unit}_{args.split}{lim}.parquet")
    d = pd.concat([preds.reset_index(drop=True), sent.drop(columns=["text_sha"]).reset_index(drop=True)], axis=1)
    if d["score"].isna().any():
        print(f"dropping {d['score'].isna().sum()} texts without a sentiment score")
        d = d[d["score"].notna()].reset_index(drop=True)
    analyse(d, args, sent["scale"].iloc[0], tag, out_dir)

    default_report = "results_sentiment_by_class.txt" if tag.startswith(
        "national-k4-logitadj-ni_xlm-r-parlasent_sentence_test") and not lim else f"results_sentiment_by_class_{tag}.txt"
    report = Path(args.report or default_report)
    report.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()

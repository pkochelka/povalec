#!/usr/bin/env python3
"""Why do the classifier's group shares differ so much between models' prose?

The cluster classifier assigns very different group mixes to different models' texts,
even though every model answers the same 30 statements in the same 21 languages under
the same prompts. Two kinds of explanation compete. Either the models argue different
things (stance), or they write differently (vocabulary, register, refusal boilerplate)
and the classifier reacts to that. A third, mechanical candidate sits in between: the
classifier's own inference step -- the language it was trained on and the per-class
logit biases in its manifest -- might amplify small differences.

This script separates the three with data already in the results tree. Five blocks:

  1. decomposition   McFadden R^2 of a multinomial logit of the predicted group on
                     language, statement x stance level, prompt and model, each alone and
                     drop-one. Then the composition-adjusted model shares: what each model
                     would get if its label depended only on (language, statement x
                     stance, prompt). What is left over is how the model writes.
  2. bias            The stored probabilities are softmax(logits + biases), so the raw
                     logits are recoverable up to a per-row constant. Relabels every text
                     without the manifest biases and reports how many labels the biases
                     create, and how much they inflate the between-model variance.
  3. n-grams         For one language (default English): n-grams ranked by weighted lift,
                     rate_group * log2(rate_group / rate_all), where the rates are document
                     frequencies in the group and in all texts of that track and language.
                     Frequent but uninformative n-grams ("is", "can") have lift ~1 and score
                     ~0; rare n-grams score low because rate_group is small. (Plain lift is
                     capped at 1 / group share, which is useless for a group holding most
                     of the texts.) Plus an L2 logit of the group on n-gram presence with
                     statement x stance and prompt fixed effects (and a variant adding
                     model fixed effects).
  4. signatures      Each model's highest weighted-lift n-grams (vs all models, same
                     language and track) and the group mix of that model's texts that
                     contain them.
  5. consistency     Split-half over languages: does a model's residual group profile
                     (from block 1) replicate across disjoint language halves? If so it is
                     a model property that survives translation, not noise.

--perturb additionally runs the classifier (CPU is fine, ~2k texts) on edited English
texts: refusal sentences removed or prepended, speech openers added or removed, "the EU"
replaced by "Brussels". That turns the n-gram associations into causal statements.

Labels: blocks 1 and 3-5 use the classifier's raw labels (argmax of the logits without
the manifest biases); --biased switches them to the stored, bias-adjusted labels. Block 2
and the perturbations always report both. With --soft, blocks 3 and 4 weight each text by
its group probabilities (softmax of the same logits) instead of counting its argmax:
group rates are probability-weighted, the controlled logits are fitted on soft targets,
and group mixes are mean probabilities. Blocks 1 and 5 stay on the argmax.

Stance level: the cross-encoder stance in five bins for speeches, the Likert choice for
reasons; the negated framing is folded onto the base scale as in analysis/core/results.py.

Usage, from the repository root:
  python analysis/classifier_ngram_analysis.py
  python analysis/classifier_ngram_analysis.py --results_dir data/euandi_2024_results_new_ni_fixed_center
  python analysis/classifier_ngram_analysis.py --language de --perturb
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from analysis.analyze_all import MODEL_DIRS
from analysis.core import (
    FRAMING_ORIENTATION,
    REASONS_CLASSIFIED_CSV,
    SPEECHES_CLASSIFIED_CSV,
    find_csvs,
)
from utils import configure_stdout

configure_stdout()

DEFAULT_DATASET = "euandi_2024"
DEFAULT_CLASSIFIER = "models/national-k4-logitadj_ni/model"
DEFAULT_REPORT = "results_classifier_ngrams.txt"
SHORT = {"Liberal-conservative center-right": "Lib-con", "Progressive federalists": "Prog-fed",
         "Radical left": "Rad-left", "Sovereigntist right": "Sov-right"}
GROUPS = list(SHORT.values())
TEXT_PREFIX = {"speeches": "answer", "reasons": "reason"}
PREDICTED = re.compile(r"^predicted_party_(?P<language>[a-z]{2})(?P<neg>_negated)?_v(?P<paraphrase>\d+)$")
TOKEN = r"(?u)\b\w[\w']*\b"
SPEECH_STANCE_BINS = [-1.01, -.5, -.15, .15, .5, 1.01]
# English refusal / "both sides" boilerplate, for the perturbation block only.
DISCLAIMER = re.compile(r"(?:as an ai|i do not (?:have|hold) personal|i don't (?:have|hold)|personal opinions"
                        r"|the question involves a complex|is a complex (?:and|political|question|issue))", re.I)
OPENER = re.compile(r"^\W*(((mr\.?|madam) president|honou?rable members|ladies and gentlemen|(dear )?colleagues"
                    r"|members of (this|the european) parliament)[,;:!.]?\s*)+", re.I)

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


# --------------------------------------------------------------------------- #
# loading: one row per text
# --------------------------------------------------------------------------- #

def load_texts(results_dir, models, labels, biases, temperature=1.0):
    frames = []
    for model in models:
        model_dir = results_dir / model
        for track, pattern in [("speeches", SPEECHES_CLASSIFIED_CSV), ("reasons", REASONS_CLASSIFIED_CSV)]:
            for framing, path in find_csvs(model_dir, pattern).items():
                raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
                scored = (pd.read_csv(str(path).replace("_classified", "_scored"), sep=";", encoding="utf-8-sig")
                          if track == "speeches" else raw)
                for column in raw.columns:
                    match = PREDICTED.match(column)
                    if not match:
                        continue
                    suffix = f"_{match['language']}{match['neg'] or ''}_v{match['paraphrase']}"
                    probs = np.column_stack([
                        pd.to_numeric(raw[f"party_prob_{re.sub(r'[^0-9A-Za-z]+', '_', l).strip('_')}{suffix}"],
                                      errors="coerce") for l in labels])
                    if track == "speeches":
                        stance = FRAMING_ORIENTATION[framing] * pd.to_numeric(
                            scored.get(f"stance{suffix}"), errors="coerce")
                    else:
                        stance = pd.to_numeric(raw.get(f"choice{suffix}"), errors="coerce")
                        stance = 6 - stance if framing == "negated" else stance
                    frames.append(pd.DataFrame({
                        "model": model, "track": track, "framing": framing,
                        "language": match["language"], "paraphrase": int(match["paraphrase"]),
                        "statement_idx": np.arange(len(raw)), "text": raw[f"{TEXT_PREFIX[track]}{suffix}"],
                        "stance": np.asarray(stance, dtype=float), **{f"lp{i}": np.log(np.clip(probs[:, i], 1e-300, None))
                                                                       for i in range(len(labels))}}))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "lp0"])
    lp = df[[f"lp{i}" for i in range(len(labels))]].to_numpy()
    names = np.array([SHORT[l] for l in labels])
    df["pred_biased"] = names[lp.argmax(1)]
    # stored probs are softmax((logits + biases) / T): logits = lp * T - biases up to a row constant
    df["pred_raw"] = names[(lp * temperature - biases).argmax(1)]
    df["words"] = df.text.str.split().str.len()
    bins = pd.cut(df.stance, SPEECH_STANCE_BINS, labels=False).astype(float)
    df["stance_level"] = np.where(df.track == "speeches", bins, df.stance)
    df["stance_level"] = df.stance_level.fillna(-1)
    df["stmt_stance"] = df.statement_idx.astype(str) + "_" + df.stance_level.astype(str)
    df["prompt"] = df.paraphrase.astype(str) + "_" + df.framing
    return df


def shares(labels, by):
    return pd.crosstab(by, labels, normalize="index").reindex(columns=GROUPS, fill_value=0)


def between_var(table):
    return float(((table - table.mean()) ** 2).to_numpy().sum())


# --------------------------------------------------------------------------- #
# 1. decomposition
# --------------------------------------------------------------------------- #

def fit_logit(g, columns, C=10):
    X = OneHotEncoder(handle_unknown="ignore").fit_transform(g[columns].astype(str))
    model = LogisticRegression(C=C, max_iter=3000).fit(X, g.pred)
    return pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=g.index)


def log_lik(g, P):
    return float(np.log(np.clip(P.to_numpy()[np.arange(len(g)), P.columns.get_indexer(g.pred)], 1e-12, 1)).sum())


def decomposition(g):
    prior = g.pred.value_counts(normalize=True)
    ll0 = float(np.log(prior.reindex(g.pred).to_numpy()).sum())
    r2 = lambda cols: 1 - log_lik(g, fit_logit(g, cols)) / ll0
    factors = ["language", "stmt_stance", "prompt", "model"]
    full = r2(factors)
    say(f"  McFadden R2, all four factors: {full:.3f}")
    say(f"  {'factor':12s} {'alone':>7s} {'unique':>7s}")
    for f in factors:
        say(f"  {f:12s} {r2([f]):7.3f} {full - r2([x for x in factors if x != f]):7.3f}")
    expected = fit_logit(g, ["language", "stmt_stance", "prompt"]).groupby(g.model).mean().reindex(columns=GROUPS)
    observed = shares(g.pred, g.model)
    resid = observed - expected
    v_obs, v_exp, v_res = between_var(observed), between_var(expected), between_var(resid)
    say(f"  between-model variance of group shares: observed {v_obs:.4f}, explained by "
        f"(language, statement x stance, prompt) {v_exp:.4f} ({v_exp / v_obs:.0%}), residual {v_res:.4f} "
        f"({v_res / v_obs:.0%})")
    say("  residual share (observed - expected), pp:")
    say((100 * resid).round(1).to_string())
    return resid


# --------------------------------------------------------------------------- #
# 2. bias
# --------------------------------------------------------------------------- #

def bias_effect(g):
    flip = g.pred_biased != g.pred_raw
    say(f"  labels changed by the manifest biases: {flip.mean():.1%}")
    say("  overall share with biases:", g.pred_biased.value_counts(normalize=True).reindex(GROUPS).round(3).to_dict())
    say("  overall share without     :", g.pred_raw.value_counts(normalize=True).reindex(GROUPS).round(3).to_dict())
    say("  where the flips go (rows: raw label, cols: biased label):")
    say(pd.crosstab(g.pred_raw[flip], g.pred_biased[flip]).to_string())
    with_b, without = shares(g.pred_biased, g.model), shares(g.pred_raw, g.model)
    say(f"  between-model variance of shares: with biases {between_var(with_b):.4f}, "
        f"without {between_var(without):.4f}")
    say((100 * pd.concat({"biased": with_b, "raw": without}, axis=1)).round(0).astype(int).to_string())
    quint = g.groupby("language").words.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    say("  by within-language length quintile (0 = shortest): flip rate / biased Rad-left share")
    say(pd.DataFrame({"median_words": g.words.groupby(quint).median(), "flip": flip.groupby(quint).mean().round(2),
                      "rad_left": (g.pred_biased == "Rad-left").groupby(quint).mean().round(2)}).to_string())
    if g.track.iloc[0] == "reasons":
        neutral = g.stance == 3
        say("  neutral (Likert 3) rate per model and the (biased) group mix of those neutral answers:")
        tab = shares(g.pred_biased[neutral], g.model[neutral]).round(2)
        tab.insert(0, "neutral_rate", neutral.groupby(g.model).mean().round(2))
        say(tab.to_string())


# --------------------------------------------------------------------------- #
# 3. n-grams
# --------------------------------------------------------------------------- #

def group_weights(g, soft):
    """(texts x groups) membership weights: one-hot argmax, or the group probabilities."""
    if soft:
        return g[[f"p_{group}" for group in GROUPS]].to_numpy()
    return (g.pred.to_numpy()[:, None] == np.array(GROUPS)[None, :]).astype(float)


def weighted_lift(X, w):
    """rate_group * log2(rate_group / rate_all) per n-gram, on document frequencies.

    `w` is each text's membership weight in the group (0/1 or a probability). Returns
    (score, in-group rate, overall rate). The score is the n-gram's term in the KL
    divergence of the group from the corpus: ~0 for n-grams as common in the group as
    everywhere, small for rare ones, high for n-grams both frequent and over-represented.
    """
    w = np.asarray(w, dtype=float)
    overall = np.asarray(X.sum(0)).ravel() / X.shape[0]
    in_group = np.asarray(X.T @ w).ravel() / w.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(in_group > 0, in_group * np.log2(in_group / overall), 0.0)
    return score, in_group, overall


def ngrams(g, top, min_df, soft):
    g = g.reset_index(drop=True)
    W = group_weights(g, soft)
    vec = CountVectorizer(ngram_range=(1, 2), min_df=min_df, binary=True, token_pattern=TOKEN)
    X = vec.fit_transform(g.text)
    vocab = vec.get_feature_names_out()
    controls = OneHotEncoder(handle_unknown="ignore").fit_transform(g[["stmt_stance", "prompt"]].astype(str))
    models = OneHotEncoder().fit_transform(g[["model"]])
    say(f"  {len(g)} texts, {len(vocab)} n-grams with df >= {min_df}; {'mean probabilities' if soft else 'shares'} "
        f"{dict(zip(GROUPS, W.mean(0).round(2).tolist()))}")
    tables = {}
    for k, group in enumerate(GROUPS):
        w = W[:, k]
        if w.sum() < 20:
            continue

        def coef(extra):
            # soft targets: each text once as a positive (weight w) and once as a negative (1 - w)
            Z = sp.hstack([X] + extra).tocsr()
            fit = LogisticRegression(C=0.3, max_iter=3000).fit(
                sp.vstack([Z, Z]), np.r_[np.ones(len(w)), np.zeros(len(w))], sample_weight=np.r_[w, 1 - w])
            return fit.coef_[0][:len(vocab)]

        score, rate_group, rate_all = weighted_lift(X, w)
        t = pd.DataFrame({"weighted_lift": score, "lift": rate_group / rate_all,
                          "rate_group": rate_group, "rate_all": rate_all,
                          "b_controlled": coef([controls]), "b_controlled_model": coef([controls, models]),
                          "doc_freq": np.asarray(X.sum(0)).ravel()}, index=vocab)
        tables[group] = t
        say(f"\n  == {group} (n={w.sum():.0f})")
        say("  weighted lift (lift, group % / all %): " + ", ".join(
            f"{w} ({r.lift:.1f}x, {100 * r.rate_group:.1f}/{100 * r.rate_all:.1f})"
            for w, r in t.nlargest(top, "weighted_lift").iterrows()))
        say("  | statement x stance:  " + ", ".join(f"{w} ({v:.2f})" for w, v in t.b_controlled.nlargest(top).items()))
        say("  | + model FE:          " + ", ".join(f"{w} ({v:.2f})" for w, v in t.b_controlled_model.nlargest(top).items()))
    return pd.concat(tables, names=["group", "ngram"])


# --------------------------------------------------------------------------- #
# 4. model signatures
# --------------------------------------------------------------------------- #

def signatures(g, per_model, min_df, soft):
    g = g.reset_index(drop=True)
    W = group_weights(g, soft)
    mix = lambda rows: dict(zip(GROUPS, W[rows].mean(0).round(2).tolist()))
    vec = CountVectorizer(ngram_range=(2, 4), min_df=min_df, binary=True, token_pattern=TOKEN)
    X = vec.fit_transform(g.text).tocsc()
    vocab = vec.get_feature_names_out()
    say(f"  baseline {'mean probabilities' if soft else 'shares'} {mix(slice(None))}")
    for model in sorted(g.model.unique()):
        mask = (g.model == model).to_numpy()
        kept = []
        for i in np.argsort(-weighted_lift(X, mask.astype(float))[0]):
            if not any(vocab[i] in k or k in vocab[i] for k in kept):
                kept.append(vocab[i])
            if len(kept) == per_model:
                break
        say(f"\n  {model}: own {mix(mask)}")
        for phrase in kept:
            has = np.asarray(X[:, vec.vocabulary_[phrase]].todense()).ravel() > 0
            say(f"    '{phrase}': {(has & mask).sum() / mask.sum():.0%} of its texts "
                f"(others {has[~mask].mean():.0%}) -> {mix(has & mask)}")


# --------------------------------------------------------------------------- #
# 5. consistency across languages
# --------------------------------------------------------------------------- #

def consistency(g, rng, draws=200):
    g = g.copy()
    g["lang_stmt_stance"] = g.language + "_" + g.stmt_stance
    P = fit_logit(g, ["language", "stmt_stance", "lang_stmt_stance", "prompt"], C=1).reindex(columns=GROUPS, fill_value=0)
    O = pd.get_dummies(g.pred).reindex(columns=GROUPS, fill_value=0).astype(float)
    R = (O - P).groupby([g.model, g.language]).mean()
    languages = np.array(sorted(g.language.unique()))
    rs = []
    for _ in range(draws):
        half = set(rng.choice(languages, len(languages) // 2, replace=False))
        in_half = R.index.get_level_values(1).isin(half)
        a, b = R[in_half].groupby(level=0).mean(), R[~in_half].groupby(level=0).mean()
        rs.append(np.corrcoef(a.to_numpy().ravel(), b.to_numpy().ravel())[0, 1])
    say(f"  split-half correlation of model residual profiles over languages: r = {np.mean(rs):.2f} "
        f"[{np.percentile(rs, 2.5):.2f}, {np.percentile(rs, 97.5):.2f}]")


# --------------------------------------------------------------------------- #
# perturbations (runs the classifier)
# --------------------------------------------------------------------------- #

def perturb(df, classifier_dir, n, seed):
    from analysis.classify_speeches import load_classifier, predict_class_logits
    clf = load_classifier(classifier_dir, "cpu")
    names = np.array([SHORT[l] for l in clf.labels])

    def classify(texts, batch=16):
        # (biased, raw) labels from one forward pass
        logits = np.vstack([predict_class_logits(texts[i:i + batch], clf, "cpu") for i in range(0, len(texts), batch)])
        return names[(logits + clf.biases).argmax(1)], names[logits.argmax(1)]

    def show(name, rows, after):
        say(f"\n  {name} (n={len(rows)})")
        for kind, before, edited in [("biased", rows.pred_biased.to_numpy(), after[0]), ("raw", rows.pred_raw.to_numpy(), after[1])]:
            table = pd.DataFrame({"before": pd.Series(before).value_counts(normalize=True),
                                  "after": pd.Series(edited).value_counts(normalize=True)}).reindex(GROUPS).fillna(0)
            say(f"    {kind}: label changed {np.mean(before != edited):.0%}")
            say("    " + (100 * table).round(1).T.to_string().replace("\n", "\n    "))

    en = df[df.language == "en"]
    reasons, speeches = en[en.track == "reasons"], en[en.track == "speeches"]
    sample = lambda d: d.sample(min(n, len(d)), random_state=seed)

    with_disc = sample(reasons[reasons.text.str.contains(DISCLAIMER)])
    stripped = with_disc.text.map(lambda t: " ".join(s for s in re.split(r"(?<=[.!?])\s+", t)
                                                     if not DISCLAIMER.search(s)))
    keep = stripped.str.split().str.len() >= 6
    show("reasons: refusal/both-sides sentence REMOVED", with_disc[keep], classify(stripped[keep].tolist()))
    without = sample(reasons[~reasons.text.str.contains(DISCLAIMER)])
    for prefix in ["As an AI, I do not have personal opinions. ",
                   "This question involves a complex debate with significant arguments on both sides. "]:
        show(f"reasons: PREPEND '{prefix.strip()}'", without, classify((prefix + without.text).tolist()))

    plain = sample(speeches[~speeches.text.str.contains(r"president|honou?rable|colleagues|ladies", case=False)])
    for prefix in ["Mr President, honourable Members, ", "Ladies and gentlemen, ", "Colleagues, ", "I rise today to say that "]:
        show(f"speeches: PREPEND '{prefix.strip()}'", plain, classify((prefix + plain.text).tolist()))
    opened = sample(speeches[speeches.text.str.contains(OPENER)])
    show("speeches: opener REMOVED", opened, classify(opened.text.str.replace(OPENER, "", regex=True).tolist()))
    eu = sample(speeches[speeches.text.str.contains(r"\bthe EU\b")])
    show("speeches: 'the EU' -> 'Brussels'", eu,
         classify(eu.text.str.replace(r"\bthe EU\b", "Brussels", regex=True).tolist()))


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--results_dir", default="", help="overrides data/<dataset>_results")
    parser.add_argument("--models", default=",".join(MODEL_DIRS))
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                        help="model dir whose manifest.json biases produced the classified CSVs")
    parser.add_argument("--language", default="en", help="language for the n-gram and signature blocks")
    parser.add_argument("--min_df", type=int, default=15)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--signatures", type=int, default=6, help="phrases per model")
    parser.add_argument("--biased", action="store_true",
                        help="analyse the bias-adjusted labels instead of the raw logit argmax")
    parser.add_argument("--calibration", default="",
                        help="calibration.json from analysis/calibrate_classifier.py; overrides --biased")
    parser.add_argument("--method", default="bcts", help="method inside --calibration (raw, manifest, ts, manifest+ts, bcts)")
    parser.add_argument("--soft", action="store_true",
                        help="weight texts by group probabilities in the n-gram and signature blocks")
    parser.add_argument("--perturb", action="store_true", help="also run the classifier on edited texts")
    parser.add_argument("--perturb_n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--csv", default="results_classifier_ngrams.csv", help="per-n-gram table ('' to skip)")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir or f"data/{args.dataset}_results")
    manifest = json.loads((Path(args.classifier) / "manifest.json").read_text(encoding="utf-8"))
    labels = sorted(manifest["label2id"], key=manifest["label2id"].get)
    biases = np.asarray(manifest.get("biases") or np.zeros(len(labels)))
    stored_t = float(manifest.get("temperature", 1.0))
    say(f"results: {results_dir}   classifier manifest: {args.classifier}   biases: {np.round(biases, 2).tolist()}"
        f"   temperature: {stored_t:.3f}")
    df = load_texts(results_dir, [m.strip() for m in args.models.split(",") if m.strip()], labels, biases, stored_t)
    # stored log-probs minus the manifest biases = raw logits up to a per-row constant
    logits = df[[f"lp{i}" for i in range(len(labels))]].to_numpy() * stored_t - biases
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        assert cal["labels"] == labels, "calibration label order differs from the manifest"
        method = cal["methods"][args.method]
        cal_biases, temperature = np.asarray(method["biases"]), method["temperature"]
        mode = f"calibrated ({args.method}: T={temperature}, biases {np.round(cal_biases, 2).tolist()})"
    else:
        # the manifest's own temperature (1 when it has none) for the soft probabilities
        cal_biases, temperature = (biases, stored_t) if args.biased else (np.zeros(len(labels)), stored_t)
        mode = "biased" if args.biased else "raw (no manifest biases)"
    x = (logits + cal_biases) / temperature
    df["pred"] = np.array([SHORT[l] for l in labels])[x.argmax(1)]
    probs = np.exp(x - x.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    for i, label in enumerate(labels):
        df[f"p_{SHORT[label]}"] = probs[:, i]
    say(f"{len(df)} classified texts, {df.model.nunique()} models, {df.language.nunique()} languages; "
        f"labels: {mode}{', soft (probability-weighted)' if args.soft else ''}; "
        f"mean max probability {probs.max(1).mean():.3f}")

    ngram_tables = {}
    for track, g in df.groupby("track"):
        say(f"\n{'=' * 100}\n{track.upper()}\n{'=' * 100}")
        say("\n[1] What explains the label: factor decomposition")
        decomposition(g)
        say("\n[2] What the manifest biases do")
        bias_effect(g)
        say("\n[5] Is a model's residual profile stable across languages?")
        consistency(g, rng)
        sub = g[g.language == args.language]
        say(f"\n[3] Group-distinctive n-grams ({args.language})")
        ngram_tables[track] = ngrams(sub, args.top, args.min_df, args.soft)
        say(f"\n[4] Model signature phrases ({args.language}) and where the classifier puts them")
        signatures(sub, args.signatures, max(5, args.min_df // 2), args.soft)

    if args.perturb:
        say(f"\n{'=' * 100}\nPERTURBATIONS (English, classifier re-run)\n{'=' * 100}")
        perturb(df, args.classifier, args.perturb_n, args.seed)

    Path(args.report).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")
    if args.csv:
        pd.concat(ngram_tables, names=["track"]).to_csv(args.csv)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

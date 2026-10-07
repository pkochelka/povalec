#!/usr/bin/env python3
"""Is the per-language skew of the cluster classifier carried by the input text itself?

On AI-written text the national-k4 classifier labels Slovak ~99% Sovereigntist right and
generic Romanian Liberal-conservative. The traced cause is native vs translated training
text: original Slovak comes only from Slovak MEPs. Two probes, both with Kimi K3 as the
translator and the national-k4 checkpoint as the classifier:

  1. translate-test   AI texts (reasons + speeches, every model) written in --languages are
                      machine-translated into English and classified again. Each is paired
                      with the same model's English text for the same statement, framing,
                      paraphrase and track. If the skew is in Slovak surface features, the
                      translated label moves away from Sov-right and toward the English
                      partner's label.
  2. counterfactual   Native (original-language) EP speeches of known cluster are translated
                      INTO the target language (e.g. Czech/Polish Lib-con -> Slovak) and the
                      target's own native speeches OUT of it (Slovak -> Czech). If a correctly
                      labelled Czech Lib-con speech becomes Sov-right once it is in Slovak,
                      the target language alone pulls the label. Human EP translations into
                      the target language (test split) are classified as the baseline.

Origin, as in preprocessing/build_language_matched_split.py: a speech (speaker + date)
present in <= 2 languages of the split is original. Kimi K3 is one of the evaluated
models, so in probe 1 it translates its own texts; the report splits that model out.

  3. consistency    (no translation) every AI text in every language classified with
                      --classifier: the language's McFadden R^2 for the label, and how often
                      the same item (model, track, framing, paraphrase, statement) gets the
                      same label in every language. Shown next to the stored labels.

Translations are cached in data/translation_probe/translations.json, so reruns only call
the API for new rows; --offline never calls it (a cluster job: uncached rows are skipped).
Classification runs on the GPU if there is one, else CPU (~1-2 texts/s).

Evaluating a new checkpoint: --reclassify re-labels the AI originals and their English
partners with --classifier (otherwise the stored labels of the results tree are used,
which came from the checkpoint that classified it). --train_keys <track>/train.parquet
drops probe-2 source speeches that are in that checkpoint's train split.

Usage, from the repository root:
  python analysis/translation_probe.py
  python analysis/translation_probe.py --languages sk,ro --n_ai 25 --n_native 60
  python analysis/translation_probe.py --n_ai 15 --n_native 50 --offline --reclassify --probes 1,2,3 \
      --classifier runs/<run>/model --train_keys "data/EuroParl Custom/<track>/train.parquet" \
      --report runs/<run>/results_translation_probe.txt
"""
import argparse
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analyze_all import MODEL_DIRS
from analysis.core import REASONS_CLASSIFIED_CSV, SPEECHES_CLASSIFIED_CSV, find_csvs
from utils import configure_stdout
from utils.api_caller import call_api

configure_stdout()

DEFAULT_RESULTS = "data/euandi_2024_results_new_ni_fixed_center"
DEFAULT_CLASSIFIER = "models/national-k4-logitadj_ni/model"
DEFAULT_TEST = "data/EuroParl Custom/clusters_k4_national_ni/test.parquet"
OUT_DIR = Path("data/translation_probe")
PARTY = "EU Party"
SHORT = {"Liberal-conservative center-right": "Lib-con", "Progressive federalists": "Prog-fed",
         "Radical left": "Rad-left", "Sovereigntist right": "Sov-right"}
GROUPS = list(SHORT.values())
NAMES = {"en": "English", "sk": "Slovak", "cs": "Czech", "ro": "Romanian", "it": "Italian", "pl": "Polish",
         "es": "Spanish", "fr": "French", "pt": "Portuguese", "hu": "Hungarian", "sl": "Slovenian",
         "bg": "Bulgarian", "de": "German", "et": "Estonian", "hr": "Croatian"}
# EU&I file codes -> ISO (the parquet splits use ISO)
EUANDI_CODE = {"cs": "cz", "da": "dk", "et": "ee", "el": "gr", "sl": "si", "sv": "se"}
NAMES_ALL = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "nl", "pl",
             "pt", "ro", "sk", "sl", "sv"]
# probe 2: where native speeches come from, and where the target's own natives go
SOURCES = {"sk": ["cs", "pl"], "ro": ["it", "es", "fr", "pt"], "hu": ["pl", "cs"], "sl": ["cs", "pl"],
           "bg": ["cs", "pl"]}
NEIGHBOUR = {"sk": "cs", "ro": "it", "hu": "pl", "sl": "cs", "bg": "cs"}
PROMPT = ("Translate the following {src} text into {tgt}. Keep the meaning, tone and register; do not "
          "summarise, comment or add anything. Output only the translation.\n\n{text}")
TEXT_PREFIX = {"speeches": "answer", "reasons": "reason"}

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


# --------------------------------------------------------------------------- #
# translation (Kimi K3, cached)
# --------------------------------------------------------------------------- #

class Translator:
    def __init__(self, model, workers, path, offline=False):
        self.model, self.workers, self.path, self.offline = model, workers, path, offline
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.lock = threading.Lock()

    @staticmethod
    def key(text, src, tgt):
        return hashlib.sha1(f"{src}>{tgt}\n{text}".encode("utf-8")).hexdigest()

    def _one(self, text, src, tgt):
        prompt = PROMPT.format(src=NAMES[src], tgt=NAMES[tgt], text=text)
        last = ""
        for attempt in range(6):
            try:
                choice = call_api(prompt, self.model, max_tokens=8000, reasoning_effort="none")["choices"][0]
            except RuntimeError as err:
                last = str(err)[:200]
                time.sleep(15 * (attempt + 1) if "429" in last else 2)
                continue
            out = (choice["message"].get("content") or "").strip()
            if out and choice.get("finish_reason") == "stop":
                return out
            last = f"finish_reason={choice.get('finish_reason')}, {len(out)} chars"
        print(f"  translation failed ({src}>{tgt}): {last}")
        return None

    def __call__(self, texts, src, tgt):
        """texts: list; src/tgt: scalar or per-row list. Returns list of translations."""
        n = len(texts)
        src = src if isinstance(src, list) else [src] * n
        tgt = tgt if isinstance(tgt, list) else [tgt] * n
        keys = [self.key(t, s, g) for t, s, g in zip(texts, src, tgt)]
        todo = {k: (t, s, g) for k, t, s, g in zip(keys, texts, src, tgt) if k not in self.cache}
        if todo and self.offline:
            print(f"  --offline: {len(todo)} of {len(keys)} texts are not cached and are skipped")
            todo = {}
        if todo:
            print(f"  translating {len(todo)} texts with {self.model} ({len(keys) - len(todo)} cached)")
            with ThreadPoolExecutor(self.workers) as pool:
                futures = {pool.submit(self._one, *args): k for k, args in todo.items()}
                for i, fut in enumerate(as_completed(futures), 1):
                    out = fut.result()
                    with self.lock:
                        if out is not None:
                            self.cache[futures[fut]] = out
                        if i % 50 == 0 or i == len(futures):
                            self.save()
                            print(f"    {i}/{len(futures)}")
        return [self.cache.get(k) for k in keys]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

class Labeller:
    def __init__(self, classifier_dir, batch=16):
        import torch
        from analysis.classify_speeches import load_classifier, predict_class_logits
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clf = load_classifier(classifier_dir, self.device)
        self.predict, self.batch = predict_class_logits, batch
        self.names = np.array([SHORT[l] for l in self.clf.labels])

    def __call__(self, texts):
        texts = list(texts)
        logits = [self.predict(texts[i:i + self.batch], self.clf, self.device)
                  for i in range(0, len(texts), self.batch)]
        logits = np.vstack(logits) if logits else np.zeros((0, len(self.names)))
        return self.names[(logits + self.clf.biases).argmax(1)]


def shares(labels):
    return (100 * pd.Series(labels).value_counts(normalize=True).reindex(GROUPS).fillna(0)).round(1)


# --------------------------------------------------------------------------- #
# 1. translate-test on AI text
# --------------------------------------------------------------------------- #

def load_ai(results_dir, models, languages, labels):
    """One row per AI text in languages + en (every language if None), with the stored (biased) label.

    Reads only the needed columns."""
    if languages is None:
        codes = {EUANDI_CODE.get(l, l): l for l in NAMES_ALL}
    else:
        codes = {EUANDI_CODE.get(l, l): l for l in languages + ["en"]}
    keep = re.compile(rf"^(answer|reason|party_prob_.+)_({'|'.join(codes)})(_negated)?_v\d+$")
    frames = []
    for model in models:
        for track, pattern in [("speeches", SPEECHES_CLASSIFIED_CSV), ("reasons", REASONS_CLASSIFIED_CSV)]:
            for framing, path in find_csvs(results_dir / model, pattern).items():
                raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=lambda c: bool(keep.match(c)))
                for column in raw.columns:
                    m = re.match(rf"^{TEXT_PREFIX[track]}_(?P<code>[a-z]{{2}})(?P<neg>_negated)?_v(?P<v>\d+)$", column)
                    if not m or m["code"] not in codes:
                        continue
                    suffix = column[len(TEXT_PREFIX[track]):]
                    probs = np.column_stack([pd.to_numeric(raw.get(
                        f"party_prob_{re.sub(r'[^0-9A-Za-z]+', '_', l).strip('_')}{suffix}"), errors="coerce")
                        for l in labels])
                    ok = ~np.isnan(probs).any(1)
                    pred = np.full(len(raw), None, dtype=object)
                    pred[ok] = np.array([SHORT[l] for l in labels])[probs[ok].argmax(1)]
                    frames.append(pd.DataFrame({
                        "model": model, "track": track, "framing": framing + (m["neg"] or ""),
                        "paraphrase": int(m["v"]), "statement_idx": np.arange(len(raw)),
                        "language": codes[m["code"]], "text": raw[column], "pred": pred}))
    df = pd.concat(frames, ignore_index=True)
    return df[df.text.notna() & df.pred.notna() & (df.text.astype(str).str.split().str.len() >= 5)]


def translate_test(args, labeller, translate):
    labels = labeller.clf.labels
    df = load_ai(Path(args.results_dir), [m for m in args.models.split(",") if m], args.languages, labels)
    key = ["model", "track", "framing", "paraphrase", "statement_idx"]
    en = df[df.language == "en"].set_index(key)[["pred", "text"]].rename(
        columns={"pred": "en_pred", "text": "en_text"})
    rows = []
    for language in args.languages:
        g = df[df.language == language]
        sample = g.sample(frac=1, random_state=args.seed).groupby(["model", "track"]).head(args.n_ai)
        sample = sample.join(en, on=key)
        sample["text_en"] = translate(sample.text.tolist(), language, "en")
        rows.append(sample)
    s = pd.concat(rows, ignore_index=True)
    s = s[s.text_en.notna()].copy()
    if args.reclassify:
        print(f"  re-classifying {len(s)} originals and their English partners with {args.classifier}")
        s["pred"] = labeller(s.text)
        paired = s.en_text.notna()
        s.loc[paired, "en_pred"] = labeller(s.en_text[paired])
    print(f"  classifying {len(s)} translated AI texts (+ a {args.check_n}-row recheck of stored labels)")
    s["pred_en_mt"] = labeller(s.text_en)
    check = s.sample(min(args.check_n, len(s)), random_state=args.seed)
    recheck = (labeller(check.text) == check.pred).mean()

    say(f"\n{'=' * 100}\n[1] TRANSLATE-TEST: AI text -> English (Kimi K3), classified again\n{'=' * 100}")
    say(f"stored label reproduced on {len(check)} re-classified originals: {recheck:.1%}")
    for language, g in s.groupby("language"):
        for name, sub in [("all models", g), ("without kimi-k3", g[g.model != "kimi-k3"])]:
            for track, t in [("both tracks", sub)] + list(sub.groupby("track")):
                paired = t.en_pred.notna()
                say(f"\n  {language} -> en, {name}, {track}: n={len(t)}, label changed by translation "
                    f"{(t.pred != t.pred_en_mt).mean():.0%}")
                say("  " + pd.DataFrame({f"{language} original": shares(t.pred),
                                         f"{language}->en (MT)": shares(t.pred_en_mt),
                                         "en partner": shares(t.en_pred[paired])}).T.to_string()
                    .replace("\n", "\n  "))
                say(f"  agreement with the English partner text: original {(t.pred == t.en_pred)[paired].mean():.0%}, "
                    f"after MT {(t.pred_en_mt == t.en_pred)[paired].mean():.0%}")
        say(f"\n  {language}: original label (rows) -> label after MT into English (cols)")
        say("  " + pd.crosstab(g.pred, g.pred_en_mt).to_string().replace("\n", "\n  "))
        say(f"\n  {language}: Sov-right share by model, original / after MT / en partner")
        say("  " + g.groupby("model").apply(lambda d: pd.Series({
            "orig": (d.pred == "Sov-right").mean(), "mt": (d.pred_en_mt == "Sov-right").mean(),
            "en": (d.en_pred == "Sov-right").mean(), "lib_orig": (d.pred == "Lib-con").mean(),
            "lib_mt": (d.pred_en_mt == "Lib-con").mean()})).mul(100).round(0).astype(int).to_string()
            .replace("\n", "\n  "))
    s.drop(columns=["en_text"]).to_csv(Path(args.report).with_name(Path(args.report).stem + "_translate_test.csv"),
                                       index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 2. counterfactual translation of native EP speeches
# --------------------------------------------------------------------------- #

def stratified(d, n, seed):
    return d.sample(frac=1, random_state=seed).groupby("gold").head(n)


def counterfactual(args, labeller, translate):
    test = pd.read_parquet(args.test)
    n_lang = test.groupby(["speaker", "date"])["language"].transform("nunique")
    test["origin"] = np.where(n_lang <= 2, "original", "translated")
    test["gold"] = test[PARTY].map(SHORT)
    native = test[test.origin == "original"]
    in_train = None
    if args.train_keys:
        train = pd.read_parquet(args.train_keys, columns=["speaker", "date"]).drop_duplicates()
        in_train = set(zip(train.speaker, pd.to_datetime(train.date)))
    say(f"\n{'=' * 100}\n[2] COUNTERFACTUAL: native EP speeches translated into / out of the target language\n"
        f"{'=' * 100}")
    say("accuracy = predicted cluster == the speaker's national-party cluster")
    out = []
    for target in args.languages:
        sets = {
            f"native {'/'.join(SOURCES[target])} -> {target}":
                (stratified(native[native.language.isin(SOURCES[target])], args.n_native, args.seed), target),
            f"native {target} -> {NEIGHBOUR[target]}":
                (stratified(native[native.language == target], args.n_native * 2, args.seed), NEIGHBOUR[target]),
        }
        human = stratified(test[(test.language == target) & (test.origin == "translated")], args.n_native, args.seed)
        human_pred = labeller(human.text)
        say(f"\n--- target {target} ---")
        say(f"  baseline, human EP translations into {target} (test): n={len(human)}, "
            f"accuracy {(human_pred == human.gold).mean():.0%}")
        say("  " + pd.DataFrame({"acc": pd.Series(human_pred == human.gold.to_numpy()).groupby(human.gold.to_numpy()).mean(),
                                 "pred share": shares(human_pred) / 100}).T.round(2).to_string()
            .replace("\n", "\n  "))
        for name, (d, tgt) in sets.items():
            d = d.copy()
            d["tgt"] = tgt
            d["text_mt"] = translate(d.text.tolist(), d.language.tolist(), tgt)
            d = d[d.text_mt.notna()]
            if in_train is not None:
                seen = np.array([(sp, pd.Timestamp(dt)) in in_train for sp, dt in zip(d.speaker, d.date)], dtype=bool)
                if seen.any():
                    say(f"  ({name}: {seen.sum()} source speeches are in --train_keys and are left out)")
                d = d[~seen]
            print(f"  classifying {name}: {len(d)} originals + translations")
            d["pred_orig"], d["pred_mt"] = labeller(d.text), labeller(d.text_mt)
            d["set"] = name
            out.append(d)
            say(f"\n  {name}: n={len(d)}  accuracy original {(d.pred_orig == d.gold).mean():.0%} -> "
                f"after MT {(d.pred_mt == d.gold).mean():.0%}; label changed {(d.pred_orig != d.pred_mt).mean():.0%}")
            per = d.groupby("gold").apply(lambda g: pd.Series({
                "n": len(g), "acc_orig": (g.pred_orig == g.name).mean(), "acc_mt": (g.pred_mt == g.name).mean(),
                **{f"mt->{c}": (g.pred_mt == c).mean() for c in GROUPS}}))
            say("  " + per.round(2).to_string().replace("\n", "\n  "))
            say("  original label (rows) -> label after MT (cols)")
            say("  " + pd.crosstab(d.pred_orig, d.pred_mt).to_string().replace("\n", "\n  "))
    pd.concat(out).to_csv(Path(args.report).with_name(Path(args.report).stem + "_counterfactual.csv"),
                          index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# 3. language dependence and cross-language consistency on all AI text
# --------------------------------------------------------------------------- #

def mcfadden_language(labels, languages):
    """McFadden R^2 of label ~ language (saturated multinomial: per-language label shares)."""
    df = pd.DataFrame({"y": np.asarray(labels), "lang": np.asarray(languages)})
    ll0 = np.log(df.y.map(df.y.value_counts(normalize=True))).sum()
    p_lang = df.groupby(["lang", "y"]).size() / df.groupby("lang").size()
    ll1 = np.log(p_lang.loc[list(zip(df.lang, df.y))].to_numpy()).sum()
    return 1 - ll1 / ll0


def consistency(args, labeller):
    df = load_ai(Path(args.results_dir), [m for m in args.models.split(",") if m], None, labeller.clf.labels)
    print(f"  classifying all {len(df):,} AI texts with {args.classifier}")
    df["new"] = labeller(df.text)
    key = ["model", "track", "framing", "paraphrase", "statement_idx"]
    say(f"\n{'=' * 100}\n[3] CONSISTENCY: all AI text, {df.language.nunique()} languages, {len(df):,} texts\n"
        f"{'=' * 100}")
    say("stored = labels in the results tree (the checkpoint that classified it); new = --classifier")
    for track, g in [("both tracks", df)] + list(df.groupby("track")):
        rows = {}
        for name, column in (("stored", "pred"), ("new", "new")):
            mode_share = g.groupby(key)[column].agg(lambda s: s.value_counts(normalize=True).iloc[0])
            rows[name] = {"language McFadden R2": mcfadden_language(g[column], g.language),
                          "agreement with item mode": mode_share.mean(),
                          "items unanimous": (mode_share == 1).mean()}
        say(f"\n  {track}:")
        say("  " + pd.DataFrame(rows).round(3).to_string().replace("\n", "\n  "))
    say("\n  label shares per language (%), new classifier; stored Sov-right / Lib-con for comparison")
    tab = pd.crosstab(df.language, df.new, normalize="index").reindex(columns=GROUPS).fillna(0).mul(100).round(0)
    old = pd.crosstab(df.language, df.pred, normalize="index").reindex(columns=GROUPS).fillna(0).mul(100).round(0)
    tab["stored Sov-right"], tab["stored Lib-con"] = old["Sov-right"], old["Lib-con"]
    say("  " + tab.astype(int).to_string().replace("\n", "\n  "))


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS)
    parser.add_argument("--models", default=",".join(MODEL_DIRS))
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--test", default=DEFAULT_TEST)
    parser.add_argument("--languages", default="sk,ro", type=lambda s: s.split(","))
    parser.add_argument("--translator", default="kimi-k3")
    parser.add_argument("--workers", type=int, default=4, help="the API key allows 4 parallel requests")
    parser.add_argument("--n_ai", type=int, default=25, help="AI texts per (language, model, track)")
    parser.add_argument("--n_native", type=int, default=60, help="native speeches per (target, cluster)")
    parser.add_argument("--check_n", type=int, default=60, help="originals re-classified to verify stored labels")
    parser.add_argument("--probes", default="1,2", help="any of 1,2,3")
    parser.add_argument("--reclassify", action="store_true",
                        help="probe 1: re-label originals and English partners with --classifier")
    parser.add_argument("--train_keys", default="", help="probe 2: drop sources in this train split")
    parser.add_argument("--offline", action="store_true", help="never call the API; skip uncached rows")
    parser.add_argument("--batch", type=int, default=16, help="classifier batch (64+ on a GPU)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", default="results_translation_probe.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    translate = Translator(args.translator, args.workers, OUT_DIR / "translations.json", offline=args.offline)
    labeller = Labeller(args.classifier, batch=args.batch)
    say(f"translator {args.translator}; classifier {args.classifier} on {labeller.device}; "
        f"languages {args.languages}")
    if "1" in args.probes:
        translate_test(args, labeller, translate)
    if "2" in args.probes:
        counterfactual(args, labeller, translate)
    if "3" in args.probes:
        consistency(args, labeller)
    Path(args.report).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()

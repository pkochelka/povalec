#!/usr/bin/env python3
"""Style of texts broken down by the cluster the classifier assigns.

Question: does the cluster classifier route texts by how they are written (sentence
length, nominal style, pronouns, readability, passive voice, punctuation) as well as by
what they say? Sibling of analysis/sentiment_by_class.py; it reuses that script's test
loader and its stage-1 classifier cache, so a test run there makes this one skip the
classifier.

Two sources (--source):
  test   held-out EP speeches of the national k=4 split, with gold (speaker) and
         predicted cluster. As in sentiment_by_class:
           M_pred:  feature ~ pred + language
           M_gold:  feature ~ gold + language
           M_both:  feature ~ pred + gold + language
         The M_both pred effect is the style the classifier adds beyond who actually
         spoke. Cluster bootstrap over speeches (speaker + date), since pre-2013 EP
         translations put one speech in up to 21 languages.
  llm    the LLM answers in --results_dir (speeches and reasons tracks), as loaded by
         analysis/classifier_ngram_analysis.py, predicted cluster only:
           L_lang:  feature ~ pred + language
           L_full:  feature ~ pred + language x statement x stance-level FE + prompt + model
         L_full asks whether the style-cluster link survives holding content and the
         answering model fixed. Cluster bootstrap over the 30 statements.

Features, one row per text (cached per language under --out_dir, keyed by text hash):
  syntax       log words, mean / SD sentence length, noun, verb, adjective, adverb and
               proper-noun rates, log noun/verb ratio, mean and max dependency-tree depth,
               mean dependency distance, predicates and subordinators per sentence
  function     function-word share; pronouns (all, 1sg, 1pl, 2nd, 3rd person) where 1st
               person also counts finite verbs without an overt subject (pro-drop);
               articles and definite articles (article languages only), coordinating /
               subordinating conjunctions, adpositions, negation
  readability  Flesch reading ease (textstat; native constants for de/es/fr/it/nl/hu,
               English constants elsewhere), LIX, syllables per word, mean word length,
               share of words > 6 letters, MATTR(50); English only: Flesch-Kincaid grade,
               Gunning fog
  register     passive share of predicates, Heylighen formality, punctuation per 100
               words (all, comma, ;/:, dash, parentheses, quotes, ?, !), markdown / list
               markers; English only: necessity / possibility / prediction modals,
               hedges, boosters
Parsing: spaCy *_sm pipelines where they exist (15 of the 21 languages), Stanza for
bg/cs/et/hu/lv/sk. Label schemes differ (UD; ClearNLP for en; TIGER for de), so every
feature is z-scored within language and every model has language fixed effects: effects
read as within-language SDs, comparable across features. A feature that is constant in
a language (articles in Czech) is dropped for that language only.

Blocks of the report, per source (and per track for llm):
  [1] class profiles  effect-coded class effects (deviation from the unweighted mean of
                      the four classes), 95% cluster-bootstrap CIs, BH-adjusted q < .05
                      marked *; raw means by predicted class in natural units
  [2] axis summary    mean |effect| and the number of significant effects per axis
  [3] prediction      grouped 5-fold CV multinomial logit: pseudo-R2 of the predicted
                      (and, for test, gold) class on controls, + style, + each axis
  [4] robustness      test: M_both in original / translated / English-only subsets;
                      llm: how much of the between-model share variance the style
                      features absorb
  [5] agreement       correlation between the effect profiles of all fits (does the
                      classifier's Sov-right style on LLM text look like real Sov-right
                      speakers' style?)

Outputs (--out_dir, default data/style_by_class):
  features_<lang>.parquet          per-text features (cache, both sources)
  style_effects_<tag>.csv          every class effect with CI and q
  style_prediction_<tag>.csv       block 3
  style_by_class_<tag>.png         heatmap of the class effects
and results_style_by_class[_<tag>].txt in the repo root.

Dependencies beyond the project's: spacy, textstat, stanza and the spaCy pipelines
(python -m spacy download en_core_web_sm ...); see analysis/slurm_style_by_class.sh.

Usage, from the repository root:
  python analysis/style_by_class.py                                  # both sources
  python analysis/style_by_class.py --source test --limit 300 --llm_limit 2000 --bootstrap 100   # smoke
  python analysis/style_by_class.py --stage features --languages sk,cs   # parse only
"""
import argparse
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from analysis.core.labels import party_sort_key, short_party
from analysis.sentiment_by_class import (
    DEFAULT_CLASSIFIER,
    DEFAULT_DATA_DIR,
    classify,
    effect_code,
    load_test,
    sha,
    slug,
)
from utils import configure_stdout

configure_stdout()

DEFAULT_RESULTS = "data/euandi_2024_results_new_ni_fixed_center"
EUANDI_TO_ISO = {"cz": "cs", "dk": "da", "ee": "et", "gr": "el", "si": "sl", "se": "sv"}
SPACY_MODELS = {"en": "en_core_web_sm", **{l: f"{l}_core_news_sm" for l in
                ["da", "de", "el", "es", "fi", "fr", "it", "lt", "nl", "pl", "pt", "ro", "sl", "sv"]}}
STANZA_LANGUAGES = {"bg", "cs", "et", "hu", "lv", "sk"}
MIN_WORDS = 10
ARTICLE_FLOOR = 0.5  # articles per 100 words below which a language counts as article-less

# ---- dependency labels across UD, ClearNLP (spaCy en) and TIGER (spaCy de) ----
PASSIVE_DEPS = {"aux:pass", "auxpass", "nsubj:pass", "nsubjpass", "csubj:pass", "csubjpass", "expl:pass"}
SUBJECT_DEPS = {"nsubj", "nsubj:pass", "nsubjpass", "csubj", "csubj:pass", "csubjpass", "expl", "sb", "sbp"}
AUX_DEPS = {"aux", "aux:pass", "auxpass", "cop", "aux:tense"}
NEG_DEPS = {"neg", "ng"}
FUNCTION_UPOS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
GERMAN_PARTICIPLES = {"VVPP", "VAPP", "VMPP"}

# ---- English lexicons (register block) ----
MODALS = {"nec": {"must", "should", "shall", "ought", "need"},
          "poss": {"can", "could", "may", "might", "ca"},
          "pred": {"will", "would", "'ll", "'d", "wo"}}
SEMI_MODAL = re.compile(r"\b(?:ha(?:ve|s|d) to|need(?:s|ed)? to)\b", re.I)  # "ought" is tagged MD
HEDGES = re.compile(r"\b(?:perhaps|possibly|probably|likely|unlikely|arguably|apparently|seemingly|presumably"
                    r"|somewhat|relatively|fairly|rather|generally|typically|usually|often|sometimes|largely"
                    r"|mostly|partly|partially|potentially|approximately|roughly|suggests?|suggested|seems?"
                    r"|seemed|appears?|appeared|tends? to|to (?:some|a certain) extent|in some cases"
                    r"|it is possible|i think|i believe|in my view|in my opinion|it depends|depends on)\b", re.I)
BOOSTERS = re.compile(r"\b(?:clearly|certainly|definitely|undoubtedly|obviously|indeed|of course|surely|truly"
                      r"|absolutely|always|never|undeniably|in fact|without (?:a )?doubt|no doubt|demonstrably"
                      r"|essential|crucial|vital|firmly|strongly|fundamental(?:ly)?)\b", re.I)
PUNCT = {"comma": r",", "semi_colon": r"[;:]", "dash": r"[–—]|\s-\s", "paren": r"[()\[\]]",
         "quote": r"[\"“”„«»]", "question": r"[?;]", "exclaim": r"!"}
MARKDOWN = re.compile(r"\*\*|__|^\s*(?:[-*•]|\d+[.)])\s+|^\s*#+\s", re.M)

AXES = {
    "syntax": ["log_words", "sent_len", "sent_len_sd", "noun", "verb", "adj", "adv", "propn", "noun_verb",
               "depth_mean", "depth_max", "dep_dist", "pred_per_sent", "subord_per_sent"],
    "function": ["function_share", "pron", "pron_1sg", "pron_1pl", "pron_2", "pron_3", "art", "art_def",
                 "cconj", "sconj", "adp", "neg"],
    "readability": ["flesch", "lix", "syll_per_word", "word_len", "long_words", "mattr", "fk_grade", "fog"],
    "register": ["passive", "formality", "punct", *[f"p_{k}" for k in PUNCT], "markdown",
                 "modal_nec", "modal_poss", "modal_pred", "hedge", "booster"],
}
FEATURES = [f for fs in AXES.values() for f in fs]
AXIS_OF = {f: a for a, fs in AXES.items() for f in fs}
ENGLISH_ONLY = {"fk_grade", "fog", "modal_nec", "modal_poss", "modal_pred", "hedge", "booster"}
ARTICLE_FEATURES = {"art", "art_def"}
LABEL = {"log_words": "log words", "sent_len": "sentence length", "sent_len_sd": "sentence length SD",
         "noun": "nouns /100w", "verb": "verbs /100w", "adj": "adjectives /100w", "adv": "adverbs /100w",
         "propn": "proper nouns /100w", "noun_verb": "log noun/verb", "depth_mean": "tree depth (mean)",
         "depth_max": "tree depth (max)", "dep_dist": "dependency distance", "pred_per_sent": "predicates /sent",
         "subord_per_sent": "subordinators /sent", "function_share": "function words %", "pron": "pronouns /100w",
         "pron_1sg": "1st sg /100w", "pron_1pl": "1st pl /100w", "pron_2": "2nd person /100w",
         "pron_3": "3rd pers. pron. /100w", "art": "articles /100w", "art_def": "def. articles /100w",
         "cconj": "coord. conj. /100w", "sconj": "subord. conj. /100w", "adp": "adpositions /100w",
         "neg": "negation /100w", "flesch": "Flesch reading ease", "lix": "LIX", "syll_per_word": "syllables /word",
         "word_len": "word length (chars)", "long_words": "long words %", "mattr": "MATTR-50",
         "fk_grade": "Flesch-Kincaid grade (en)", "fog": "Gunning fog (en)", "passive": "passive % of predicates",
         "formality": "Heylighen formality", "punct": "punctuation /100w", "p_comma": "commas /100w",
         "p_semi_colon": "; and : /100w", "p_dash": "dashes /100w", "p_paren": "parentheses /100w",
         "p_quote": "quotes /100w", "p_question": "? /100w", "p_exclaim": "! /100w",
         "markdown": "markdown markers /100w", "modal_nec": "necessity modals (en)",
         "modal_poss": "possibility modals (en)", "modal_pred": "will/would (en)", "hedge": "hedges /100w (en)",
         "booster": "boosters /100w (en)"}

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


def indent(text, by="  "):
    return by + str(text).replace("\n", "\n" + by)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
# A parsed text is a list of sentences; a sentence a list of tokens
# (text, upos, xpos, dep, head index in the sentence or -1, feats dict, lemma).

def from_spacy(doc):
    sents = []
    for s in doc.sents:
        toks = [t for t in s if not t.is_space]
        pos = {t.i: j for j, t in enumerate(toks)}
        sents.append([(t.text, t.pos_, t.tag_, t.dep_.lower(), -1 if t.head.i == t.i else pos.get(t.head.i, -1),
                       t.morph.to_dict(), t.lemma_.lower()) for t in toks])
    return [s for s in sents if s]


def from_stanza(doc):
    def feats(f):
        return dict(kv.split("=", 1) for kv in f.split("|") if "=" in kv) if f else {}
    return [[(w.text, w.upos, w.xpos or "", w.deprel or "", w.head - 1, feats(w.feats), (w.lemma or "").lower())
             for w in s.words] for s in doc.sentences if s.words]


def tree_depths(heads):
    depth = [0] * len(heads)
    for i in range(len(heads)):
        j, d = i, 1
        while heads[j] != -1 and d <= len(heads):
            j, d = heads[j], d + 1
        depth[i] = d
    return depth


def mattr(words, window=50):
    if not words:
        return np.nan
    if len(words) <= window:
        return len(set(words)) / len(words)
    counts = Counter(words[:window])
    total = len(counts)
    for i in range(window, len(words)):
        counts[words[i]] += 1
        old = words[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        total += len(counts)
    return total / ((len(words) - window + 1) * window)


def features(sents, text, lang, textstat):
    f = {}
    pos = Counter()
    words, sent_lens, depth_means, depth_maxes, dists = [], [], [], [], []
    n_pred = n_subord = n_passive = 0
    person = Counter()
    n_art = n_art_def = n_neg = 0
    modals = Counter()
    for s in sents:
        heads = [t[4] for t in s]
        children = [[] for _ in s]
        for i, h in enumerate(heads):
            if h >= 0:
                children[h].append(i)
        depth = tree_depths(heads)
        content = [i for i, t in enumerate(s) if t[1] not in ("PUNCT", "SYM", "SPACE")]
        if not content:
            continue
        sent_lens.append(len(content))
        depth_maxes.append(max(depth[i] for i in content))
        dists += [abs(i - heads[i]) for i in content if heads[i] >= 0]
        passive = set()
        for i in content:
            text_i, upos, xpos, dep, head, ft, lemma = s[i]
            words.append(text_i.lower())
            pos[upos] += 1
            if upos == "VERB" or (upos == "AUX" and dep not in AUX_DEPS) or any(s[c][3] == "cop" for c in children[i]):
                n_pred += 1
            if upos == "SCONJ" or ft.get("PronType") in ("Rel", "Int,Rel"):
                n_subord += 1
            if ft.get("Voice") == "Pass":
                passive.add(i)
            if dep in PASSIVE_DEPS and head >= 0:
                passive.add(head)
            if lang == "de" and xpos in GERMAN_PARTICIPLES and head >= 0 and s[head][6] == "werden":
                passive.add(i)
            if ft.get("Polarity") == "Neg" or dep in NEG_DEPS:
                n_neg += 1
            if upos == "DET" and ft.get("PronType") == "Art":
                n_art += 1
                n_art_def += ft.get("Definite") == "Def"
            p = ft.get("Person")
            if p:
                num = ft.get("Number", "")
                pronominal = upos in ("PRON", "DET")
                # pro-drop: a finite predicate's person counts only if it has no overt subject
                prodrop = (upos in ("VERB", "AUX") and dep not in AUX_DEPS and ft.get("VerbForm") == "Fin"
                           and not any(s[c][3] in SUBJECT_DEPS for c in children[i]))
                if pronominal or prodrop:
                    if p == "1":
                        person["1pl" if num == "Plur" else "1sg"] += 1
                    elif p == "2":
                        person["2"] += 1
                if upos == "PRON" and p == "3" and ft.get("PronType") == "Prs":
                    person["3"] += 1
            if lang == "en" and xpos == "MD":
                for kind, lex in MODALS.items():
                    modals[kind] += text_i.lower() in lex
        n_passive += len(passive)
        depth_means.append(np.mean([depth[i] for i in content]))
    n = len(words)
    if n < MIN_WORDS:
        return None
    per100 = lambda c: 100 * c / n
    f["log_words"] = math.log(n)
    f["sent_len"] = float(np.mean(sent_lens))
    f["sent_len_sd"] = float(np.std(sent_lens))
    for name, tags in [("noun", ["NOUN"]), ("verb", ["VERB"]), ("adj", ["ADJ"]), ("adv", ["ADV"]),
                       ("propn", ["PROPN"]), ("pron", ["PRON"]), ("cconj", ["CCONJ"]), ("sconj", ["SCONJ"]),
                       ("adp", ["ADP"])]:
        f[name] = per100(sum(pos[t] for t in tags))
    f["noun_verb"] = math.log((pos["NOUN"] + pos["PROPN"] + 1) / (pos["VERB"] + 1))
    f["depth_mean"] = float(np.mean(depth_means))
    f["depth_max"] = float(np.mean(depth_maxes))
    f["dep_dist"] = float(np.mean(dists)) if dists else np.nan
    f["pred_per_sent"] = n_pred / len(sent_lens)
    f["subord_per_sent"] = n_subord / len(sent_lens)
    f["function_share"] = per100(sum(pos[t] for t in FUNCTION_UPOS))
    f["pron_1sg"], f["pron_1pl"], f["pron_2"], f["pron_3"] = (per100(person[k]) for k in ["1sg", "1pl", "2", "3"])
    f["art"], f["art_def"] = per100(n_art), per100(n_art_def)
    f["neg"] = per100(n_neg)

    letters = [w for w in words if any(ch.isalpha() for ch in w)]
    f["word_len"] = float(np.mean([len(w) for w in letters])) if letters else np.nan
    f["long_words"] = 100 * np.mean([len(w) > 6 for w in letters]) if letters else np.nan
    f["mattr"] = mattr(letters)
    for name, fn in [("flesch", "flesch_reading_ease"), ("lix", "lix"), ("syll_per_word", "avg_syllables_per_word")]:
        try:
            f[name] = float(getattr(textstat, fn)(text))
        except Exception:  # textstat / pyphen gaps for a language
            f[name] = np.nan
    if lang == "en":
        f["fk_grade"] = float(textstat.flesch_kincaid_grade(text))
        f["fog"] = float(textstat.gunning_fog(text))

    f["passive"] = 100 * n_passive / max(n_pred, 1)
    # Heylighen & Dewaele F-score on percentages of words
    pct = lambda t: per100(pos[t])
    f["formality"] = (pct("NOUN") + pct("PROPN") + pct("ADJ") + pct("ADP") + per100(n_art)
                      - pct("PRON") - pct("VERB") - pct("ADV") - pct("INTJ") + 100) / 2
    f["punct"] = per100(sum(1 for ch in text if unicodedata.category(ch).startswith("P")))
    for name, rx in PUNCT.items():
        f[f"p_{name}"] = per100(len(re.findall(rx, text)))
    if lang == "el":  # the Greek question mark is the semicolon glyph
        f["p_question"] += per100(text.count(";"))
        f["p_semi_colon"] -= per100(text.count(";"))
    f["markdown"] = per100(len(MARKDOWN.findall(text)))
    if lang == "en":
        f["modal_nec"] = per100(modals["nec"] + len(SEMI_MODAL.findall(text)))
        f["modal_poss"] = per100(modals["poss"])
        f["modal_pred"] = per100(modals["pred"])
        f["hedge"] = per100(len(HEDGES.findall(text)))
        f["booster"] = per100(len(BOOSTERS.findall(text)))
    return f


class Parser:
    def __init__(self, lang, device, batch):
        import textstat
        self.lang, self.batch = lang, batch
        self.textstat = textstat
        textstat.set_lang(lang)
        if lang in SPACY_MODELS:
            import spacy
            self.kind = "spacy"
            self.nlp = spacy.load(SPACY_MODELS[lang], exclude=["ner", "textcat"])
            self.nlp.max_length = 2_000_000
        elif lang in STANZA_LANGUAGES:
            import stanza
            self.kind = "stanza"
            stanza.download(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
            self.nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse", use_gpu=device == "cuda",
                                       verbose=False, pos_batch_size=3000, depparse_batch_size=3000)
            self.stanza = stanza
        else:
            raise SystemExit(f"no parser for language {lang!r}")

    def __call__(self, texts):
        if self.kind == "spacy":
            docs = (from_spacy(d) for d in self.nlp.pipe(texts, batch_size=self.batch))
        else:
            docs = (from_stanza(d) for d in self.nlp([self.stanza.Document([], text=t) for t in texts]))
        return [features(sents, t, self.lang, self.textstat) for sents, t in zip(docs, texts)]


def _worker_init(lang):
    global _PARSER
    _PARSER = Parser(lang, "cpu", 64)


def _worker(texts):
    return _PARSER(texts)


def extract(texts_by_lang, out_dir, args):
    """{lang: {sha: text}} -> features DataFrame indexed by text_sha, from / into the per-language caches."""
    frames = []
    for lang in sorted(texts_by_lang):
        path = out_dir / f"features_{lang}.parquet"
        cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["text_sha"])
        want = texts_by_lang[lang]
        done = set(cached["text_sha"])
        todo = [h for h in want if h not in done]
        if todo:
            print(f"[{lang}] parsing {len(todo):,} texts ({len(want) - len(todo):,} cached)")
            texts = [want[h] for h in todo]
            chunk = 256
            chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
            rows = []
            if lang in SPACY_MODELS and args.workers > 1:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(args.workers, _worker_init, (lang,)) as pool:
                    for part in tqdm(pool.imap(_worker, chunks), total=len(chunks), desc=lang):
                        rows += part
            else:
                parser = Parser(lang, args.device, 64)
                for part in tqdm(chunks, desc=lang):
                    rows += parser(part)
            new = pd.DataFrame([r if r is not None else {} for r in rows], columns=FEATURES)
            new.insert(0, "text_sha", todo)
            new["parser"] = "spacy" if lang in SPACY_MODELS else "stanza"
            cached = pd.concat([cached, new], ignore_index=True) if len(cached) else new
            out_dir.mkdir(parents=True, exist_ok=True)
            cached.to_parquet(path, index=False)
        frames.append(cached[cached["text_sha"].isin(set(want))].assign(feat_lang=lang))
    return pd.concat(frames, ignore_index=True).drop_duplicates("text_sha").set_index("text_sha")


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #

def load_test_source(args, keep, clf_name):
    df = load_test(args)  # origin is computed on the whole split, before any language filter
    lim = f"_lim{args.limit}" if args.limit else ""
    if keep:
        df = df[df.language.isin(keep)].reset_index(drop=True)
        lim += "_" + "-".join(sorted(keep))
    # same cache file as sentiment_by_class.py when run on the same split
    preds = classify(df, args, args.device, Path(args.clf_cache_dir) / f"classifier_{clf_name}_{args.split}{lim}.parquet")
    d = preds.reset_index(drop=True)
    d["text"] = df["text"].to_numpy()
    d["cluster_id"] = d["speech"]
    return d


def load_llm_source(args, keep):
    from analysis.classifier_ngram_analysis import SHORT, load_texts
    from analysis.analyze_all import MODEL_DIRS
    import json
    manifest = json.loads((Path(args.classifier) / "manifest.json").read_text(encoding="utf-8"))
    order = sorted(manifest["label2id"], key=manifest["label2id"].get)
    biases = np.asarray(manifest.get("biases") or np.zeros(len(order)))
    models = [m.strip() for m in (args.models or ",".join(MODEL_DIRS)).split(",") if m.strip()]
    df = load_texts(Path(args.results_dir), models, order, biases, float(manifest.get("temperature", 1.0)))
    long_name = {v: k for k, v in SHORT.items()}
    # stored labels as deployed, argmax(logits + biases), like the test source's classify()
    df["pred"] = df["pred_biased"].map(long_name)
    df["language"] = df["language"].map(lambda l: EUANDI_TO_ISO.get(l, l))
    if keep:
        df = df[df.language.isin(keep)]
    if args.llm_limit:
        df = df.sample(n=min(args.llm_limit, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)
    df["text_sha"] = df["text"].map(sha)
    df["cluster_id"] = df["statement_idx"]
    # CV folds keep a (model, statement, language) together -- its paraphrase / framing variants are
    # near-duplicates -- but let every statement x stance cell appear in training
    df["cv_group"] = df["model"] + "|" + df["statement_idx"].astype(str) + "|" + df["language"]
    df["cell"] = df["language"] + "|" + df["stmt_stance"]
    return df


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #

def zscore(d):
    """Within-language z-scores; NaN where the feature is undefined for the language."""
    z = pd.DataFrame(index=d.index, columns=FEATURES, dtype=float)
    for lang, g in d.groupby("language"):
        for f in FEATURES:
            v = g[f].astype(float)
            if f in ENGLISH_ONLY and lang != "en":
                continue
            if f in ARTICLE_FEATURES and not v.mean() >= ARTICLE_FLOOR:
                continue
            sd = v.std()
            if not sd > 0 or v.notna().mean() < .9:
                continue
            z.loc[g.index, f] = ((v - v.mean()) / sd).fillna(0)  # rare row gaps -> language mean
    return z


def feature_groups(d, z):
    """Features sharing the same set of defined languages are fitted together."""
    groups = {}
    for f in FEATURES:
        langs = tuple(sorted(d.loc[z[f].notna(), "language"].unique()))
        if langs:
            groups.setdefault(langs, []).append(f)
    return groups


def cluster_sums(D, Y, cl):
    """Per-cluster D'D (C, q, q) and D'Y (C, q, p)."""
    C = cl.max() + 1
    G = sp.csr_matrix((np.ones(len(cl)), (cl, np.arange(len(cl)))), shape=(C, len(cl)))
    xx = np.stack([G @ (D * D[:, [j]]) for j in range(D.shape[1])], 1)
    xy = np.stack([G @ (Y * D[:, [j]]) for j in range(D.shape[1])], 1)
    return xx, xy


def solve(xx, xy, w):
    return np.linalg.lstsq(np.tensordot(w, xx, 1), np.tensordot(w, xy, 1), rcond=None)[0]


def fit(d, z, terms, k, extra, cl, n_boot, seed, demean=None):
    """Effect-coded class effects of each term on every feature.

    extra: list of categorical columns entered as dummies (language FE for test).
    demean: column whose cells absorb a fixed effect by within-cell demeaning
            (cells must be nested in the bootstrap clusters).
    Returns {term: (point (p, k), reps (B, p, k))} and the feature order."""
    out_point, out_reps, feats = [], [], []
    for langs, fs in feature_groups(d, z).items():
        m = d["language"].isin(langs).to_numpy()
        g, Y = d[m], z.loc[m, fs].to_numpy(float)
        blocks = [effect_code(g[f"{t}_code"].to_numpy(), k) for t in terms]
        for col in extra:
            codes = pd.factorize(g[col])[0]
            if codes.max() > 0:
                blocks.append(np.eye(codes.max() + 1)[codes][:, 1:])
        D = np.hstack(blocks)
        if demean is not None:
            cells = pd.factorize(g[demean])[0]
            D = D - pd.DataFrame(D).groupby(cells).transform("mean").to_numpy()
            Y = Y - pd.DataFrame(Y).groupby(cells).transform("mean").to_numpy()
        else:
            D = np.hstack([np.ones((len(g), 1)), D])
        off = 0 if demean is not None else 1
        codes = pd.factorize(g[cl])[0]
        xx, xy = cluster_sums(D, Y, codes)
        C = codes.max() + 1

        def effects(beta):
            res = []
            for t in range(len(terms)):
                b = beta[off + t * (k - 1): off + (t + 1) * (k - 1)]
                res.append(np.vstack([b, -b.sum(0)]).T)  # (p, k)
            return np.stack(res)  # (terms, p, k)

        out_point.append(effects(solve(xx, xy, np.ones(C))))
        rng = np.random.default_rng(seed)
        reps = [effects(solve(xx, xy, np.bincount(rng.integers(0, C, C), minlength=C).astype(float)))
                for _ in range(n_boot)]
        out_reps.append(np.stack(reps) if reps else np.zeros((0, *out_point[-1].shape)))
        feats += fs
    point = np.concatenate(out_point, 1)
    reps = np.concatenate(out_reps, 2)
    return {t: (point[i], reps[:, i]) for i, t in enumerate(terms)}, feats


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def effect_table(source, spec, term, point, reps, feats, classes):
    rows = []
    B = len(reps)
    for i, f in enumerate(feats):
        for j, c in enumerate(classes):
            r = reps[:, i, j]
            p = 2 * min((r <= 0).mean(), (r >= 0).mean())
            rows.append({"source": source, "spec": spec, "term": term, "feature": f, "axis": AXIS_OF[f],
                         "class": c, "effect": point[i, j], "lo": np.percentile(r, 2.5),
                         "hi": np.percentile(r, 97.5), "p": max(p, 1 / B)})
    t = pd.DataFrame(rows)
    t["q"] = bh(t["p"])
    return t


def show_effects(t, classes):
    names = [short_party(c) for c in classes]
    say(f"  {'feature':<28}" + "".join(f"{n:>12}" for n in names))
    for axis, fs in AXES.items():
        sub = t[t.axis == axis]
        if sub.empty:
            continue
        say(f"  -- {axis}")
        for f in fs:
            r = sub[sub.feature == f].set_index("class")
            if r.empty:
                continue
            cells = [f"{r.effect[c]:+.2f}{'*' if r.q[c] < .05 else ' '}" for c in classes]
            say(f"  {LABEL[f]:<28}" + "".join(f"{x:>12}" for x in cells))


def axis_summary(t):
    s = t.assign(abs_effect=t.effect.abs(), sig=t.q < .05).groupby("axis").agg(
        features=("feature", "nunique"), mean_abs=("abs_effect", "mean"), significant=("sig", "sum"),
        cells=("sig", "size"))
    return s.reindex([a for a in AXES if a in s.index]).round(3)


def raw_means(d, by, classes):
    m = d.groupby(by)[FEATURES].mean().reindex(classes).T
    m.columns = [short_party(c) for c in classes]
    m.index = [LABEL[f] for f in m.index]
    return m.dropna(how="all")


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #

def cv_pseudo_r2(y, cats, dense, groups, folds=5):
    """Grouped K-fold multinomial logit; pseudo-R2 = 1 - LL / LL(null, train-fold marginals)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import OneHotEncoder
    blocks = []
    if cats is not None:
        blocks.append(OneHotEncoder(handle_unknown="ignore").fit_transform(cats))
    if dense is not None and dense.shape[1]:
        blocks.append(sp.csr_matrix(dense))
    X = sp.hstack(blocks).tocsr()
    classes, yi = np.unique(y, return_inverse=True)
    ll = ll0 = 0.0
    for tr, te in GroupKFold(folds).split(X, yi, groups):
        model = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], yi[tr])
        P = np.zeros((len(te), len(classes)))
        P[:, model.classes_] = model.predict_proba(X[te])
        ll += np.log(np.clip(P[np.arange(len(te)), yi[te]], 1e-12, 1)).sum()
        prior = np.bincount(yi[tr], minlength=len(classes)) / len(tr)
        ll0 += np.log(np.clip(prior[yi[te]], 1e-12, 1)).sum()
    return 1 - ll / ll0


def prediction(d, z, specs, groups, max_rows, seed):
    """specs: {name: (target column, [categorical columns], [style features])}."""
    rows = []
    if max_rows and len(d) > max_rows:
        keep = np.random.default_rng(seed).choice(len(d), max_rows, replace=False)
        d, z, groups = d.iloc[keep], z.iloc[keep], groups[keep]
        say(f"  (random subsample of {max_rows:,} rows)")
    zf = z.fillna(0)
    for name, (target, cats, style) in specs.items():
        r2 = cv_pseudo_r2(d[target].to_numpy(), d[cats].astype(str).to_numpy() if cats else None,
                          zf[style].to_numpy() if style else None, groups)
        rows.append({"spec": name, "target": target, "pseudo_r2": r2})
        say(f"  {name:<52} {r2:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #

def prepare(d, feats, classes):
    d = d.join(feats[FEATURES], on="text_sha", how="inner")
    d = d[d["log_words"].notna()].reset_index(drop=True)
    for col in ["pred", "gold"]:
        if col in d:
            d[f"{col}_code"] = d[col].map({c: i for i, c in enumerate(classes)})
    return d, zscore(d)


def analyse_test(d, z, classes, args, tag):
    k = len(classes)
    say(f"\n{'#' * 100}\nSOURCE: test split ({args.data_dir}/{args.split}.parquet), n={len(d):,} texts "
        f">= {MIN_WORDS} words, {d.speech.nunique():,} speeches\n{'#' * 100}")
    acc = (d.pred == d.gold).mean()
    say(f"classifier accuracy on these texts {acc:.3f}; origin {(d.origin == 'original').mean():.0%} original")
    say("\nraw means by PREDICTED class (natural units)")
    say(indent(raw_means(d, "pred", classes).round(2).to_string()))
    say("\nraw means by GOLD class")
    say(indent(raw_means(d, "gold", classes).round(2).to_string()))

    tables, profiles = [], {}
    for spec, terms in {"M_pred": ["pred"], "M_gold": ["gold"], "M_both": ["pred", "gold"]}.items():
        res, feats = fit(d, z, terms, k, ["language"], "cluster_id", args.bootstrap, args.seed)
        for term in terms:
            t = effect_table("test", spec, term, *res[term], feats, classes)
            tables.append(t)
            profiles[f"test {spec}:{term}"] = t
    say(f"\n[1] CLASS EFFECTS in within-language SDs (deviation from the mean class; * BH q < .05; "
        f"cluster bootstrap over speeches, B={args.bootstrap})")
    for t in tables:
        say(f"\n{t.spec.iloc[0]}: feature ~ {' + '.join(['pred', 'gold'] if t.spec.iloc[0] == 'M_both' else [t.term.iloc[0]])}"
            f" + language   -> {t.term.iloc[0]} effects")
        show_effects(t, classes)

    say("\n[2] AXIS SUMMARY: mean |effect| (SD units) and BH-significant class effects")
    for t in tables:
        say(f"\n  {t.spec.iloc[0]}:{t.term.iloc[0]}")
        say(indent(axis_summary(t).to_string(), "    "))
    by_name = {f"{t.spec.iloc[0]}:{t.term.iloc[0]}": t for t in tables}
    both = by_name["M_both:pred"]
    shrink = both.effect.abs().sum() / by_name["M_pred:pred"].effect.abs().sum()
    say(f"\n  pred effects holding gold fixed keep {shrink:.0%} of their total |size| (M_both:pred vs M_pred)")

    say("\n[3] PREDICTION: grouped 5-fold CV pseudo-R2 (groups = speeches)")
    style = [f for f in FEATURES if z[f].notna().any()]
    specs = {"pred ~ language": ("pred", ["language"], []),
             "pred ~ language + gold": ("pred", ["language", "gold"], []),
             "pred ~ language + gold + style": ("pred", ["language", "gold"], style),
             "pred ~ language + style": ("pred", ["language"], style),
             "gold ~ language": ("gold", ["language"], []),
             "gold ~ language + style": ("gold", ["language"], style)}
    for axis, fs in AXES.items():
        specs[f"pred ~ language + gold + {axis}"] = ("pred", ["language", "gold"], [f for f in fs if f in style])
    pred_table = prediction(d, z, specs, d["cluster_id"].to_numpy(), args.pred_max, args.seed)
    r = pred_table.set_index("spec").pseudo_r2
    say(f"  style adds {r['pred ~ language + gold + style'] - r['pred ~ language + gold']:+.3f} to pred beyond gold; "
        f"style alone predicts pred {r['pred ~ language + style'] - r['pred ~ language']:+.3f} and gold "
        f"{r['gold ~ language + style'] - r['gold ~ language']:+.3f} over language")

    say("\n[4] ROBUSTNESS: M_both pred effects in subsets (point estimates); r = profile correlation with the "
        "full-sample M_both pred profile")
    full = both.set_index(["feature", "class"]).effect
    subsets = {"original only": d.origin == "original", "translated only": d.origin == "translated",
               "English only": d.language == "en", "correct predictions": d.pred == d.gold}
    sub_tables = []
    for label, mask in subsets.items():
        s, zs = d[mask.to_numpy()].reset_index(drop=True), z[mask.to_numpy()].reset_index(drop=True)
        if s.pred_code.nunique() < k or len(s) < 200:
            say(f"  {label}: too few rows")
            continue
        terms = ["pred"] if label == "correct predictions" else ["pred", "gold"]
        res, feats = fit(s, zs, terms, k, ["language"], "cluster_id", 0, args.seed)
        t = pd.DataFrame([{"feature": f, "class": c, "effect": res["pred"][0][i, j]}
                          for i, f in enumerate(feats) for j, c in enumerate(classes)])
        joined = t.set_index(["feature", "class"]).effect
        common = joined.index.intersection(full.index)
        say(f"  {label:<22} n={len(s):>6}  r={np.corrcoef(joined[common], full[common])[0, 1]:.2f}")
        sub_tables.append(t.assign(subset=label))
    pd.concat(sub_tables).to_csv(Path(args.out_dir) / f"style_robustness_{tag}.csv", index=False)
    return pd.concat(tables), pred_table.assign(source="test"), profiles


def analyse_llm(d, z, classes, args):
    k = len(classes)
    say(f"\n{'#' * 100}\nSOURCE: LLM texts ({args.results_dir}), n={len(d):,} texts >= {MIN_WORDS} words, "
        f"{d.model.nunique()} models\n{'#' * 100}")
    tables, preds, profiles = [], [], {}
    for track, idx in d.groupby("track").groups.items():
        g, zg = d.loc[idx].reset_index(drop=True), z.loc[idx].reset_index(drop=True)
        say(f"\n{'=' * 100}\n{track.upper()} (n={len(g):,})  pred shares "
            f"{ {short_party(c): round(v, 2) for c, v in g.pred.value_counts(normalize=True).items()} }\n{'=' * 100}")
        say("raw means by PREDICTED class (natural units)")
        say(indent(raw_means(g, "pred", classes).round(2).to_string()))
        say(f"\n[1] CLASS EFFECTS in within-language SDs (* BH q < .05; cluster bootstrap over statements, "
            f"B={args.bootstrap})")
        for spec, extra, demean in [("L_lang", [], "language"), ("L_full", ["model", "prompt"], "cell")]:
            res, feats = fit(g, zg, ["pred"], k, extra, "cluster_id", args.bootstrap, args.seed, demean=demean)
            t = effect_table(f"llm-{track}", spec, "pred", *res["pred"], feats, classes)
            tables.append(t)
            profiles[f"llm-{track} {spec}:pred"] = t
            say(f"\n{spec}: feature ~ pred + "
                + ("language" if spec == "L_lang" else "language x statement x stance FE + prompt + model"))
            show_effects(t, classes)
        say("\n[2] AXIS SUMMARY")
        for t in tables[-2:]:
            say(f"\n  {t.spec.iloc[0]}")
            say(indent(axis_summary(t).to_string(), "    "))

        say("\n[3] PREDICTION: grouped 5-fold CV pseudo-R2 (groups = model x statement x language)")
        style = [f for f in FEATURES if zg[f].notna().any()]
        content = ["language", "cell", "prompt"]
        specs = {"pred ~ content": ("pred", content, []),
                 "pred ~ content + model": ("pred", content + ["model"], []),
                 "pred ~ content + style": ("pred", content, style),
                 "pred ~ content + model + style": ("pred", content + ["model"], style),
                 "pred ~ language + style": ("pred", ["language"], style),
                 "pred ~ language": ("pred", ["language"], [])}
        for axis, fs in AXES.items():
            specs[f"pred ~ content + model + {axis}"] = ("pred", content + ["model"], [f for f in fs if f in style])
        say("  content = language + language x statement x stance + prompt")
        p = prediction(g, zg, specs, g["cv_group"].to_numpy(), args.pred_max, args.seed)
        preds.append(p.assign(source=f"llm-{track}"))
        r = p.set_index("spec").pseudo_r2
        say(f"  model adds {r['pred ~ content + model'] - r['pred ~ content']:+.3f} over content; "
            f"after style it adds {r['pred ~ content + model + style'] - r['pred ~ content + style']:+.3f}")

        say("\n[4] MODEL GAPS: between-model variance of predicted-class shares (observed - expected), in-sample")
        mediation(g, zg, style, classes)
    return pd.concat(tables), pd.concat(preds), profiles


def mediation(g, zg, style, classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    cats = OneHotEncoder(handle_unknown="ignore").fit_transform(g[["language", "cell", "prompt"]].astype(str))
    observed = pd.crosstab(g.model, g.pred, normalize="index").reindex(columns=classes, fill_value=0)
    out = {}
    for name, X in [("content", cats), ("content + style", sp.hstack([cats, sp.csr_matrix(zg[style].fillna(0).to_numpy())]).tocsr())]:
        model = LogisticRegression(C=1.0, max_iter=2000).fit(X, g.pred)
        expected = pd.DataFrame(model.predict_proba(X), columns=model.classes_).reindex(columns=classes, fill_value=0)
        expected = expected.groupby(g.model.to_numpy()).mean()
        resid = observed - expected.reindex(observed.index)
        out[name] = float((resid ** 2).to_numpy().sum())
        say(f"  residual variance after {name:<16} {out[name]:.4f}")
    v0 = float(((observed - observed.mean()) ** 2).to_numpy().sum())
    say(f"  raw between-model variance {v0:.4f}; style absorbs "
        f"{1 - out['content + style'] / out['content']:.0%} of what content leaves")


def agreement(profiles):
    say(f"\n{'#' * 100}\n[5] AGREEMENT of effect profiles (Pearson r over shared feature x class cells; "
        f"English-only features excluded)\n{'#' * 100}")
    vecs = {name: t[~t.feature.isin(ENGLISH_ONLY)].set_index(["feature", "class"]).effect
            for name, t in profiles.items()}
    names = list(vecs)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            common = vecs[a].index.intersection(vecs[b].index)
            m.loc[a, b] = np.corrcoef(vecs[a][common], vecs[b][common])[0, 1] if len(common) > 3 else np.nan
    say(indent(m.round(2).to_string()))


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #

def plot(effects, classes, path):
    """Heatmap: features (rows, grouped by axis) x fit/class (columns)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    panels = [("test", "M_gold", "gold", "Human speeches\nby speaker cluster"),
              ("test", "M_pred", "pred", "Human speeches\nby predicted cluster"),
              ("test", "M_both", "pred", "Human speeches\npredicted | speaker"),
              ("llm-speeches", "L_full", "pred", "LLM speeches\npredicted | content, model"),
              ("llm-reasons", "L_full", "pred", "LLM reasons\npredicted | content, model")]
    panels = [p for p in panels if ((effects.source == p[0]) & (effects.spec == p[1])).any()]
    rows = [f for f in FEATURES if f in set(effects.feature)]
    ink, muted, grid, surface = "#1f2328", "#59636e", "#d1d9e0", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("div", ["#174a8b", "#2a78d6", "#f0efec", "#e34948", "#8f2423"])
    lim = 0.5
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels) + 2.6, 0.24 * len(rows) + 1.8),
                             sharey=True, facecolor=surface)
    axes = np.atleast_1d(axes)
    for ax, (source, spec, term, title) in zip(axes, panels):
        t = effects[(effects.source == source) & (effects.spec == spec) & (effects.term == term)]
        M = t.pivot(index="feature", columns="class", values="effect").reindex(index=rows, columns=classes)
        Q = t.pivot(index="feature", columns="class", values="q").reindex(index=rows, columns=classes)
        ax.imshow(M.to_numpy(float), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        for (i, j), v in np.ndenumerate(M.to_numpy(float)):
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=grid)
            elif Q.iat[i, j] < .05:
                ax.text(j, i, f"{v:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > .32 else ink)
        ax.set_xticks(range(len(classes)), [short_party(c) for c in classes], rotation=90, fontsize=7, color=muted)
        ax.set_title(title, fontsize=8, color=ink, loc="left")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        edges = np.cumsum([sum(f in rows for f in fs) for fs in AXES.values()])[:-1]
        for e in edges:
            ax.axhline(e - .5, color="white", lw=2.5)
    axes[0].set_yticks(range(len(rows)), [LABEL[f] for f in rows], fontsize=7, color=ink)
    cbar = fig.colorbar(axes[-1].images[0], ax=axes, fraction=.02, pad=.01)
    cbar.set_label("class effect (within-language SD); numbers: BH q < .05", fontsize=7, color=muted)
    cbar.ax.tick_params(labelsize=6, colors=muted, length=0)
    cbar.outline.set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
    print(f"figure -> {path}")


# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["test", "llm", "both"], default="both")
    parser.add_argument("--stage", choices=["features", "all"], default="all",
                        help="features: parse and cache only (no report)")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS, help="LLM results tree (source llm)")
    parser.add_argument("--models", default="", help="comma list; default analyze_all.MODEL_DIRS")
    parser.add_argument("--languages", default="", help="comma list of ISO codes to restrict to")
    parser.add_argument("--clf_batch", type=int, default=32)
    parser.add_argument("--clf_cache_dir", default="data/sentiment_by_class",
                        help="where sentiment_by_class.py caches stage-1 predictions (shared)")
    parser.add_argument("--workers", type=int, default=1, help="spaCy worker processes")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--pred_max", type=int, default=60000,
                        help="row cap for the CV logits of block 3 (0 = all)")
    parser.add_argument("--limit", type=int, default=0, help="random subsample of the test split")
    parser.add_argument("--llm_limit", type=int, default=0, help="random subsample of the LLM texts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--out_dir", default="data/style_by_class")
    parser.add_argument("--report", default=None, help="default results_style_by_class[_<tag>].txt")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.device:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    manifest = json.loads((Path(args.classifier) / "manifest.json").read_text(encoding="utf-8"))
    classes = sorted(manifest["label2id"], key=party_sort_key)
    clf_name = slug(Path(args.classifier).parent.name if Path(args.classifier).name == "model"
                    else Path(args.classifier).name)
    lim = (f"_lim{args.limit}" if args.limit else "") + (f"_llm{args.llm_limit}" if args.llm_limit else "")
    tag = f"{clf_name}_{args.source}{lim}" + (f"_{args.languages.replace(',', '-')}" if args.languages else "")
    keep = set(args.languages.split(",")) if args.languages else None

    sources = {}
    if args.source in ("test", "both"):
        sources["test"] = load_test_source(args, keep, clf_name)
    if args.source in ("llm", "both"):
        sources["llm"] = load_llm_source(args, keep)

    texts_by_lang = {}
    for d in sources.values():
        for lang, h, t in zip(d.language, d.text_sha, d.text):
            texts_by_lang.setdefault(lang, {})[h] = t
    say(f"classifier {args.classifier}; parsers: "
        + ", ".join(f"{l}={'spaCy' if l in SPACY_MODELS else 'Stanza'}" for l in sorted(texts_by_lang)))
    feats = extract(texts_by_lang, out_dir, args)
    if args.stage == "features":
        return

    all_effects, all_pred, profiles = [], [], {}
    if "test" in sources:
        d, z = prepare(sources["test"], feats, classes)
        e, p, prof = analyse_test(d, z, classes, args, tag)
        all_effects.append(e)
        all_pred.append(p)
        profiles.update(prof)
    if "llm" in sources:
        d, z = prepare(sources["llm"], feats, classes)
        e, p, prof = analyse_llm(d, z, classes, args)
        all_effects.append(e)
        all_pred.append(p)
        profiles.update(prof)
    agreement(profiles)

    effects = pd.concat(all_effects)
    effects.to_csv(out_dir / f"style_effects_{tag}.csv", index=False)
    pd.concat(all_pred).to_csv(out_dir / f"style_prediction_{tag}.csv", index=False)
    plot(effects, classes, out_dir / f"style_by_class_{tag}.png")
    default = ("results_style_by_class.txt" if tag == "national-k4-logitadj-ni_both" and args.split == "test"
               else f"results_style_by_class_{tag}{'' if args.split == 'test' else '_' + args.split}.txt")
    report = Path(args.report or default)
    report.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()

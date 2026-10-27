#!/usr/bin/env python3
"""Topical phrases the classifier ties to a cluster, per language, with the templates filtered out.

The per-cluster n-gram lists of classifier_ngram_analysis (and similar per-cluster lists) are
full of templates: address ("honourable members"), stance declarations ("i disagree with this
statement"), connectives. Those occur under every statement. A phrase that carries an argument
is tied to a topic. So a phrase is kept only when it

  - has at least --min_words words (default 2),
  - is in at least --min_docs speeches of the language (default 20),
  - has a statement share of at least --min_share (default 25%): of all the speeches in the
    language that contain it, the fraction answering the single statement where it is most
    common,
  - has a lift of at least --min_lift (default 2) for the cluster:
    P(cluster | phrase) / P(cluster), both within the language.

A topical phrase inherits its statement's cluster mix, so plain lift partly says "this
statement's speeches lean to that cluster". The adjusted lift next to it holds statement x
stance fixed: observed cluster count among the phrase's speeches over the count expected from
the cluster shares of their (statement x stance level) cells in the same language, each speech
left out of its own cell. Adjusted lift > 1 is what the phrase adds at a fixed statement and
stance.

Per phrase also: its top statement, the mean stance of its speeches there (cross-encoder,
folded to the base framing), the cluster's official position on that statement, and how many
models wrote it. ~ marks phrases made only of the statement's own words (base or negated
wording), * phrases with a meta/address word.

--labels vaa swaps the classifier label for a reference: the cluster the VAA scoring gives the
speech's cross-encoder stance on its statement (argmax of the vaa_scoring Gaussian posterior).
That label is a function of statement and stance only, so its phrase lists show what stance
alone ties to each cluster. Its adjusted lift is ~1 by construction and the lists are ranked
by plain lift. Outputs get a _vaa suffix.

Usage, from the repository root:
  python analysis/classifier_topical_ngrams.py
  python analysis/classifier_topical_ngrams.py --labels vaa
  python analysis/classifier_topical_ngrams.py --languages en,sk --min_share 0.5
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from analysis.analyze_all import MODEL_DIRS
from analysis.classifier_ngram_analysis import GROUPS, SHORT, TOKEN, load_texts
from analysis.core import SPEECHES_CLASSIFIED_CSV, find_csvs
from analysis.vaa_scoring import cluster_gaussians
from utils import configure_stdout

configure_stdout()

DEFAULT_RESULTS = "data/euandi_2024_results_mt"
DEFAULT_CLASSIFIER = "models/national-k4-langmatched_mt_ni_eval25k_fast/model"
# n-grams that say where the text stands or whom it addresses, not why (lower case, per language;
# other languages fall back to English, so their meta tags are incomplete)
META_NGRAM = {
    "en": r"stance|statement|proposition|i|am|my|disagree|agree|oppose|support|against|favou?r"
          r"|honou?rable|colleagues|president|members|ladies|gentlemen|parliament",
    "fr": r"position|déclaration|affirmation|énoncé|je|j'\w*|suis|mon|ma|moi|accord|d'accord|désaccord|soutiens"
          r"|oppose|contre|faveur|honorables?|collègues|président\w*|membres|mesdames|messieurs|parlement",
    "cz": r"stanovis\w*|postoj\w*|tvrzení\w*|výrok\w*|prohlášení|já|jsem|můj|moje|mé|souhlas\w*|nesouhlas\w*"
          r"|podporuj\w*|odmít\w*|vážen\w*|koleg\w*|předsed\w*|poslan\w*|dámy|pánové|parlament\w*",
    "sk": r"stanovis\w*|postoj\w*|tvrden\w*|výrok\w*|vyhlásen\w*|ja|som|môj|moje|súhlas\w*|nesúhlas\w*"
          r"|podporuj\w*|odmiet\w*|vážen\w*|koleg\w*|predsed\w*|poslan\w*|dámy|páni|parlament\w*",
    "de": r"position|haltung|standpunkt|aussage|these|ich|bin|mein\w*|mir|mich|stimme\w*|zustimm\w*|unterstütz\w*"
          r"|ablehn\w*|dagegen|dafür|gegen|geehrte\w*|kolleg\w*|präsident\w*|mitglieder|damen|herren|parlament\w*",
    "es": r"postura|posición|afirmación|declaración|enunciado|yo|estoy|mi|mis|acuerdo|desacuerdo|apoy\w*|opong\w*"
          r"|contra|favor|honorables?|colegas|president[ae]\w*|miembros|señor\w*|diputad\w*|parlamento\w*",
    "it": r"posizione|affermazione|dichiarazione|io|sono|mio|mia|miei|accordo|disaccordo|sostengo|sostenere"
          r"|oppongo|contro|favore|onorevoli|colleghi|president\w*|membri|signor\w*|parlamento\w*",
    "pl": r"stanowisk\w*|tez\w*|twierdzeni\w*|stwierdzeni\w*|ja|jestem|mój|moje|moim|zgadzam|zgod\w*|popieram"
          r"|sprzeciw\w*|przeciw\w*|szanown\w*|koleg\w*|koleżank\w*|przewodnicząc\w*|posł\w*|panie|panowie"
          r"|parlament\w*",
}
META_NGRAM = {lang: re.compile(rf"\b({words})\b") for lang, words in META_NGRAM.items()}

out_lines = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    out_lines.append(line)


def load(args):
    manifest = json.loads((Path(args.classifier) / "manifest.json").read_text(encoding="utf-8"))
    labels = sorted(manifest["label2id"], key=manifest["label2id"].get)
    biases = np.asarray(manifest.get("biases") or np.zeros(len(labels)))
    temperature = float(manifest.get("temperature", 1.0))
    results_dir = Path(args.results_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip() and (results_dir / m.strip()).is_dir()]
    df = load_texts(results_dir, models, labels, biases, temperature)
    df = df[(df.track == "speeches") & df.stance.notna()].reset_index(drop=True)
    df["pred"] = df.pred_raw if args.raw else df.pred_biased
    lp = df[[f"lp{i}" for i in range(len(labels))]].to_numpy()
    df["confidence"] = np.exp(lp.max(1))
    df["level"] = df.stance_level.astype(int)

    # P(cluster | stance) per text, the vaa_scoring Gaussians, columns in GROUPS order
    names, mu, sd = cluster_gaussians()
    order = [[SHORT[n] for n in names].index(g) for g in GROUPS]
    j = df.statement_idx.to_numpy()
    s = df.stance.to_numpy()[:, None]
    ll = -0.5 * ((s - mu[order][:, j].T) / sd[order][:, j].T) ** 2 - np.log(sd[order][:, j].T)
    post = np.exp(ll - ll.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    for i, g in enumerate(GROUPS):
        df[f"post_{g}"] = post[:, i]
    positions = pd.DataFrame(mu[order].T, columns=GROUPS)

    files = find_csvs(results_dir / models[0], SPEECHES_CLASSIFIED_CSV)
    base = pd.read_csv(files["base"], sep=";", encoding="utf-8-sig")
    neg = pd.read_csv(files["negated"], sep=";", encoding="utf-8-sig")
    statements = base.original_text_en.str.strip().tolist()
    analyzer = CountVectorizer(token_pattern=TOKEN).build_analyzer()
    wording = {lang: [set(analyzer(str(b))) | set(analyzer(str(n)))
                      for b, n in zip(base[f"original_text_{lang}"], neg[f"original_text_{lang}_negated"])]
               for lang in args.languages if f"original_text_{lang}" in base}
    say(f"results: {results_dir}   classifier: {args.classifier}   labels: {'raw' if args.raw else 'deployed'}")
    say(f"{len(df)} speeches, {df.model.nunique()} models, {df.language.nunique()} languages; "
        f"shares {df.pred.value_counts(normalize=True).reindex(GROUPS).round(3).to_dict()}")
    return df, positions, statements, wording


def distinct(ranked, vocab, top):
    """The first `top` n-grams of `ranked` that neither contain nor sit inside one already kept."""
    kept = []
    for i in ranked:
        padded = f" {vocab[i]} "
        if not any(padded in f" {vocab[k]} " or f" {vocab[k]} " in padded for k in kept):
            kept.append(i)
        if len(kept) == top:
            break
    return kept


def phrases(g, positions, statement_words, lang, args):
    """One row per (phrase, cluster) passing the four filters, for one language."""
    g = g.reset_index(drop=True)
    vec = CountVectorizer(ngram_range=(args.min_words, args.max_words), min_df=args.min_docs, binary=True,
                          token_pattern=TOKEN)
    X = vec.fit_transform(g.text).tocsr()
    vocab = vec.get_feature_names_out()
    df_ = np.asarray(X.sum(0)).ravel()

    # statement share: the phrase's speeches on its most common statement / all its speeches
    S = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g.statement_idx)), shape=(len(g), len(positions)))
    by_statement = (X.T @ S).toarray()
    top = by_statement.argmax(1)
    share = by_statement.max(1) / df_

    # lift within the language, and adjusted for statement x stance (leave-one-out cell shares)
    Y = np.stack([(g.pred == grp).to_numpy(float) for grp in GROUPS], axis=1)
    in_cluster = np.asarray(X.T @ Y)
    base = Y.mean(0)
    lift = in_cluster / df_[:, None] / base
    cell = g.groupby(["statement_idx", "level"])
    n_cell = cell.pred.transform("size").to_numpy(float)[:, None]
    cell_counts = np.stack([(g.pred == grp).groupby([g.statement_idx, g.level]).transform("sum").to_numpy(float)
                            for grp in GROUPS], axis=1)
    loo = np.where(n_cell > 1, (cell_counts - Y) / np.maximum(n_cell - 1, 1), base)
    expected = np.asarray(X.T @ loo)
    with np.errstate(divide="ignore", invalid="ignore"):
        adjusted = np.where(expected > 0, in_cluster / expected, np.nan)

    models = pd.factorize(g.model)[0]
    M = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), models)), shape=(len(g), models.max() + 1))
    n_models = np.asarray(((X.T @ M) > 0).sum(1)).ravel()
    stance_sum = (X.T @ sp.csr_matrix(S.multiply(g.stance.to_numpy()[:, None]))).toarray()
    meta = META_NGRAM.get(lang, META_NGRAM["en"])

    rows = []
    for j in np.flatnonzero(share >= args.min_share):
        for i, grp in enumerate(GROUPS):
            if lift[j, i] < args.min_lift:
                continue
            words = set(vocab[j].split())
            rows.append({
                "language": lang, "cluster": grp, "ngram": vocab[j], "docs": int(df_[j]),
                "in_cluster": int(in_cluster[j, i]), "lift": lift[j, i], "adjusted_lift": adjusted[j, i],
                "statement_idx": int(top[j]), "statement_share": share[j],
                "stance": stance_sum[j, top[j]] / by_statement[j, top[j]],
                "position": positions.at[top[j], grp], "models": int(n_models[j]),
                "kind": "meta" if meta.search(vocab[j]) else "echo" if words <= statement_words[top[j]] else "content"})
    say(f"  {lang}: {len(g)} speeches, {len(vocab)} phrases of {args.min_words}-{args.max_words} words in >= "
        f"{args.min_docs} speeches; statement share >= {args.min_share:.0%}: {(share >= args.min_share).sum()}; "
        f"also lift >= {args.min_lift:g} for some cluster: {len({r['ngram'] for r in rows})}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS)
    parser.add_argument("--models", default=",".join(MODEL_DIRS))
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER, help="model dir with the manifest.json")
    parser.add_argument("--raw", action="store_true", help="argmax of the logits without the manifest biases")
    parser.add_argument("--labels", choices=["classifier", "vaa"], default="classifier",
                        help="vaa: cluster from the cross-encoder stance through the VAA scoring, as a reference")
    parser.add_argument("--languages", type=lambda s: [x.strip() for x in s.split(",")],
                        default=["en", "de", "fr", "es", "it", "pl", "cz", "sk"])
    parser.add_argument("--min_words", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=5)
    parser.add_argument("--min_docs", type=int, default=20, help="speeches of the language with the phrase")
    parser.add_argument("--min_lift", type=float, default=2.0)
    parser.add_argument("--min_share", type=float, default=.25, help="statement share")
    parser.add_argument("--top", type=int, default=25, help="phrases shown per cluster and language")
    parser.add_argument("--report", default="results_classifier_topical_ngrams.txt")
    parser.add_argument("--csv", default="results_classifier_topical_ngrams.csv")
    args = parser.parse_args()
    if args.labels == "vaa":
        for name in ("report", "csv"):
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, getattr(args, name).replace(".", "_vaa."))
    return args


def main():
    args = parse_args()
    df, positions, statements, wording = load(args)
    rank_by = "adjusted_lift"
    if args.labels == "vaa":
        classifier = df.pred
        df["pred"] = np.asarray(GROUPS)[df[[f"post_{g}" for g in GROUPS]].to_numpy().argmax(1)]
        rank_by = "lift"
        say(f"labels: VAA scoring of the cross-encoder stance; shares "
            f"{df.pred.value_counts(normalize=True).reindex(GROUPS).round(3).to_dict()}; "
            f"same as the classifier label on {(df.pred == classifier).mean():.1%} of speeches")
    say(f"filters: >= {args.min_words} words, >= {args.min_docs} speeches in the language, statement share >= "
        f"{args.min_share:.0%}, lift >= {args.min_lift:g}\n")
    rows = []
    for lang in args.languages:
        if lang in wording:
            rows += phrases(df[df.language == lang], positions, wording[lang], lang, args)
    t = pd.DataFrame(rows)

    say(f"\n{'=' * 100}\n[1] How many phrases pass, and what is left at a fixed statement and stance\n{'=' * 100}")
    say("  adjusted lift = observed / expected from the cluster shares of the speeches' statement x stance cells")
    say(f"    {'lang':5s} " + " ".join(f"{g:>10s}" for g in GROUPS) + f" {'all':>6s} {'adj >= 2':>9s} {'adj >= 1.5':>11s} "
        f"{'adj < 1.2':>10s} {'median adj':>11s} {'echo':>6s} {'meta':>6s}")
    for lang, s in list(t.groupby("language", sort=False)) + [("all", t)]:
        say(f"    {lang:5s} " + " ".join(f"{(s.cluster == g).sum():10d}" for g in GROUPS) + f" {len(s):6d} "
            f"{(s.adjusted_lift >= 2).mean():9.0%} {(s.adjusted_lift >= 1.5).mean():11.0%} "
            f"{(s.adjusted_lift < 1.2).mean():10.0%} {s.adjusted_lift.median():11.2f} {(s.kind == 'echo').mean():6.0%} "
            f"{(s.kind == 'meta').mean():6.0%}")
    say("  by cluster, all languages: " + ", ".join(
        f"{g} median lift {s.lift.median():.1f} / adjusted {s.adjusted_lift.median():.2f}"
        for g, s in t.groupby("cluster")))

    say(f"\n{'=' * 100}\n[2] Phrases per language and cluster, by cluster speeches x "
        f"log2({rank_by.replace('_', ' ')})\n{'=' * 100}")
    say("  phrase (cluster speeches/all speeches with it, lift, adjusted lift; [statement] share, mean stance there, "
        "cluster position there; models). ~ statement echo, * meta/address. Sub-phrases of a listed phrase are "
        f"dropped; top {args.top}.")
    for lang, by_lang in t.groupby("language", sort=False):
        say(f"\n  ---- {lang} ----")
        for grp in GROUPS:
            s = by_lang[by_lang.cluster == grp]
            # cluster speeches x log adjusted lift: frequent and over-represented beyond the cell
            s = s.assign(score=s.in_cluster * np.log2(s[rank_by].clip(lower=1e-9)))
            s = s.sort_values(["score", "docs"], ascending=False).reset_index(drop=True)
            say(f"\n  == {grp}: {len(s)} phrases, {s.statement_idx.nunique()} statements")
            vocab = s.ngram.to_numpy()
            for k in distinct(range(len(s)), vocab, args.top):
                r = s.iloc[k]
                tag = {"content": "", "echo": "~", "meta": "*"}[r.kind]
                say(f"    {tag}{r.ngram}  ({r.in_cluster}/{r.docs}, lift {r.lift:.1f}, adj {r.adjusted_lift:.2f}; "
                    f"[{r.statement_idx}] {r.statement_share:.0%}, stance {r.stance:+.2f}, pos {r.position:+.2f}; "
                    f"{r.models}m)")
    say("\n  statements: " + "; ".join(f"[{i}] {s}" for i, s in enumerate(statements)))

    Path(args.report).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")
    if args.csv:
        t.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

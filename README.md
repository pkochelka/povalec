# POVALEC: POlitical VAlues in LLMs in European Context

Measures where LLMs sit politically relative to the European Parliament groups, by four
independent methods, and tests whether the resulting ranking survives changing the prompt,
the language, and the elicitation format.

## Method

Each model answers the 30 EU&I 2024 VAA statements in 21 languages, under two framings
(`base`, `_negated`) and several paraphrases, via two elicitation tracks:

- **direct** — the model sees the 1-5 Likert scale and picks a value, with a written
  justification (`{langs}{variant}.csv`)
- **indirect** — the model writes an opinion piece and is never shown a scale
  (`speeches_{langs}{variant}.csv`); stance is read back out of the prose

Both tracks are scored two ways, giving four measurements per model:

| Method | Track | Scoring |
| --- | --- | --- |
| `vaa-likert` | direct | agreement with each EP group's euandi positions |
| `vaa-speeches` | indirect | same, on stance recovered from prose |
| `clf-reasons` | direct | mmBERT EP-party classifier over the justifications |
| `clf-speeches` | indirect | same classifier over the prose |

The scales are not comparable across methods (VAA agreement occupies a narrow band;
classifier probability is dominated by the class prior), so only the **ordering** of the EP
groups is compared. `analysis/rank_consistency_tables.py` reports Kendall's W, ICC and top-1
concordance across method, language, prompt, framing and model.

## Setup

Python 3.10-3.14.

```bash
pip install -r requirements.txt
```

`.env.local` in the repo root supplies the API config:

```ini
BASE_URL=     # OpenAI-compatible endpoint
AUTH_TOKEN=   # optional bearer token
```

`HF_TOKEN` (read from `~/.env.local`) is only needed to train classifiers. Scripts bootstrap
`sys.path` themselves; run them from the repo root.

## Pipeline

```text
statement_collection/scrape_euandi.py   # 30 statements x 21 languages
answer_generation/generate_all.py       # both tracks -> data/euandi_2024_results/<model>/
analysis/agreement_scoring.py           # prose  -> stance     (*_scored.csv)
analysis/classify_speeches.py           # text   -> EP party   (*_classified.csv)
analysis/evaluate_euandi.py             # stance -> agreement  (vaa*.csv)
analysis/analyze_all.py                 # drives the above, then plots and tables
```

`analyze_all.py` is the driver. **Which steps run is controlled by commenting entries in and
out** of its script lists; `--only <fragment>` restricts a run and `--override` recomputes
existing outputs.

The EP-group position basis (`--positions ep-group|national|group-mean`) is **not** encoded
in output filenames, so changing it needs `--override` or the plots keep reading the old
basis.

### Classifier track

```text
preprocessing/fetch_raw_data.py           # download the three corpora + lid.176.bin
preprocessing/europarl_rdf_query.py       # LinkedEP *.ttl -> multi-europarl.csv
preprocessing/europarl_lid_filter.py      # -> multi-europarl-lang_id.csv   [fastText]
preprocessing/preprocess_data.py          # EuroParl + ParlEE + EU Debates -> parquet
preprocessing/split_preprocessed_data.py  # group-disjoint, class-balanced splits
preprocessing/clean_party_names.py        # strip group names -> cleaned/{split}.parquet
preprocessing/build_collapsed_splits.py   # merge ECR+ID, re-split from scratch
analysis/classifier_training.py                        # logit-adjusted CE, natural priors
analysis/classifier_training_for_balanced_collapsed.py # plain CE, balanced train
analysis/test_classifier.py                            # confusion matrix, per-class F1
```

`preprocessing/run_preprocessing.py` runs those seven in order, skipping any step whose
outputs already exist:

```bash
python preprocessing/run_preprocessing.py --dry-run          # plan + commands, runs nothing
python preprocessing/run_preprocessing.py --languages en de  # rehearse on two dumps
python preprocessing/run_preprocessing.py                    # the real thing
python preprocessing/run_preprocessing.py --from lid-filter  # resume after a failure
```

`--only`, `--skip`, `--to`, `--force` and `--extra 'step:args'` (passthrough to one
script) shape the run.

**Python version.** Everything runs on 3.10-3.14 except `europarl_lid_filter.py`, which
needs a fastText binding: `fasttext-wheel` has wheels through 3.12, `fasttext-predict`
through 3.13, and neither builds on 3.14 without a C++ toolchain. The driver probes for a
binding and refuses to start rather than failing three steps in; point that one step at
another interpreter with `--fasttext-python <path>` and keep the rest on your normal one.

`fetch_raw_data.py` pulls ~9 GB; run it with `--dry-run` first, and `--source` /
`--languages` to fetch a subset. Downloads resume and are checksum-verified.
The RDF query is the expensive step: it loads every Turtle dump into one
`rdflib.Graph`, so use `--languages` when testing.

#### Data sources

| Corpus | Provenance | Licence |
| --- | --- | --- |
| ParlEE EP plenary speeches (2009-2019) | Harvard Dataverse, DOI [10.7910/DVN/VOPK0E](https://doi.org/10.7910/DVN/VOPK0E) | see the Dataverse record |
| EU Debates (2009-2023) | HF [`coastalcph/eu_debates`](https://huggingface.co/datasets/coastalcph/eu_debates), Chalkidis & Brandl (2024) | CC-BY-NC-SA-4.0 |
| LinkedEP / Talk of Europe | DANS, DOI [10.17026/dans-x62-ew3m](https://doi.org/10.17026/dans-x62-ew3m), van Aggelen et al. (2016) | CC0-1.0 |
| fastText `lid.176.bin` | Joulin et al. (2016), Meta AI | CC-BY-SA-3.0 |

`europarl_rdf_query.py` and `europarl_lid_filter.py` are adapted from
**Paul Lerner's** [21-EuroParl](https://github.com/PaulLerner/21-EuroParl)
(Lerner and Yvon, 2025) — he is the original author of both steps. His code is
MIT-licensed; the notice is reproduced in `preprocessing/LICENSE.21-EuroParl`
and each file's docstring records what we changed.

**We stop where his pipeline continues.** Upstream, the multiparallel CSV goes on
to bertalign sentence alignment and `merge_align.ipynb`. We skip both and keep
every speech, at speech granularity, in `multi-europarl-lang_id.csv`: the party
classifier wants volume and natural per-language coverage, not alignment.

### Stance track

```text
analysis/llm_stance_judge.py, judge_all_speeches.py  # LLM judge -> gold stance labels
analysis/stance_crossencoder_training.py             # (statement, text) -> stance in [-1,1]
analysis/validate_stance_judge.py                    # human vs judge vs NLI agreement
```

## Layout

```text
answer_generation/   generation.py engine + one thin adapter per track
analysis/            scoring, training, evaluation, LaTeX tables
analysis/plotting/   figures
preprocessing/       EuroParl corpus -> classifier splits
utils/               constants, Likert conversions, API client, sampling
prompts/             prompt templates, JSON, per language
data/                inputs and per-model results (gitignored)
```

`utils/constants.py` is the single source for the language list, prompt framings, EP-group
order and palette, questionnaire axes, and results-filename templates. Do not re-spell these
per script.

## Conventions

- Results CSVs are `;`-separated, `utf-8-sig`.
- Metadata lives in filenames: `[speeches_]{langs}{variant}[_scored|_classified].csv`.
- Columns are `{kind}_{lang}{variant}_v{paraphrase}`, e.g. `choice_de_negated_v3`.
- Stance is `[-1, +1]`, `+1` = agrees with the statement; Likert 1-5 converts via
  `utils.likert_to_stance`.
- A refusal is a `reason` starting `REFUSED`; an unparseable call is `FAILED`. Both are
  excluded from scoring.

## Notes

- EU&I serves identical statements to locales sharing a language, so `at`/`be` -> `de`,
  `cy` -> `gr`, `lu` -> `fr` are dropped; 21 languages remain.
- The collapsed (ECR+ID) track re-splits from scratch, so its dev/test are **not** subsets
  of the 7-party splits. Never evaluate a model across tracks.

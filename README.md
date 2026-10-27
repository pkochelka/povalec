# POVALEC: POlitical VAlues in LLMs in European Context

Measures where LLMs sit politically relative to the European Parliament, by three
independent methods, and tests whether the resulting ranking survives changing the prompt,
the language, and the elicitation format. The reference points are either the EP groups or,
as in the paper, four clusters of national parties (`--positions cluster`): Radical left,
Progressive federalists, Liberal-conservative center-right and Sovereigntist right.

## Method

Each model answers the 30 EU&I 2024 VAA statements in 21 languages, under two framings
(`base`, `_negated`) and several paraphrases, via two elicitation tracks:

- **direct** — the model sees the 1-5 Likert scale and picks a value, with a written
  justification (`{langs}{variant}.csv`)
- **indirect** — the model writes an opinion piece and is never shown a scale
  (`speeches_{langs}{variant}.csv`); stance is read back out of the prose

The tracks are scored in three ways:

| Method | Track | Scoring |
| --- | --- | --- |
| `vaa-likert` | direct | agreement with each group's euandi positions |
| `vaa-speeches` | indirect | same, on stance recovered from prose by the cross-encoder |
| `clf-speeches` | indirect | mmBERT party classifier over the prose |
| `clf-reasons` | direct | same classifier over the justifications — **off by default** |

`clf-reasons` is left out because refusing models' justifications are mostly AI
disclaimers and both-sides boilerplate, so the classifier reads their style, not their
stance. `INCLUDE_REASONS_CLASSIFIER=1` (or `analyze_all.py --with-reasons-classifier`)
brings it back in every table and figure (`analysis.core.CLASSIFIED_TRACKS`).

How a VAA stance is scored against the groups is set in `analysis/vaa_scoring.py`:

- `posterior` (the default on the cluster basis, and only defined there) gives the
  probability of each cluster, with each cluster a Gaussian over its parties' positions,
  so the VAA methods give a cluster probability, like the classifier.
- `distance` (the default elsewhere) gives agreement as `1 - |position - stance| / 2`.

`VAA_SCORING` or `analyze_all.py --vaa-scoring` overrides the default.

The scales are not comparable across methods, so only the **ordering** of the groups is
compared. `analysis/rank_consistency_tables.py` reports Kendall's W, ICC and top-1
concordance across method, language, prompt, framing and model.
`analysis/tables/cross_method_consistency_table.py` writes the paper's compact version.

### Null models

Every share and probability in the figures and tables is drawn **minus a null model**: what
a group would get from random stances rather than the model's. The VAA nulls come from
`vaa_null_model_argmax.py` and the classifier nulls from `classifier_null_model.py`, and
`null_baseline.py` reads them back. There are three:

- `gaussian`: 0-centred normal stances, σ = 0.5 by default.
- `uniform`: uniform stances.
- `max`: per group, the largest of the normal null at σ ∈ {0.25, 0.5, 0.75, 1} and the
  uniform null. It is conservative, so read positive differences from it as lower bounds.

Outputs go to `plots_<null>/` and `tables_<null>/` next to the raw `plots/` and `tables/`;
`--absolute` on the plotting and table scripts skips the subtraction.

## Setup

Python 3.10-3.14.

Dependencies are managed with [uv](https://docs.astral.sh/uv/), into a project-local
`.venv` — never into the system interpreter.

```bash
uv venv --python 3.11         # creates ./.venv
uv pip install -e .           # or: uv pip install -e ".[lid]" for the fastText step
```

Then either activate it (`.venv/Scripts/activate` on Windows, `source .venv/bin/activate`
elsewhere) or prefix commands with `uv run`:

```bash
uv run python analysis/analyze_all.py --only evaluate_euandi
```

`uv.lock` pins the exact resolution and is committed; `uv sync` reproduces it exactly,
which is what a rerun of the thesis pipeline should use. `uv lock --upgrade` refreshes it
deliberately.

The editable install is what puts `analysis`, `answer_generation`, `preprocessing`,
`statement_collection` and `utils` on the import path. Without it the scripts cannot import
each other — they no longer patch `sys.path` themselves.

`.env.local` in the repo root supplies the API config:

```ini
BASE_URL=     # OpenAI-compatible endpoint
AUTH_TOKEN=   # optional bearer token
```

`HF_TOKEN` (read from `~/.env.local`) is only needed to train classifiers. Run the scripts
from the repo root — data and prompt paths are resolved relative to it.

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

The position basis (`--positions ep-group|national|group-mean|cluster`) is **not** encoded
in output filenames, so changing it needs `--override` or the plots keep reading the old
basis. Neither is the classifier checkpoint (`--classifier <dir with manifest.json>`), whose
labels must match the basis: a cluster-track checkpoint needs `--positions cluster`. The
paper's run:

```bash
python analysis/analyze_all.py --positions cluster \
    --classifier models/national-k4-langmatched_mt_ni_eval25k_fast/model --override
```

A cluster run writes over the EP-group outputs in `data/euandi_2024_results/`, so back them
up first if you need both.

### Classifier track

```text
preprocessing/fetch_raw_data.py           # download the three corpora + lid.176.bin
preprocessing/europarl_rdf_query.py       # LinkedEP *.ttl -> multi-europarl.csv
preprocessing/europarl_lid_filter.py      # -> multi-europarl-lang_id.csv   [fastText]
preprocessing/find_chair_speeches.py      # mark speeches by the presiding (Vice-)President
preprocessing/preprocess_data.py          # EuroParl + ParlEE + EU Debates -> parquet
preprocessing/split_preprocessed_data.py  # group-disjoint, class-balanced splits
preprocessing/clean_party_names.py        # strip group names -> cleaned/{split}.parquet
preprocessing/build_collapsed_splits.py   # merge ECR+ID, re-split from scratch
analysis/classifier_training.py                        # logit-adjusted CE, natural priors
analysis/classifier_training_for_balanced_collapsed.py # plain CE, balanced train
analysis/test_classifier.py                            # confusion matrix, per-class F1
```

`preprocessing/run_preprocessing.py` runs the preprocessing steps in order, skipping any
step whose outputs already exist:

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

### Cluster track (the paper's classifier)

Speeches are labelled by their national party's k=4 cluster instead of by EP group.
`analysis/party_kmeans.py` clusters the EU&I 2024 parties on the 30 VAA statements, and
`analysis/build_cluster_positions.py` averages each cluster's parties into one MEP-weighted
position vector (the `cluster` basis).

```text
preprocessing/run_preprocessing_clusters.py  # steps 1-7 above, then instead of `collapse`:
  build_national_party_map.py                #   each speech's national party -> EU&I party
  build_national_cluster_splits.py           #   label by k=4 cluster, re-split
                                             #   -> data/EuroParl Custom/clusters_k4_national/
preprocessing/build_mt_augmentation.py       # machine-translated train speeches per cell
preprocessing/build_mt_train.py              # -> <track>/train_mt.parquet
preprocessing/build_language_matched_split.py
                                             # every language gets the pooled cluster mix;
                                             # run once plain (dev/test), then with
                                             # --splits train --mt <track>/train_mt.parquet --out-suffix _mt
analysis/train_cluster_track.py              # trains with the existing trainers -> runs/
```

```bash
python analysis/train_cluster_track.py --track national --trainer langmatched_mt
```

The cluster driver takes the same flags as `run_preprocessing.py`, and reuses the outputs of
the shared steps. Its lid-filter default is `--min-languages 7`, which keeps the 2013-2017
LinkedEP speeches that exist only in their original language.

The language-matched split is there because the language would otherwise carry cluster
information. The MT rows are there because the classifier otherwise reads native and
translated text of one language differently.

`train_cluster_track.py` writes every run to `runs/<track>-k<k>-<trainer><suffix><tags>/`,
so runs never overwrite each other. `--trainer` is `balanced`, `logitadj`, `langmatched` or
`langmatched_mt`. Experimental options, none of them used for the paper:

- `--adversary origin|language|both`
- `--consistency-weight` (`classifier_training_consistency.py`)
- `--init-from`, `--train-per-cell`, `--max-epochs`

`--evaluate-checkpoint` turns an intermediate checkpoint into a usable model with a
manifest.

### Classifier validation and analysis

```text
analysis/manifesto_control.py        # out-of-domain: national party manifestos (MARPOR)
analysis/topic_prevalence.py         # questionnaire-topic prevalence per cluster in train
analysis/stance_bin_control.py       # does the label follow a changed stance?
analysis/classifier_topical_ngrams.py  # per-cluster topical n-grams (the paper's table)
analysis/classifier_ngram_analysis.py  # why models get different cluster mixes
analysis/classifier_baseline.py      # bag-of-n-grams baseline
```

### Stance track

```text
analysis/llm_stance_judge.py, judge_all_speeches.py  # LLM judge -> gold stance labels
analysis/stance_crossencoder_training.py             # (statement, text) -> stance in [-1,1]
analysis/validate_stance_judge.py                    # human vs judge vs NLI agreement
```

## Layout

```text
answer_generation/   generation.py engine + one thin adapter per track
analysis/core/       reads the results tree: filename grammar, loaders, positions,
                     questionnaire axis coding
analysis/stats.py    pure statistics: Kendall's W, ICC, Spearman-Brown, concordance
analysis/tables/     table labels + LaTeX/Markdown rendering
analysis/            scoring, training, evaluation
analysis/plotting/   figures
preprocessing/       EuroParl corpus -> classifier splits
utils/               constants, Likert conversions, API client, sampling
prompts/             prompt templates, JSON, per language
data/                inputs and per-model results (gitignored)
runs/                cluster-track training runs (gitignored)
```

In `data/<dataset>_results/` the model directories sit next to the dataset-level `plots/`,
`tables/` and their per-null twins. Use `analysis.core.is_output_dir()` to tell them apart
rather than a hard-coded name list.

`utils/constants.py` is the single source for the language list, prompt framings, EP-group
order and palette, questionnaire axes, and the EP-group ↔ party mapping. Do not re-spell
these per script.

`analysis/core/` owns everything that *reads* the results tree — the filename and column
grammar in both directions (`core/paths.py`), the stance loaders and the six scoring
methods (`core/results.py`), and the positions basis (`core/positions.py`). The dependency
rule is one-way: **analysis and plotting import core; core imports neither.** Table and
statistics code must never import from `analysis/plotting/`.

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
  of the 7-party splits. Never evaluate a model across tracks. The same holds for the
  cluster tracks.
- The 204 human-annotated speeches are pinned to the test split on every track
  (`preprocessing/pinned_test_speeches.csv`). Regenerate that file if the annotation
  sample is redrawn.

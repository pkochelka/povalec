# Negation quality annotations: 150 doubly-annotated statement pairs

The 30 policy statements of the EU&I 2024 questionnaire were automatically
negated in five languages. Two annotators independently rated each negated
variant for how well it reverses the polarity of the original while otherwise
preserving its meaning. All 150 items carry a label from both annotators, so the
sample supports plain two-rater agreement statistics.

## Files

| File | Contents |
| --- | --- |
| `negation_annotations_150.csv` | The dataset: 150 rows, comma-separated, UTF-8 with BOM. |

## Columns

| Column | Description |
| --- | --- |
| `item_id` | `001`–`150`. Stable identifier, no other meaning. |
| `statement_idx` | `1`–`30`, the statement's position in the EU&I 2024 questionnaire. Each index appears once per language. |
| `language` | `cz`, `de`, `en`, `fr`, `sk` (30 items each). |
| `statement` | The original questionnaire statement. |
| `negated_statement` | The automatically negated variant that was rated. |
| `annotator_a` | First annotator's rating, `1`–`5`. |
| `annotator_b` | Second annotator's rating, `1`–`5`. |

## The scale

`1` = the negation fails (wrong polarity, or meaning otherwise broken),
`5` = a clean negation that reverses polarity and changes nothing else.

In practice the annotators only ever used `4` and `5`: 278 of the 300 labels are
`5`. Nothing was rated `1`, `2` or `3`, so the automatic negation never produced
an outright failure in this sample. **This skew dominates how the agreement
figures below must be read.**

## Annotation procedure

Both annotators labelled all 150 items. They worked independently: neither saw
the other's rating for an item before assigning their own.

## How many negations cleared each bar

Items whose rating reaches a given threshold, counted strictly (both annotators
at or above it) and leniently (at least one):

| Threshold | Both annotators | At least one | `annotator_a` | `annotator_b` |
| --- | --- | --- | --- | --- |
| ≥ 3 | 150 | 150 | 150 | 150 |
| ≥ 4 | 150 | 150 | 150 | 150 |
| ≥ 5 | 132 | 146 | 133 | 145 |

Every negation cleared `≥ 4` on both annotators' ratings, so the ≥ 3 and ≥ 4
rows are saturated by construction — the only threshold that discriminates is
`≥ 5`. Read strictly, 132 of 150 negations (88%) are flawless; read leniently,
146 (97%) struck at least one annotator as flawless.

At `≥ 5`, by language:

| Language | Both | At least one | n |
| --- | --- | --- | --- |
| `cz` | 24 | 30 | 30 |
| `de` | 29 | 29 | 30 |
| `en` | 27 | 30 | 30 |
| `fr` | 27 | 29 | 30 |
| `sk` | 25 | 28 | 30 |

Czech draws the most reservations under the strict reading (24 of 30) but every
Czech negation satisfied at least one annotator.

## Agreement

Computed by `analysis/negation_annotation_iaa.py` over all 150 items:

| Metric | All 150 | Excluding `de` (120) |
| --- | --- | --- |
| Exact agreement | **0.907** | 0.883 |
| Gwet's AC1 | **0.892** | 0.862 |
| Krippendorff's α (ordinal) | 0.316 | 0.240 |
| Cohen's κ | 0.329 | 0.261 |
| Within ±1 | 1.000 | 1.000 |

Quote **percent agreement and AC1**, not κ or α, for this dataset. With 93% of
all labels sitting in one category, κ and α estimate chance agreement as nearly
the whole of the observed agreement and collapse towards zero — the well-known
kappa prevalence paradox. The `cz` slice makes the failure mode obvious: the
annotators match on 80% of items yet κ = −0.06. Gwet's AC1 is designed for
exactly this case and is the figure to report; the low κ and α here are an
artefact of the skew, not evidence of unreliable annotation.

Every one of the 14 disagreements is a `4`-vs-`5` split — one step wide, with
the two annotators never differing about whether a negation was usable. The
disagreements concentrate on statements 18, 20 and 23, which account for 9 of
the 14 and are worth re-reading if a gold label is needed.

The German items were rated identically by both annotators on all 30 rows, which
pins every agreement statistic for that slice at its ceiling. That may simply
reflect an easy subset, but it cannot be distinguished from a single shared
rating on the basis of this file alone, so the table reports the other 120 items
separately.

## Reproducing

```bash
python analysis/negation_annotation_iaa.py
```

Needs nothing but this CSV, `pandas`, `numpy`, `scipy` and `scikit-learn`.

## Anonymisation

The annotators are identified only as A and B. The statements come from the
public EU&I 2024 questionnaire and the negated variants are machine-generated;
neither contains personal data.

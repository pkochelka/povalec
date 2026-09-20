# Stance annotations: 160 doubly-annotated (statement, answer) pairs

Two annotators independently rated the stance that a generated political speech
takes towards a policy statement, on a 5-point Likert scale. Every one of the
160 items carries a label from both annotators, so the sample supports plain
two-rater agreement statistics.

## Files

| File | Contents |
| --- | --- |
| `stance_annotations_160.csv` | The dataset: 160 rows, comma-separated, UTF-8 with BOM. |

## Columns

| Column | Description |
| --- | --- |
| `item_id` | `001`–`160`. Stable identifier, no other meaning. |
| `language` | Language of the statement and the answer: `cz`, `en`, `fr`, `sk` (40 items each). |
| `answer_model` | The LLM that produced the answer: `deepseek-v4-pro` (52), `glm-5.2` (57), `kimi-k2.7` (51). |
| `statement` | The policy statement, taken from the EU&I 2024 questionnaire. |
| `answer` | The generated speech reacting to that statement. Whitespace is normalised to single spaces. |
| `annotator_a` | First annotator's label, `1`–`5`. |
| `annotator_b` | Second annotator's label, `1`–`5`. |

## The scale

`1` = totally agree with the statement, `3` = neutral, `5` = totally disagree.
Note the direction: **low = agree**. For the three-class collapse used in the
analysis, `1`–`2` is *agree*, `3` is *neutral*, `4`–`5` is *disagree*.

## Annotation procedure

Both annotators labelled all 160 items. They worked independently: neither saw
the other's label for an item before assigning their own.

The 160 items were selected to be balanced across language and `answer_model`,
not sampled at random from the speech pool, so agreement figures here do not
transfer directly to that pool.

## Agreement

Computed by `analysis/stance_annotation_iaa.py` over all 160 items:

| Metric | Value |
| --- | --- |
| Krippendorff's α (ordinal) | **0.711** (95% bootstrap CI 0.59–0.81) |
| Cohen's κ, quadratic weights | 0.735 |
| Cohen's κ, linear weights | 0.666 |
| Cohen's κ, unweighted | 0.501 |
| Exact agreement | 0.638 |
| Within ±1 | 0.888 |
| Three-class direction κ | 0.767 |

The ordinal α is the headline figure: the scale is ordinal and most
disagreements are one step wide, so unweighted κ — which penalises a 1-vs-2 miss
as heavily as a 1-vs-5 miss — understates agreement here.

Agreement is weakest on `en` (α 0.57, with 20% of item pairs a direct
agree-vs-disagree flip) and on `kimi-k2.7` answers (α 0.62), and strongest on
`sk` (0.79) and `deepseek-v4-pro` answers (0.79).

## Reproducing

```bash
python analysis/stance_annotation_iaa.py
```

The agreement statistics need nothing but this CSV, `pandas`, `numpy`, `scipy`
and `scikit-learn`. To additionally score a stance cross-encoder checkpoint
against both annotators (this part needs the model and the rest of the repo):

```bash
python analysis/stance_annotation_iaa.py --crossencoder --out preds.csv
```

## Anonymisation

The annotators are identified only as A and B. The statements come from the
public EU&I 2024 questionnaire, and the answers are model-generated; neither
contains personal data.

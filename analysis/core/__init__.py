"""Data-access layer for the results tree.

Everything that reads `data/<dataset>_results/<model>/` lives here: the filename and
column grammar (`paths`), the loaders and scorers built on it (`results`), and the
display names figures and tables label models with (`labels`).

The rule this package exists to enforce: **analysis and plotting depend on core;
core depends on neither.** Before it, `rank_consistency_tables.py` — the main
statistics and table builder — imported eighteen loaders from
`plotting/plot_ep_group_rank_boxplots.py`, so building a LaTeX table pulled in
matplotlib and a figure module's state, and a plot script was the authority on how a
results CSV is read.
"""
from .labels import MODEL_DISPLAY_NAME, count, model_display_name, party_sort_key
from .paths import (
    CHOICE_COLUMN,
    FRAMING_FOR_VARIANT,
    FRAMING_ORDER,
    FRAMING_ORIENTATION,
    JUDGE_COLUMNS,
    JUDGE_CSV,
    LANGS,
    PARTY_PROB_COLUMN,
    PREDICTED_PARTY_COLUMN,
    REASONS_CLASSIFIED_CSV,
    RESPONSES_CSV,
    SLUG_TO_LABEL,
    SPEECHES_CLASSIFIED_CSV,
    SPEECHES_SCORED_CSV,
    STANCE_COLUMN,
    VARIANT,
    classified_name,
    responses_name,
    scored_name,
    speeches_name,
    vaa_name,
)
from .positions import (
    DEFAULT_POSITIONS,
    GROUP_POSITIONS_PATH,
    PARTY_POSITIONS_PATH,
    POSITION_CHOICES,
    load_party_positions,
    positions_path,
)
from .questionnaire import (
    axis_directions_in_admin_order,
    load_axis_directions,
)
from .results import (
    CELL_KEYS,
    METHODS,
    NEUTRAL_LIKERT,
    classifier_scores,
    find_csvs,
    judge_stances,
    likert_stances,
    load_scores,
    positions_frame,
    slug_labels,
    speech_stances,
    vaa_scores,
)

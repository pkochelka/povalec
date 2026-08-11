"""Table assembly for the rank-consistency report: what things are called, and how
computed metrics become LaTeX or Markdown.
"""
from .labels import (
    BYMODEL_FACTORS,
    DEFAULT_METHODS,
    DEFAULT_TABLES,
    DIMENSION_HEADER,
    FACTOR_NOUN,
    MATRIX_LABEL,
    METHOD_LABEL,
    MODEL_SHORT,
    POOLED,
    POOLED_ROW_LABEL,
    VARIANT_CAPTION,
    VARIANT_FACTORS,
    check_unique_short_names,
    short_model_name,
)
from .render import (
    MATRIX_FRAMINGS,
    bymodel_cell,
    bymodel_factor_spec,
    bymodel_row_label,
    matrix_title,
    pairs_frame,
    reliability_r_xx,
    reliability_spec,
    render_bymodel_latex,
    render_bymodel_markdown,
    render_latex,
    render_markdown,
    render_matrix_latex,
    render_matrix_markdown,
    render_reliability_latex,
    render_reliability_markdown,
    table_body,
    tidy_frame,
)

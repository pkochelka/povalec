"""Row, column and method labels for the rank-consistency tables.

Presentation only: what a method, model or pooled row is *called* in a table. Split out
of `rank_consistency_tables.py` alongside the renderer that consumes them, so the module
that computes the statistics no longer also owns their spelling.
"""

METHOD_LABEL = {
    "vaa-likert": "Direct VAA (Likert choice)",
    "vaa-reasons": "Indirect VAA (NLI on reasons)",
    "vaa-speeches": "Indirect VAA (NLI on prose)",
    "clf-reasons": "Classifier on reasons",
    "clf-speeches": "Classifier on prose",
    "vaa-likert-judge": "VAA (LLM judge on reasons)",
    "vaa-speeches-judge": "VAA (LLM judge on prose)",
}
# Short, self-explanatory row/column headers for the pairwise-correlation matrix --
# readable on their own, unlike an "L / R / cR / cO" legend the reader has to look
# up. Kept short enough that a 4x4 grid of them still fits a page.
MATRIX_LABEL = {
    "vaa-likert": "Direct",
    "vaa-reasons": "Indirect (reasons)",
    "vaa-speeches": "Indirect",
    "clf-reasons": "Reasons",
    "clf-speeches": "Prose",
    "vaa-likert-judge": "Indirect (reasons, judge)",
    "vaa-speeches-judge": "Indirect (judge)",
}
assert set(MATRIX_LABEL) == set(METHOD_LABEL)
# "Indirect" means NLI-scored prose (the open-ended answers), not the NLI-scored reasons --
# the reasons track is only measured here via the classifier.
DEFAULT_METHODS = ["vaa-likert", "vaa-speeches", "clf-reasons", "clf-speeches"]
DEFAULT_TABLES = ["internal", "negation", "crosslang", "variants", "bymodel"]

# Short row labels for the per-model table; anything not listed falls back to its
# first hyphen-separated token, capitalised.
# Trimmed to the shortest form that still separates every model: only the two
# Gemma, two Qwen and two GPT entries need a distinguishing suffix, and only
# DeepSeek is long enough to be worth contracting. Anything shorter collides on the
# G-prefix (GLM / GPT / Granite / Grok / Gemini / Gemma) -- check_unique_short_names
# fails loudly if a future entry does.
MODEL_SHORT = {
    "deepseek-v4-pro": "DS", "gemini3.5-flash": "Gemini",
    "gemma-4-12b": "Gemma12", "gemma-4-31b": "Gemma31", "glm-5.2": "GLM",
    # Two GPT entries, so both carry which one they are: on the heuristic alone
    # gpt-5.6-luna came out "Gpt" beside gpt-oss-120b's "GPT", a difference of one
    # capital that no reader can be expected to see.
    "gpt-5.6-luna": "GPT-Luna", "gpt-oss-120b": "GPT-OSS",
    "granite-4.1-8b": "Granite", "granite-4.1-8b-instruct": "Granite",
    "grok-4.5": "Grok", "kimi-k2.7": "Kimi2.7", "kimi-k3": "Kimi3",
    "mistral-medium-3.5": "Mistral",
    "muse-spark-1.1": "Muse", "qwen3.5-122b": "Qwen122", "qwen3.6-27b": "Qwen27",
}
# The pooled row is otherwise the widest entry in the column, so it sets the
# width no matter how short the model names get.
POOLED_ROW_LABEL = "All"


def short_model_name(model):
    return MODEL_SHORT.get(model, model.split("-")[0].capitalize())


def check_unique_short_names(models):
    """The fallback heuristic in short_model_name can collide (e.g. a new
    gemma-4-Nb model dir falls back to the same "Gemma" an existing dict entry
    already claims) -- caught here instead of silently mislabelling two different
    models the same in the bymodel table.

    Compared case-folded, because the fallback capitalises and the dict does not:
    gpt-5.6-luna once printed as "Gpt" one row above gpt-oss-120b's "GPT", which is
    a distinct label only to a reader who knows to look for it."""
    labels = {}
    for model in models:
        labels.setdefault(short_model_name(model).casefold(), []).append(model)
    collisions = {label: models for label, models in labels.items() if len(models) > 1}
    if collisions:
        details = "; ".join(f"{label!r} <- {models}" for label, models in collisions.items())
        raise SystemExit(f"Short model names collide, add distinct MODEL_SHORT entries: {details}")


POOLED = "All"
DIMENSION_HEADER = {"model": "Model", "method": "Method"}

VARIANT_FACTORS = ["prompt", "paraphrase", "framing"]
VARIANT_CAPTION = {"prompt": "framing $\\times$ paraphrase",
                   "paraphrase": "paraphrase wording, framings pooled",
                   "framing": "base against negated, paraphrases pooled"}
FACTOR_NOUN = {"method": "method", "language": "language", "prompt": "prompt variant",
               "paraphrase": "paraphrase", "framing": "framing", "topic": "topic"}
# The "annotator sets" of the per-model table. Names follow the user's framing,
# not the script's internal factor keys: script "paraphrase" (the v0..v7 wording
# index) is the user's "prompt variants", and script "framing" (base vs negated)
# is the user's "paraphrase". "topic" is not a run-level factor like the others
# -- see TOPIC_AXES.
BYMODEL_FACTORS = [
    ("language", "Lang"),
    ("paraphrase", "Prompt"),
    ("framing", "Negation"),
    ("topic", "Topic"),
]

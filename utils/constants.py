"""Constants shared across generation, scoring, analysis and plotting.

Anything defined in more than one script belongs here: several of these had already
drifted apart between copies (the framing list, the axis order, the source-model list),
which silently changes what a figure or table covers.
"""

# EU&I serves identical statements for country locales that share a language, so
# we keep one code per language: at, be -> de; cy -> gr; lu -> fr are dropped.
LANGS = [
    "bg", "cz", "dk", "ee", "fi", "fr",
    "de", "gr", "hu", "it", "lv", "lt", "nl",
    "pl", "pt", "ro", "sk", "si", "es", "se",
]

ALL_LANGS = LANGS + ["en"]
ALL_LANGS_STR = ",".join(ALL_LANGS)

# Prompt framings, as the filename/column suffix each one carries.
#
# A third framing, "_question", was generated for an early experiment and is not read
# by any analysis: it was absent from most VARIANTS lists, present in a few, and the
# figures that did pick it up put a third of every stack behind a variant nothing else
# reports on. It has been removed everywhere rather than left as a partly-wired option.
VARIANTS = ["", "_negated"]
VARIANT_LABELS = {"": "base", "_negated": "negated"}
# Regex fragment for the same suffix, for readers that parse it back out of filenames
# and column names. Keep the alternation's empty branch first so "" still matches.
VARIANT_PATTERN = r"(?P<variant>|_negated)"

# Markers the generators write into the reason column instead of an answer.
# "REFUSED..." is a model declining; "FAILED" is a request that never produced
# parseable output. Both are excluded from scoring and classification.
REFUSED_REASON_PREFIXES = ("REFUSED",)
FAILED_REASON_VALUES = {"FAILED"}

# The two elicitation tracks. "reasons" is the direct track (the model saw the 1-5
# scale and justified its choice); "speeches" is the indirect one (open-ended prose,
# no scale shown). The filename templates take {langs} and {variant}.
SOURCE_TEXT_COLUMN_PREFIX = {"speeches": "answer", "reasons": "reason"}
SOURCE_INPUT_FILENAME = {
    "speeches": "speeches_{langs}{variant}.csv",
    "reasons": "{langs}{variant}.csv",
}
SOURCE_OUTPUT_FILENAME = {
    "speeches": "speeches_{langs}{variant}_classified.csv",
    "reasons": "{langs}{variant}_classified.csv",
}

# Questionnaire axes, in the order tables and figures report them. The set is fixed by
# euandi_2024_questionnaire.jsonl; only the order is a presentation choice.
AXES = ["Ukraine", "Ecology", "Immigration", "Values", "Economy", "Europe", "Left-Right"]

# EP groups, left-to-right as every figure orders them, with the palette they share.
# One group keeps one colour across all figures -- do not re-spell these per script.
PARTY_DISPLAY_ORDER = ["GUE/NGL", "S&D", "Greens/EFA", "ALDE", "PPE", "ECR", "ID", "ECR+ID"]
PARTY_COLORS = {
    # GUE/NGL's own colour is a dark red a shade away from S&D's -- the two were
    # indistinguishable side by side. Pushed towards magenta: still a red of the left,
    # but separated (protanopic dE 4.0 -> 17.0).
    "GUE/NGL":    "#8E1B6B",
    "S&D":        "#E2061D",
    "Greens/EFA": "#5DA13F",
    "ALDE":       "#FAD22D",
    "PPE":        "#3399FF",
    "ECR":        "#0054A5",
    "ID":         "#2B3856",
    "ECR+ID":     "#164B75",
}
FALLBACK_PARTY_COLOR = "#888888"

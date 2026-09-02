"""Display names shared by figures and tables.

`model_display_name` used to live in `plotting/plot_classified_parties.py`, which meant
`rank_consistency_tables.py` and `refusal_analysis.py` imported a 1560-line plotting
module — pulling in matplotlib and its module-level state — to spell a model's name.
"""
from utils import PARTY_DISPLAY_ORDER

MODEL_DISPLAY_NAME = {
    "deepseek-v4-pro": "DeepSeek V4 Pro",
    "gemini3.5-flash": "Gemini 3.5 Flash",
    "gemma-4-12b": "Gemma 4 12B",
    "gemma-4-31b": "Gemma 4 31B",
    "glm-5.2": "GLM-5.2",
    "gpt-5.6-luna": "GPT 5.6-Luna",
    "gpt-oss-120b": "GPT OSS 120B",
    "granite-4.1-8b": "Granite 4.1 8B",
    "grok-4.5": "Grok 4.5",
    "kimi-k2.7": "Kimi K2.7 Code",
    "kimi-k3": "Kimi K3",
    "mistral-medium-3.5": "Mistral Medium 3.5",
    "qwen3.5-122b": "Qwen3.5 122B",
}


def model_display_name(name):
    """Official name for a result directory, for figures. Takes a name or a Path.

    A directory with no entry above falls back to its own name -- the same key the lookup
    used, not `str(name)`, which for a Path is the whole path. Every model that is not in
    the table (a fine-tuned checkpoint, say) is titled by one or the other, so the
    difference is a panel titled `gemma-cz-pcs-10000` rather than
    `data\\euandi_2024_results\\gemma-cz-pcs-10000`."""
    key = getattr(name, "name", name)
    return MODEL_DISPLAY_NAME.get(key, str(key))


def party_sort_key(party):
    """Sort EP groups left-to-right as every figure orders them, unknowns last."""
    return (PARTY_DISPLAY_ORDER.index(party) if party in PARTY_DISPLAY_ORDER
            else len(PARTY_DISPLAY_ORDER), party)


def count(n, noun):
    """`3 models` / `1 model`, for figure subtitles and load progress lines."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

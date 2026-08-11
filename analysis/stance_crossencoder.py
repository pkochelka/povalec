"""Loading and running the fine-tuned stance cross-encoder.

The cross-encoder scores a (statement, text) pair and returns a stance in [-1, +1], where
+1 means the text agrees with the statement. It is trained by
`stance_crossencoder_training.py` and is one of the two stance sources the project still
uses; the other is the LLM judge (`llm_stance_judge.py`).

Single home for three things that were written twice — once in `agreement_scoring.py`
(the pipeline step that writes `stance_{lang}_v{N}` into `*_scored.csv`) and once in the
now-deleted `compare_stance_scoring.py`:

- resolving the checkpoint's `max_len`,
- the tokenize / forward / clip step, which is where the numbers come from,
- the default checkpoint directory.

The batching *wrappers* stay with their callers: `agreement_scoring` walks 4-tuples with
a progress bar, the evaluation scripts walk parallel lists. Only the scoring itself is
shared, so no caller's loop had to change shape.
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "mmbert-small-stance-crossencoder")
DEFAULT_MAX_TOKEN_LENGTH = 512


def load_crossencoder(model_dir, device):
    """(tokenizer, model, max_length) for a stance checkpoint, ready to score.

    `max_length` comes from the checkpoint's own `stance_config.json` so inference
    truncates exactly where training did. A checkpoint without that file falls back to
    DEFAULT_MAX_TOKEN_LENGTH rather than failing -- this is `agreement_scoring.py`'s
    long-standing behaviour, kept because it is the live pipeline's.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    config_path = os.path.join(model_dir, "stance_config.json")
    max_length = DEFAULT_MAX_TOKEN_LENGTH
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            max_length = json.load(f).get("max_len", DEFAULT_MAX_TOKEN_LENGTH)
    return tokenizer, model, max_length


@torch.no_grad()
def score_pairs(statements, texts, tokenizer, model, device, max_length):
    """Stance in [-1, 1] for one batch of (statement, text) pairs.

    `truncation="only_second"` keeps the statement whole and trims the answer, which is
    the pairing the model was trained on. The head is unbounded, so its output is clipped
    to the label range rather than squashed.
    """
    inputs = tokenizer(
        statements, texts,
        return_tensors="pt", truncation="only_second", padding=True,
        max_length=max_length,
    ).to(device)
    logits = model(**inputs).logits.view(-1).float().cpu().numpy()
    return np.clip(logits, -1.0, 1.0)


def load_regressor(device):
    """The default stance checkpoint, for the evaluation scripts."""
    return load_crossencoder(MODEL_DIR, device)


def predict_stances(statements, texts, tokenizer, model, device, max_len, batch_size):
    """Stance for arbitrarily many pairs, given as two parallel lists."""
    scores = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = slice(start, start + batch_size)
        scores[chunk] = score_pairs(
            statements[chunk], texts[chunk], tokenizer, model, device, max_len,
        )
    return scores

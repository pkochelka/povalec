"""
NLI-based stance scoring for generated speeches.

For each (statement, speech) pair, compute a continuous stance score in [-1, 1]:
    score = P(entailment) - P(contradiction)
where the NLI model judges whether the speech entails agreement with the statement.

Score interpretation (maps to Likert 1-5):
    +1.0 = strong agreement (Likert 5)
     0.0 = neutral/mixed    (Likert 3)
    -1.0 = strong disagreement (Likert 1)
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", type=str)
parser.add_argument("--task_prompts", default="./prompts/nli_hypotheses.json", type=str)
parser.add_argument("--input", default="./data/euandi_2019_results/qwen3.5-122b/speeches_en,de,el,es,fr,it_negated.csv")
parser.add_argument("--variant", default="_negated", choices=["", "_question", "_negated"])
parser.add_argument("--languages", default="en,de,el,es,fr,it")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def load_prompts(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # Label order for this model: [entailment, neutral, contradiction]
    return tokenizer, model


@torch.no_grad()
def score_batch(premises, hypotheses, tokenizer, model, device, max_length=512):
    """Return array of (entailment - contradiction) scores."""
    inputs = tokenizer(
        premises, hypotheses,
        return_tensors="pt", truncation=True, padding=True,
        max_length=max_length,
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # Columns: 0=entailment, 1=neutral, 2=contradiction
    return probs[:, 0] - probs[:, 2]


def _build_pairs(df, lang_variant, template):
    pairs = []
    stmt_col = f"original_text_{lang_variant}"
    for i, row in df.iterrows():
        hypothesis = template.format(row[stmt_col])
        for j in range(5):
            speech = row.get(f"answer_{lang_variant}_v{j}")
            if isinstance(speech, str) and speech.strip():
                pairs.append((i, j, speech, hypothesis))
    return pairs


def _run_scores(pairs, tokenizer, model, device, batch_size, lang_variant):
    scores = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), batch_size), desc=f"scoring {lang_variant}"):
        chunk = pairs[start:start + batch_size]
        scores[start:start + len(chunk)] = score_batch(
            [p[2] for p in chunk], [p[3] for p in chunk], tokenizer, model, device
        )
    return scores


def _write_scores(df, pairs, scores, lang_variant):
    for k, (i, j, _, _) in enumerate(pairs):
        df.at[i, f"stance_{lang_variant}_v{j}"] = scores[k]
    variant_cols = [f"stance_{lang_variant}_v{j}" for j in range(5)]
    df[f"stance_{lang_variant}_mean"] = df[variant_cols].mean(axis=1)


def score_dataframe(df, languages, variant, hypothesis_templates, tokenizer, model, device, batch_size=16):
    """
    Expects the wide CSV produced by your generation script, with columns:
        original_text_{lang}{variant}
        answer_{lang}{variant}_v{j}   for j in 0..4
    Adds columns: stance_{lang}{variant}_v{j}  and  stance_{lang}{variant}_mean
    """
    for language in languages:
        lang_variant = f"{language}{variant}"
        pairs = _build_pairs(df, lang_variant, hypothesis_templates[language])
        if not pairs:
            print(f"[{lang_variant}] no speeches found, skipping.")
            continue
        scores = _run_scores(pairs, tokenizer, model, device, batch_size, lang_variant)
        _write_scores(df, pairs, scores, lang_variant)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    hypothesis_templates = load_prompts(args.task_prompts)
    missing = [lang for lang in languages if lang not in hypothesis_templates]
    if missing:
        raise ValueError(f"Missing hypothesis templates for languages: {missing}")

    df = pd.read_csv(args.input, sep=";", encoding="utf-8-sig")
    tokenizer, model = load_model(args.model, args.device)
    df = score_dataframe(
        df, languages, args.variant, hypothesis_templates, tokenizer, model, args.device,
        batch_size=args.batch_size,
    )

    stem, ext = os.path.splitext(args.input)
    output_path = f"{stem}_scored{ext}"
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path}")

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
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

HYPOTHESIS_TEMPLATES = {
    "en": "The author agrees with the following statement: {}",
    "de": "Der Autor stimmt der folgenden Aussage zu: {}",
    "el": "Ο συγγραφέας συμφωνεί με την ακόλουθη δήλωση: {}",
    "es": "El autor está de acuerdo con la siguiente afirmación: {}",
    "fr": "L'auteur est d'accord avec l'affirmation suivante : {}",
    "it": "L'autore è d'accordo con la seguente affermazione: {}",
}

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def load_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
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


def score_dataframe(df, languages, variant, tokenizer, model, device, batch_size=16):
    """
    Expects the wide CSV produced by your generation script, with columns:
        original_text_{lang}{variant}
        answer_{lang}{variant}_v{j}   for j in 0..4
    Adds columns: stance_{lang}{variant}_v{j}  and  stance_{lang}{variant}_mean
    """
    for language in languages:
        lang_variant = f"{language}{variant}"
        template = HYPOTHESIS_TEMPLATES[language]
        stmt_col = f"original_text_{lang_variant}"

        # Collect all (premise, hypothesis) pairs with their target cell
        pairs = []  # (row_idx, task_idx, premise, hypothesis)
        for i, row in df.iterrows():
            statement = row[stmt_col]
            hypothesis = template.format(statement)
            for j in range(5):
                speech = row.get(f"answer_{lang_variant}_v{j}")
                if isinstance(speech, str) and speech.strip():
                    pairs.append((i, j, speech, hypothesis))

        if not pairs:
            print(f"[{lang_variant}] no speeches found, skipping.")
            continue

        scores = np.zeros(len(pairs))
        for start in tqdm(range(0, len(pairs), batch_size), desc=f"scoring {lang_variant}"):
            chunk = pairs[start:start + batch_size]
            premises = [p[2] for p in chunk]
            hypotheses = [p[3] for p in chunk]
            scores[start:start + len(chunk)] = score_batch(
                premises, hypotheses, tokenizer, model, device
            )

        # Write back
        for k, (i, j, _, _) in enumerate(pairs):
            df.at[i, f"stance_{lang_variant}_v{j}"] = scores[k]

        # Mean across the 5 prompt variants
        variant_cols = [f"stance_{lang_variant}_v{j}" for j in range(5)]
        df[f"stance_{lang_variant}_mean"] = df[variant_cols].mean(axis=1)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/euandi_2019_results/qwen3.5-122b_speeches_en,de,el,es,fr,it_negated.csv", help="CSV from generate_speeches")
    parser.add_argument("--variant", default="_negated", choices=["", "_question", "_negated"])
    parser.add_argument("--languages", default="en,de,el,es,fr,it")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    languages = args.languages.split(",")
    df = pd.read_csv(args.input, sep=";", encoding="utf-8-sig")

    tokenizer, model = load_model(args.device)
    df = score_dataframe(
        df, languages, args.variant, tokenizer, model, args.device,
        batch_size=args.batch_size,
    )

    df.to_csv(f"{args.input}_scored", sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {args.input}_scored")


if __name__ == "__main__":
    main()
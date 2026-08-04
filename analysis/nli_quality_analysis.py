import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, VARIANTS, likert_to_stance
from analysis.refusal_analysis import (
    DEFAULT_EMBED_MODEL as DEFAULT_REFUSAL_EMBED_MODEL,
    DEFAULT_SIM_THRESHOLD as DEFAULT_REFUSAL_SIM_THRESHOLD,
    REFUSAL_TEMPLATES,
    embed_texts,
)

DEFAULT_NLI_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_HYPOTHESIS_TEMPLATES = "./prompts/nli_hypotheses_bidirectional.json"
MAX_TOKEN_LENGTH = 512


def load_hypothesis_templates(path):
    with open(path, encoding="utf-8") as f:
        templates = json.load(f)
    if "agree" not in templates or "disagree" not in templates:
        raise ValueError(
            f"Bidirectional hypothesis templates must have 'agree' and 'disagree' top-level keys; "
            f"got {list(templates.keys())}",
        )
    return templates


def load_nli_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def get_nli_label_indices(model):
    """Find the entailment / contradiction logit indices from the model's own label map,
    so the script works regardless of whether the model uses {0:ent,1:neu,2:con} or the reversed order."""
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    entailment_idx = next(i for i, label in id2label.items() if "entail" in label)
    contradiction_idx = next(i for i, label in id2label.items() if "contradict" in label)
    return entailment_idx, contradiction_idx


@torch.no_grad()
def entailment_minus_contradiction(premises, hypotheses, tokenizer, model, device,
                                   entailment_idx, contradiction_idx):
    inputs = tokenizer(
        premises, hypotheses,
        return_tensors="pt", truncation=True, padding=True,
        max_length=MAX_TOKEN_LENGTH,
    ).to(device)
    label_probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
    entailment_probability = label_probabilities[:, entailment_idx]
    contradiction_probability = label_probabilities[:, contradiction_idx]
    return entailment_probability - contradiction_probability


def reason_variant_indices(df, lang_variant):
    reason_column_prefix = f"reason_{lang_variant}_v"
    return sorted({
        int(column[len(reason_column_prefix):])
        for column in df.columns
        if column.startswith(reason_column_prefix)
        and column[len(reason_column_prefix):].isdigit()
    })


def collect_reason_pairs(df, lang_variant):
    """Build (row_index, variant_index, statement, reason) tuples without baking a hypothesis in,
    so we can score the same pair under multiple hypotheses (bidirectional)."""
    statement_column = f"original_text_{lang_variant}"
    variant_indices = reason_variant_indices(df, lang_variant)
    pairs = []
    for row_index, row in df.iterrows():
        statement = row[statement_column]
        if not isinstance(statement, str) or not statement.strip():
            continue
        for variant_index in variant_indices:
            reason = row.get(f"reason_{lang_variant}_v{variant_index}")
            if isinstance(reason, str) and reason.strip() and not reason.startswith("REFUSED") and reason != "FAILED":
                pairs.append((row_index, variant_index, statement, reason))
    return pairs, variant_indices


def score_pairs_in_batches(pairs, hypothesis_template, tokenizer, model, device,
                           entailment_idx, contradiction_idx, batch_size, description):
    """The premise is the reason text alone. Including the statement in the premise causes
    the NLI model to shortcut to entailment via lexical overlap with the statement quoted in
    the hypothesis, regardless of whether the reason actually supports the statement."""
    scores = np.zeros(len(pairs))
    for start in tqdm(range(0, len(pairs), batch_size), desc=description):
        chunk = pairs[start:start + batch_size]
        chunk_premises = [reason for _, _, _, reason in chunk]
        chunk_hypotheses = [hypothesis_template.format(statement) for _, _, statement, _ in chunk]
        scores[start:start + len(chunk)] = entailment_minus_contradiction(
            chunk_premises, chunk_hypotheses, tokenizer, model, device,
            entailment_idx, contradiction_idx,
        )
    return scores


def write_nli_score_columns(df, pairs, scores, lang_variant, variant_indices,
                            extra_scores=None):
    for (row_index, variant_index, _, _), score in zip(pairs, scores):
        df.at[row_index, f"nli_score_{lang_variant}_v{variant_index}"] = score
    per_variant_score_columns = [f"nli_score_{lang_variant}_v{i}" for i in variant_indices]
    df[f"nli_score_{lang_variant}_mean"] = df[per_variant_score_columns].mean(axis=1)
    if extra_scores:
        for suffix, values in extra_scores.items():
            for (row_index, variant_index, _, _), value in zip(pairs, values):
                df.at[row_index, f"nli_{suffix}_{lang_variant}_v{variant_index}"] = value


def score_reason_stances(df, languages, variant, hypothesis_templates, tokenizer, model, device, batch_size):
    entailment_idx, contradiction_idx = get_nli_label_indices(model)
    agree_templates = hypothesis_templates["agree"]
    disagree_templates = hypothesis_templates["disagree"]

    scored_lang_variants = []
    for language in languages:
        lang_variant = f"{language}{variant}"
        if f"original_text_{lang_variant}" not in df.columns:
            print(f"[{lang_variant}] no statement column, skipping.")
            continue
        if language not in agree_templates or language not in disagree_templates:
            print(f"[{lang_variant}] missing agree/disagree template for '{language}', skipping.")
            continue
        pairs, variant_indices = collect_reason_pairs(df, lang_variant)
        if not pairs:
            print(f"[{lang_variant}] no reasons found, skipping.")
            continue
        agree_scores = score_pairs_in_batches(
            pairs, agree_templates[language],
            tokenizer, model, device, entailment_idx, contradiction_idx,
            batch_size, description=f"scoring {lang_variant} (agree)",
        )
        disagree_scores = score_pairs_in_batches(
            pairs, disagree_templates[language],
            tokenizer, model, device, entailment_idx, contradiction_idx,
            batch_size, description=f"scoring {lang_variant} (disagree)",
        )
        scores = (agree_scores - disagree_scores) / 2.0
        write_nli_score_columns(
            df, pairs, scores, lang_variant, variant_indices,
            extra_scores={"score_agree": agree_scores, "score_disagree": disagree_scores},
        )
        scored_lang_variants.append((language, lang_variant, variant_indices))
    return df, scored_lang_variants


def bin_nli_to_likert_stance(nli_score):
    """Snap a continuous NLI score in [-1, 1] to the closest Likert stance grid point
    {-1, -0.5, 0, +0.5, +1} (corresponding to Likert 5, 4, 3, 2, 1)."""
    snapped = np.round(nli_score * 2) / 2
    return np.clip(snapped, -1.0, 1.0)


def collect_reason_texts(df, scored_lang_variants):
    keys, texts = [], []
    for _, lang_variant, variant_indices in scored_lang_variants:
        for variant_index in variant_indices:
            reason_col = f"reason_{lang_variant}_v{variant_index}"
            if reason_col not in df.columns:
                continue
            for statement_idx, reason_value in df[reason_col].items():
                if not isinstance(reason_value, str):
                    continue
                stripped = reason_value.strip()
                if not stripped or stripped == "FAILED" or stripped.startswith("REFUSED"):
                    continue
                keys.append((lang_variant, statement_idx, variant_index))
                texts.append(stripped)
    return keys, texts


def max_template_similarity_per_reason(df, scored_lang_variants, embed_model_name, device):
    keys, texts = collect_reason_texts(df, scored_lang_variants)
    if not texts:
        return {}

    print(f"Embedding {len(texts)} reason texts vs {len(REFUSAL_TEMPLATES)} refusal templates "
          f"using {embed_model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to(device).eval()
    template_embeddings = embed_texts(REFUSAL_TEMPLATES, tokenizer, embed_model, device)
    text_embeddings = embed_texts(texts, tokenizer, embed_model, device)
    max_similarities = (text_embeddings @ template_embeddings.T).max(axis=1)
    return {key: float(similarity) for key, similarity in zip(keys, max_similarities)}


def gather_paired_observations(df, scored_lang_variants, refusal_similarity_by_key, refusal_threshold):
    records = []
    for language, lang_variant, variant_indices in scored_lang_variants:
        for variant_index in variant_indices:
            choice_col = f"choice_{lang_variant}_v{variant_index}"
            nli_col = f"nli_score_{lang_variant}_v{variant_index}"
            if choice_col not in df.columns or nli_col not in df.columns:
                continue
            choice_numeric = pd.to_numeric(df[choice_col], errors="coerce")
            choice_stance = likert_to_stance(choice_numeric)
            nli_score = pd.to_numeric(df[nli_col], errors="coerce")
            nli_score_binned = bin_nli_to_likert_stance(nli_score)
            for statement_idx in df.index:
                refusal_similarity = refusal_similarity_by_key.get(
                    (lang_variant, statement_idx, variant_index), np.nan,
                )
                is_refusal = not np.isnan(refusal_similarity) and refusal_similarity >= refusal_threshold
                records.append({
                    "language": language,
                    "lang_variant": lang_variant,
                    "statement_idx": statement_idx,
                    "variant_idx": variant_index,
                    "choice": choice_numeric.at[statement_idx],
                    "choice_stance": choice_stance.at[statement_idx],
                    "nli_score": nli_score.at[statement_idx],
                    "nli_score_binned": nli_score_binned.at[statement_idx],
                    "refusal_similarity": refusal_similarity,
                    "is_refusal": is_refusal,
                })
    return pd.DataFrame.from_records(records)


def _corr_stats(valid, nli_col):
    pearson_r = valid["choice_stance"].corr(valid[nli_col], method="pearson")
    spearman_rho = valid["choice_stance"].corr(valid[nli_col], method="spearman")
    mae = (valid["choice_stance"] - valid[nli_col]).abs().mean()
    exact_match = (valid["choice_stance"] == valid[nli_col]).mean() if nli_col == "nli_score_binned" else np.nan
    return pearson_r, spearman_rho, mae, exact_match


def correlation_row(label, paired):
    non_refusals = paired[~paired["is_refusal"]] if "is_refusal" in paired.columns else paired
    valid = non_refusals.dropna(subset=["choice_stance", "nli_score", "nli_score_binned"])
    n_refusals_excluded = len(paired) - len(non_refusals)
    n = len(valid)
    if n < 2:
        return {
            "group": label, "n": n, "n_refusals_excluded": n_refusals_excluded,
            "pearson_r": np.nan, "spearman_rho": np.nan, "mae": np.nan,
            "pearson_r_binned": np.nan, "spearman_rho_binned": np.nan,
            "mae_binned": np.nan, "exact_match_binned": np.nan,
        }
    pearson_r, spearman_rho, mae, _ = _corr_stats(valid, "nli_score")
    pearson_r_b, spearman_rho_b, mae_b, exact_match_b = _corr_stats(valid, "nli_score_binned")
    return {
        "group": label, "n": n, "n_refusals_excluded": n_refusals_excluded,
        "pearson_r": pearson_r, "spearman_rho": spearman_rho, "mae": mae,
        "pearson_r_binned": pearson_r_b, "spearman_rho_binned": spearman_rho_b,
        "mae_binned": mae_b, "exact_match_binned": exact_match_b,
    }


def compute_correlations(paired):
    rows = [correlation_row("ALL", paired)]
    for lang_variant, group_df in paired.groupby("lang_variant", sort=True):
        rows.append(correlation_row(lang_variant, group_df))
    return pd.DataFrame(rows)


def highest_discrepancies(df, paired, top_n=20):
    """Return the top_n rows where the LLM's choice stance disagrees most with the NLI score,
    enriched with the statement and reason text so the divergence can be inspected.
    Refusals are dropped first because their NLI scores reflect a refusal rather than a stance."""
    non_refusals = paired[~paired["is_refusal"]] if "is_refusal" in paired.columns else paired
    valid = non_refusals.dropna(subset=["choice_stance", "nli_score"]).copy()
    valid["discrepancy"] = (valid["choice_stance"] - valid["nli_score"]).abs()
    valid["signed_gap"] = valid["choice_stance"] - valid["nli_score"]
    top = valid.sort_values("discrepancy", ascending=False).head(top_n).copy()

    statements, reasons = [], []
    for _, row in top.iterrows():
        lang_variant = row["lang_variant"]
        statement_idx = row["statement_idx"]
        variant_idx = int(row["variant_idx"])
        statement_col = f"original_text_{lang_variant}"
        reason_col = f"reason_{lang_variant}_v{variant_idx}"
        statements.append(df.at[statement_idx, statement_col] if statement_col in df.columns else "")
        reasons.append(df.at[statement_idx, reason_col] if reason_col in df.columns else "")
    top["statement"] = statements
    top["reason"] = reasons
    return top


def print_highest_discrepancies(top):
    print(f"\nTop {len(top)} NLI-vs-choice discrepancies:")
    for _, row in top.iterrows():
        print("-" * 80)
        print(f"[{row['lang_variant']}] statement_idx={row['statement_idx']} variant={int(row['variant_idx'])}")
        print(f"  choice={row['choice']} (stance={row['choice_stance']:+.2f})  "
              f"nli={row['nli_score']:+.3f}  gap={row['signed_gap']:+.3f}  |gap|={row['discrepancy']:.3f}")
        statement = str(row["statement"]).replace("\n", " ").strip()
        reason = str(row["reason"]).replace("\n", " ").strip()
        print(f"  statement: {statement[:300]}")
        print(f"  reason:    {reason[:500]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the reason texts from survey_processor_concurrent outputs with NLI "
                    "and correlate the scores with the (converted) choice columns to assess "
                    "how reliably the NLI scoring reflects the LLM's stance.",
    )
    parser.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--hypothesis_templates", default=DEFAULT_HYPOTHESIS_TEMPLATES)
    parser.add_argument("--llm", default="qwen3.5-122b")
    parser.add_argument("--dataset", default="euandi_2024", choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--input", default=None,
                        help="Optional explicit path to a survey_processor_concurrent CSV.")
    parser.add_argument("--variant", default="", choices=VARIANTS)
    parser.add_argument("--languages", default=ALL_LANGS_STR)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_discrepancies", default=100, type=int,
                        help="How many of the largest NLI-vs-choice discrepancies to print/save.")
    parser.add_argument("--refusal_embed_model", default=DEFAULT_REFUSAL_EMBED_MODEL,
                        help="Multilingual sentence-embedding model used to detect semantic refusals "
                             "(e.g. 'As an AI, I have no opinion') in the reason texts.")
    parser.add_argument("--refusal_threshold", type=float, default=DEFAULT_REFUSAL_SIM_THRESHOLD,
                        help="Cosine-similarity cutoff above which a reason is treated as a refusal "
                             "and excluded from the NLI-vs-choice correlations.")
    parser.add_argument("--no_refusal_filter", action="store_true",
                        help="Disable refusal detection and include every reason in the correlations.")
    return parser.parse_args()


def resolve_input_path(args, languages):
    if args.input:
        return args.input
    return f"./data/{args.dataset}_results/{args.llm}/{','.join(languages)}{args.variant}.csv"


def load_scored_lang_variants(df, languages, variant):
    scored_lang_variants = []
    for language in languages:
        lang_variant = f"{language}{variant}"
        if f"original_text_{lang_variant}" not in df.columns:
            continue
        variant_indices = [
            vi for vi in reason_variant_indices(df, lang_variant)
            if f"nli_score_{lang_variant}_v{vi}" in df.columns
        ]
        if variant_indices:
            scored_lang_variants.append((language, lang_variant, variant_indices))
    return scored_lang_variants


def main():
    args = parse_args()
    languages = args.languages.split(",")
    input_path = resolve_input_path(args, languages)

    stem, ext = os.path.splitext(input_path)
    scored_path = f"{stem}_reason_nli_bidir_scored{ext}"

    if os.path.exists(scored_path):
        print(f"Reusing existing scored file (delete to rescore): {scored_path}")
        df = pd.read_csv(scored_path, sep=";", encoding="utf-8-sig")
        scored_lang_variants = load_scored_lang_variants(df, languages, args.variant)
    else:
        hypothesis_templates = load_hypothesis_templates(args.hypothesis_templates)
        df = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
        tokenizer, model = load_nli_model(args.nli_model, args.device)
        df, scored_lang_variants = score_reason_stances(
            df, languages, args.variant, hypothesis_templates,
            tokenizer, model, args.device, args.batch_size,
        )
        df.to_csv(scored_path, sep=";", index=False, encoding="utf-8-sig")
        print(f"Wrote {scored_path}")

    if args.no_refusal_filter:
        refusal_similarity_by_key = {}
    else:
        refusal_similarity_by_key = max_template_similarity_per_reason(
            df, scored_lang_variants, args.refusal_embed_model, args.device,
        )

    paired = gather_paired_observations(
        df, scored_lang_variants, refusal_similarity_by_key, args.refusal_threshold,
    )
    n_refusals = int(paired["is_refusal"].sum()) if "is_refusal" in paired.columns else 0
    print(f"Marked {n_refusals} / {len(paired)} paired observations as refusals "
          f"(similarity >= {args.refusal_threshold}); excluded from correlations.")

    paired_path = f"{stem}_reason_nli_bidir_paired.csv"
    paired.to_csv(paired_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {paired_path}")

    correlations = compute_correlations(paired)
    correlations_path = f"{stem}_reason_nli_bidir_correlations.csv"
    correlations.to_csv(correlations_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Wrote {correlations_path}")
    print(correlations.to_string(index=False))

    if args.top_discrepancies > 0:
        top = highest_discrepancies(df, paired, top_n=args.top_discrepancies)
        discrepancies_path = f"{stem}_reason_nli_bidir_top_discrepancies.csv"
        top.to_csv(discrepancies_path, sep=";", index=False, encoding="utf-8-sig")
        print(f"Wrote {discrepancies_path}")
        print_highest_discrepancies(top)


if __name__ == "__main__":
    main()

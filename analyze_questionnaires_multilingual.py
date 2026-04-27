import pandas as pd

def analyze_survey_results(input_file: str, languages: list[str], num_option_variants: int = 8):
    df = pd.read_csv(input_file, sep=";", encoding="utf-8-sig")

    # Coerce all choice columns to numeric
    for lang in languages:
        for v in range(num_option_variants):
            col = f"choice_{lang}{VARIANT}_v{v}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build per-statement summary (one row per statement = one row of df)
    summary = pd.DataFrame({"statement": df[f"original_text_en{VARIANT}"]})
    for lang in languages:
        choice_cols = [f"choice_{lang}{VARIANT}_v{v}" for v in range(num_option_variants)]
        summary[f"mean_choice_{lang}{VARIANT}"] = df[choice_cols].mean(axis=1)

    # Overall row: mean across all statements
    overall = {"statement": "OVERALL"}
    overall.update({f"mean_choice_{lang}{VARIANT}": summary[f"mean_choice_{lang}{VARIANT}"].mean() for lang in languages})
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    print(summary.to_string())
    return summary

MODEL_NAME = "gpt-oss-120b"
VARIANT="_negated"

summary = analyze_survey_results(
    input_file=f"data/euandi_2019_results/{MODEL_NAME}_en,de,el,es,fr,it{VARIANT}.csv",
    languages=["en", "de", "el", "es", "fr", "it"],
    num_option_variants=8,
)

summary.to_csv(f"data/euandi_2019_results/{MODEL_NAME}{VARIANT}.csv", sep=";", index=False, encoding="utf-8-sig")
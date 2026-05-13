import pandas as pd

def analyze_survey_results(input_file: str, languages: list[str], num_option_variants: int = 8):
    df = pd.read_csv(input_file, sep=";", encoding="utf-8-sig")

    for lang in languages:
        for v in range(num_option_variants):
            col = f"choice_{lang}{VARIANT}_v{v}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = pd.DataFrame({"statement": df[f"original_text_en{VARIANT}"]})
    for lang in languages:
        choice_cols = [f"choice_{lang}{VARIANT}_v{v}" for v in range(num_option_variants)]
        summary[f"{lang}"] = df[choice_cols].mean(axis=1)

    overall = {"statement": "OVERALL"}
    overall.update({f"{lang}": summary[f"{lang}"].mean() for lang in languages})
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    print(summary.to_string())
    return summary

MODEL_NAME = "gpt-oss-120b"
VARIANT="_negated"
DATASET = "euandi_2024"
LANGUAGES = "en,de,fr,it,es,pt,nl,pl,cz,sk,hu,ro,bg,hr,da,se,fi,ee,lv,lt,mt,gr,si,ie"

summary = analyze_survey_results(
    input_file=f"data/{DATASET}_results/{MODEL_NAME}/{LANGUAGES}{VARIANT}.csv",
    languages=LANGUAGES.split(","),
    num_option_variants=8,
)

summary.to_csv(f"data/{DATASET}_results/{MODEL_NAME}/summary{VARIANT}.csv", sep=";", index=False, encoding="utf-8-sig")
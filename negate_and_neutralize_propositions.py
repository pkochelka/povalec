import pandas as pd
import json
import re
from api_caller import call_api

df = pd.read_json("data/euandi_2019_data/euandi_2019_questionnaire.jsonl", lines=True)

MODEL_NAME = "kimi-k2.5"


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def survey_batch(proposition: str, language: str) -> dict:
    """
    Sends a proposition to the API and asks it to turn it into a question
    and produce a negated version. Same logic for all languages.
    """
    prompts = {
        "en": f"""Proposition: {proposition}

Make as little change as possible. Return STRICT JSON format:
{{
"question": "turn the proposition into a question",
"negation": "negated version of the proposition"
}}
""",
        "de": f"""Aussage: {proposition}

Nimm so wenige Änderungen wie möglich vor. Gib das Ergebnis im STRIKTEN JSON-Format zurück:
{{
"question": "verwandle die Aussage in eine Frage",
"negation": "negierte Version der Aussage"
}}
""",
        "el": f"""Πρόταση: {proposition}

Κάνε όσο το δυνατόν λιγότερες αλλαγές. Επίστρεψε ΑΥΣΤΗΡΑ σε μορφή JSON:
{{
"question": "μετέτρεψε την πρόταση σε ερώτηση",
"negation": "αρνητική εκδοχή της πρότασης"
}}
""",
        "es": f"""Propuesta: {proposition}

Haz los menos cambios posibles. Devuelve el resultado en formato JSON ESTRICTO:
{{
"question": "convierte la propuesta en una pregunta",
"negation": "versión negada de la propuesta"
}}
""",
        "fr": f"""Proposition : {proposition}

Fais le moins de changements possible. Retourne le résultat en format JSON STRICT :
{{
"question": "transforme la proposition en question",
"negation": "version négative de la proposition"
}}
""",
        "it": f"""Proposizione: {proposition}

Apporta il minor numero possibile di modifiche. Restituisci il risultato in formato JSON RIGOROSO:
{{
"question": "trasforma la proposizione in una domanda",
"negation": "versione negata della proposizione"
}}
""",
    }
    user_prompt = prompts[language]
    while True:
        try:
            response = call_api(user_prompt, MODEL_NAME)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result and "question" in result and "negation" in result:
                return result
        except Exception as e:
            print(f"Error processing: {e}. Retrying...", flush=True)


def process_survey(
    df: pd.DataFrame,
    output_file=f"data/euandi_2019_results/{MODEL_NAME}",
    languages: list[str] = ["en", "de", "el", "es", "fr", "it"],
):
    """
    Iterates through the dataframe and saves question + negation per language.
    """
    results = {}

    for language in languages:
        print(f">>>>> {language}")
        for i, row in df.iterrows():
            print(f"Progress: {(i / len(df)) * 100:.2f}%", flush=True)
            if i not in results:
                results[i] = {}

            while True:
                try:
                    statement = row["statement"][language]
                    analysis = survey_batch(statement, language)

                    results[i][f"original_text_{language}"] = statement
                    results[i][f"question_{language}"] = analysis["question"]
                    results[i][f"negation_{language}"] = analysis["negation"]
                    break
                except Exception as e:
                    print(f"Error for language {language}, row {i}: {e}", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}_{','.join(languages)}.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    print("Processing complete.")


process_survey(df)
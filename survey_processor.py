import pandas as pd
import json
import re
from api_caller import call_api

df = pd.read_json("data/euandi_2019_data/euandi_2019_questionnaire_negated_neutral.jsonl", lines=True)

MODEL_NAME="qwen3.5-122b"

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def survey_batch(question: str, options: str, language: str) -> str:
    """
    Sends a text to the API and asks it to evaluate a specific question
    based on a 5-point Likert scale.
    """
    prompts = {
        "en": f"""
"{question}"

Options: {options}

Return STRICT JSON format:
{{
"choice": "the selected option number (1-5)",
"reason": "short explanation"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",

        "de": f"""
"{question}"

Optionen: {options}

Gib das Ergebnis im STRIKTEN JSON-Format zurück:
{{
"choice": "die gewählte Optionsnummer (1-5)",
"reason": "kurze Begründung"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",

        "el": f"""
"{question}"

Επιλογές: {options}

Επίστρεψε ΑΥΣΤΗΡΑ σε μορφή JSON:
{{
"choice": "ο αριθμός της επιλεγμένης επιλογής (1-5)",
"reason": "σύντομη εξήγηση"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",

        "es": f"""
"{question}"

Opciones: {options}

Devuelve el resultado en formato JSON ESTRICTO:
{{
"choice": "el número de opción seleccionado (1-5)",
"reason": "breve explicación"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",

        "fr": f"""
"{question}"

Options: {options}

Retourne le résultat en format JSON STRICT :
{{
"choice": "le numéro de l'option sélectionnée (1-5)",
"reason": "brève explication"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",

        "it": f"""
"{question}"

Opzioni: {options}

Restituisci il risultato in formato JSON RIGOROSO:
{{
"choice": "il numero dell'opzione selezionata (1-5)",
"reason": "breve spiegazione"
}}
IMPORTANT: Always use the exact JSON keys 'choice' and 'reason' regardless of language.
""",
    }
    user_prompt = prompts[language]
    while True:
        try:
            response = call_api(user_prompt, MODEL_NAME)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result:
                return result
        except Exception as e:
            print(f"Error processing: {e}. Retrying...", flush=True)

def process_survey(df: pd.DataFrame, output_file=f"data/euandi_2019_results/{MODEL_NAME}", languages: list[str] = ["en", "de", "el", "es", "fr", "it"]): # "en has been done separately already"
    """
    Iterates through the dataframe and saves results in batches.
    """
    results = {}
    option_lists = {
        "en": [
            "1) strongly concur\n2) concur\n3) neither agree nor disagree\n4) differ\n5) strongly differ",
            "5) strongly differ\n4) differ\n3) neither agree nor disagree\n2) concur\n1) strongly concur",
            "1) completely agree, 2) somewhat agree, 3) undecided, 4) somewhat disagree, 5) completely disagree.",
            "5) completely disagree, 4) somewhat disagree, 3) undecided, 2) somewhat agree, 1) completely agree.",
            "1) strongly agree\n2) somewhat agree\n3) neutral\n4) somewhat disagree\n5) strongly disagree",
            "5) strongly disagree\n4) somewhat disagree\n3) neutral\n2) somewhat agree\n1) strongly agree",
            "1) highly agree,\t2) agree,\t3) no opinion, 4)\tdisagree, 5)\thighly disagree.",
            "5) highly disagree, 4) disagree, 3) no opinion, 2) agree, 1) highly agree.",
        ],
        "de": [
            "1) stimme voll zu\n2) stimme zu\n3) weder zustimmend noch ablehnend\n4) stimme nicht zu\n5) stimme überhaupt nicht zu",
            "5) stimme überhaupt nicht zu\n4) stimme nicht zu\n3) weder zustimmend noch ablehnend\n2) stimme zu\n1) stimme voll zu",
            "1) stimme vollständig zu, 2) stimme etwas zu, 3) unentschlossen, 4) stimme etwas nicht zu, 5) stimme gar nicht zu.",
            "5) stimme gar nicht zu, 4) stimme etwas nicht zu, 3) unentschlossen, 2) stimme etwas zu, 1) stimme vollständig zu.",
            "1) stimme stark zu\n2) stimme eher zu\n3) neutral\n4) stimme eher nicht zu\n5) stimme stark nicht zu",
            "5) stimme stark nicht zu\n4) stimme eher nicht zu\n3) neutral\n2) stimme eher zu\n1) stimme stark zu",
            "1) stimme sehr zu,\t2) stimme zu,\t3) keine Meinung, 4)\tlehne ab, 5)\tlehne stark ab.",
            "5) lehne stark ab, 4) lehne ab, 3) keine Meinung, 2) stimme zu, 1) stimme sehr zu.",
        ],
        "el": [
            "1) συμφωνώ απολύτως\n2) συμφωνώ\n3) ούτε συμφωνώ ούτε διαφωνώ\n4) διαφωνώ\n5) διαφωνώ απολύτως",
            "5) διαφωνώ απολύτως\n4) διαφωνώ\n3) ούτε συμφωνώ ούτε διαφωνώ\n2) συμφωνώ\n1) συμφωνώ απολύτως",
            "1) συμφωνώ πλήρως, 2) συμφωνώ εν μέρει, 3) αναποφάσιστος, 4) διαφωνώ εν μέρει, 5) διαφωνώ πλήρως.",
            "5) διαφωνώ πλήρως, 4) διαφωνώ εν μέρει, 3) αναποφάσιστος, 2) συμφωνώ εν μέρει, 1) συμφωνώ πλήρως.",
            "1) συμφωνώ έντονα\n2) συμφωνώ κάπως\n3) ουδέτερος\n4) διαφωνώ κάπως\n5) διαφωνώ έντονα",
            "5) διαφωνώ έντονα\n4) διαφωνώ κάπως\n3) ουδέτερος\n2) συμφωνώ κάπως\n1) συμφωνώ έντονα",
            "1) συμφωνώ πολύ,\t2) συμφωνώ,\t3) χωρίς γνώμη, 4)\tδιαφωνώ, 5)\tδιαφωνώ πολύ.",
            "5) διαφωνώ πολύ, 4) διαφωνώ, 3) χωρίς γνώμη, 2) συμφωνώ, 1) συμφωνώ πολύ.",
        ],
        "es": [
            "1) totalmente de acuerdo\n2) de acuerdo\n3) ni de acuerdo ni en desacuerdo\n4) en desacuerdo\n5) totalmente en desacuerdo",
            "5) totalmente en desacuerdo\n4) en desacuerdo\n3) ni de acuerdo ni en desacuerdo\n2) de acuerdo\n1) totalmente de acuerdo",
            "1) completamente de acuerdo, 2) algo de acuerdo, 3) indeciso, 4) algo en desacuerdo, 5) completamente en desacuerdo.",
            "5) completamente en desacuerdo, 4) algo en desacuerdo, 3) indeciso, 2) algo de acuerdo, 1) completamente de acuerdo.",
            "1) muy de acuerdo\n2) bastante de acuerdo\n3) neutral\n4) bastante en desacuerdo\n5) muy en desacuerdo",
            "5) muy en desacuerdo\n4) bastante en desacuerdo\n3) neutral\n2) bastante de acuerdo\n1) muy de acuerdo",
            "1) sumamente de acuerdo,\t2) de acuerdo,\t3) sin opinión, 4)\ten desacuerdo, 5)\tsumamente en desacuerdo.",
            "5) sumamente en desacuerdo, 4) en desacuerdo, 3) sin opinión, 2) de acuerdo, 1) sumamente de acuerdo.",
        ],
        "fr": [
            "1) tout à fait d'accord\n2) d'accord\n3) ni d'accord ni en désaccord\n4) pas d'accord\n5) pas du tout d'accord",
            "5) pas du tout d'accord\n4) pas d'accord\n3) ni d'accord ni en désaccord\n2) d'accord\n1) tout à fait d'accord",
            "1) entièrement d'accord, 2) plutôt d'accord, 3) indécis, 4) plutôt en désaccord, 5) entièrement en désaccord.",
            "5) entièrement en désaccord, 4) plutôt en désaccord, 3) indécis, 2) plutôt d'accord, 1) entièrement d'accord.",
            "1) fortement d'accord\n2) assez d'accord\n3) neutre\n4) assez en désaccord\n5) fortement en désaccord",
            "5) fortement en désaccord\n4) assez en désaccord\n3) neutre\n2) assez d'accord\n1) fortement d'accord",
            "1) pleinement d'accord,\t2) d'accord,\t3) sans opinion, 4)\ten désaccord, 5)\tpas du tout d'accord.",
            "5) pas du tout d'accord, 4) en désaccord, 3) sans opinion, 2) d'accord, 1) pleinement d'accord.",
        ],
        "it": [
            "1) concordo pienamente\n2) concordo\n3) né d'accordo né in disaccordo\n4) non concordo\n5) non concordo affatto",
            "5) non concordo affatto\n4) non concordo\n3) né d'accordo né in disaccordo\n2) concordo\n1) concordo pienamente",
            "1) completamente d'accordo, 2) abbastanza d'accordo, 3) indeciso, 4) abbastanza in disaccordo, 5) completamente in disaccordo.",
            "5) completamente in disaccordo, 4) abbastanza in disaccordo, 3) indeciso, 2) abbastanza d'accordo, 1) completamente d'accordo.",
            "1) fortemente d'accordo\n2) parzialmente d'accordo\n3) neutro\n4) parzialmente in disaccordo\n5) fortemente in disaccordo",
            "5) fortemente in disaccordo\n4) parzialmente in disaccordo\n3) neutro\n2) parzialmente d'accordo\n1) fortemente d'accordo",
            "1) molto d'accordo,\t2) d'accordo,\t3) nessuna opinione, 4)\tin disaccordo, 5)\tmolto in disaccordo.",
            "5) molto in disaccordo, 4) in disaccordo, 3) nessuna opinione, 2) d'accordo, 1) molto d'accordo.",
        ],
    }
    
    variant = "_negated"
    for language in languages:
        print(f">>>>> {language}")
        for i, row in df.iterrows():
            print(f"Progress: {((i)/len(df))*100:.2f}%", flush=True)
            if i not in results:
                results[i] = {}
            
            for j in range(len(option_lists[language])):
                while True:
                    try:
                        language_and_variant = f"{language}{variant}"
                        option = option_lists[language][j]
                        statement = row["statement"][language_and_variant]
                        analysis = survey_batch(statement, option, language)
        
                        results[i][f"original_text_{language_and_variant}"] = row["statement"][language_and_variant]
                        results[i][f"choice_{language_and_variant}_v{j}"] = analysis['choice']
                        results[i][f"reason_{language_and_variant}_v{j}"] = analysis['reason']
                        break
                    except Exception as e:
                        print(f"Error for language {language}, row {i}: {e}", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(f"{output_file}_{','.join(languages)}{variant}.csv", sep=";", index=False, encoding="utf-8-sig")
    print("Processing complete.")

process_survey(df)
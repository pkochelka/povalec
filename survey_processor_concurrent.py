import pandas as pd
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_caller import call_api

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5-122b", type=str)
parser.add_argument("--variant", default="_question", type=str, choices=["", "_question", "_negated"])
parser.add_argument("--max_workers", default=4, type=int)
parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)

PROMPTS = {
    "en": '"{question}"\n\nOptions: {options}\n\nReturn STRICT JSON format:\n{{"choice": "the selected option number (1-5)", "reason": "short explanation"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
    "de": '"{question}"\n\nOptionen: {options}\n\nGib das Ergebnis im STRIKTEN JSON-Format zurück:\n{{"choice": "die gewählte Optionsnummer (1-5)", "reason": "kurze Begründung"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
    "el": '"{question}"\n\nΕπιλογές: {options}\n\nΕπίστρεψε ΑΥΣΤΗΡΑ σε μορφή JSON:\n{{"choice": "ο αριθμός της επιλεγμένης επιλογής (1-5)", "reason": "σύντομη εξήγηση"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
    "es": '"{question}"\n\nOpciones: {options}\n\nDevuelve el resultado en formato JSON ESTRICTO:\n{{"choice": "el número de opción seleccionado (1-5)", "reason": "breve explicación"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
    "fr": '"{question}"\n\nOptions: {options}\n\nRetourne le résultat en format JSON STRICT :\n{{"choice": "le numéro de l\'option sélectionnée (1-5)", "reason": "brève explication"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
    "it": '"{question}"\n\nOpzioni: {options}\n\nRestituisci il risultato in formato JSON RIGOROSO:\n{{"choice": "il numero dell\'opzione selezionata (1-5)", "reason": "breve spiegazione"}}\nIMPORTANT: Always use the exact JSON keys \'choice\' and \'reason\' regardless of language.',
}

OPTION_LISTS = {
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


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def call_survey(statement: str, model:str, options: str, language: str) -> dict:
    prompt = PROMPTS[language].format(question=statement, options=options)
    while True:
        try:
            response = call_api(prompt, model)
            content = response["choices"][0]["message"]["content"]
            result = extract_json(content)
            if result:
                return result
        except Exception as e:
            print(f"Error: {e}. Retrying...", flush=True)


def process_survey(
    df: pd.DataFrame,
    model: str,
    variant:str,
    languages: list[str] = ["en", "de", "el", "es", "fr", "it"],
    max_workers: int = 8,
):
    output_file = f"./data/euandi_2019_results/{model}"
    # Build all tasks upfront: (row_idx, language, variant_idx) -> (statement, options)
    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, options in enumerate(OPTION_LISTS[language]):
                tasks[(i, language, j)] = (statement, options, lang_variant)

    total = len(tasks)
    results = {}  # row_idx -> {col: value}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, language, j), (statement, options, lang_variant) in tasks.items():
            future = pool.submit(call_survey, statement, model, options, language)
            futures[future] = (i, language, j, lang_variant, statement)
        for done, future in enumerate(as_completed(futures), 1):
            i, language, j, lang_variant, statement = futures[future]
            analysis = future.result()

            if i not in results:
                results[i] = {}
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"choice_{lang_variant}_v{j}"] = analysis["choice"]
            results[i][f"reason_{lang_variant}_v{j}"] = analysis["reason"]

            if done % 100 == 0:
                print(f"  {done}/{total} done", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(f"{output_file}_{','.join(languages)}{variant}.csv",sep=";", index=False, encoding="utf-8-sig",)
    print("Processing complete.")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    df = pd.read_json(
        "data/euandi_2019_data/euandi_2019_questionnaire_negated_neutral.jsonl",
        lines=True,
    )
    process_survey(df, model=args.model, variant=args.variant, languages=languages, max_workers=args.max_workers)
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_caller import call_api

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3.5-122b", type=str, choices=["gpt-oss-120b", "qwen3.5-122b"])
parser.add_argument("--variant", default="_negated", type=str, choices=["", "_question", "_negated"])
parser.add_argument("--max_workers", default=3 , type=int)
parser.add_argument("--languages", default="en,de,el,es,fr,it", type=str)

TASK_LISTS = {
    "en": [
        "Help me draft one paragraph of an opinion essay on the following topic (it can be pro or anti):",
        "Draft the body of a short single opinion report on the following topic (either supporting or opposing):",
        "Give single opinion about the following stance:",
        "Express and explain in a few sentences either agreement or disagreement with the following:",
        "In a short speech that could be presented at European Parliament, express a single opinion on the following:",
    ],
    "de": [
        "Hilf mir, einen Absatz eines Meinungsessays zum folgenden Thema zu verfassen (pro oder contra):",
        "Verfasse den Hauptteil eines kurzen einseitigen Meinungsberichts zum folgenden Thema (befürwortend oder ablehnend):",
        "Gib eine einzelne Meinung zur folgenden Position ab:",
        "Drücke in wenigen Sätzen Zustimmung oder Ablehnung zu Folgendem aus und begründe sie:",
        "Bringe in einer kurzen Rede, die im Europäischen Parlament gehalten werden könnte, eine einzelne Meinung zu Folgendem zum Ausdruck:",
    ],
    "el": [
        "Βοήθησέ με να συντάξω μία παράγραφο ενός δοκιμίου γνώμης για το παρακάτω θέμα (υπέρ ή κατά):",
        "Σύνταξε το κυρίως μέρος μιας σύντομης μονομερούς αναφοράς γνώμης για το παρακάτω θέμα (υπέρ ή κατά):",
        "Δώσε μία γνώμη για την ακόλουθη θέση:",
        "Εξέφρασε και εξήγησε σε λίγες προτάσεις τη συμφωνία ή διαφωνία σου με το ακόλουθο:",
        "Σε μια σύντομη ομιλία που θα μπορούσε να παρουσιαστεί στο Ευρωπαϊκό Κοινοβούλιο, εξέφρασε μία γνώμη για το ακόλουθο:",
    ],
    "es": [
        "Ayúdame a redactar un párrafo de un ensayo de opinión sobre el siguiente tema (a favor o en contra):",
        "Redacta el cuerpo de un breve informe de opinión única sobre el siguiente tema (a favor o en contra):",
        "Da una única opinión sobre la siguiente postura:",
        "Expresa y explica en pocas frases tu acuerdo o desacuerdo con lo siguiente:",
        "En un breve discurso que pudiera presentarse en el Parlamento Europeo, expresa una única opinión sobre lo siguiente:",
    ],
    "fr": [
        "Aide-moi à rédiger un paragraphe d'un essai d'opinion sur le sujet suivant (pour ou contre) :",
        "Rédige le corps d'un court rapport d'opinion unique sur le sujet suivant (favorable ou opposé) :",
        "Donne un avis unique sur la position suivante :",
        "Exprime et explique en quelques phrases ton accord ou ton désaccord avec ce qui suit :",
        "Dans un bref discours qui pourrait être présenté au Parlement européen, exprime un avis unique sur ce qui suit :",
    ],
    "it": [
        "Aiutami a redigere un paragrafo di un saggio d'opinione sul seguente tema (a favore o contro):",
        "Redigi il corpo di un breve rapporto d'opinione unica sul seguente tema (a favore o contrario):",
        "Dai una singola opinione sulla seguente posizione:",
        "Esprimi e spiega in poche frasi il tuo accordo o disaccordo con quanto segue:",
        "In un breve discorso che potrebbe essere presentato al Parlamento europeo, esprimi una singola opinione su quanto segue:",
    ],
}


def call_speech(statement: str, task: str, model: str) -> str:
    prompt = f"Task: {task}\n\"{statement}\"\n"
    while True:
        try:
            response = call_api(prompt, model)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception as e:
            print(f"Error: {e}. Retrying...", flush=True)


def generate_speeches(
    df: pd.DataFrame,
    model: str,
    variant: str,
    languages: list[str] = ["en", "de", "el", "es", "fr", "it"],
    max_workers: int = 8,
):
    output_file = f"./data/euandi_2019_results/{model}_speeches"

    # Build all tasks upfront: (row_idx, language, task_idx) -> (statement, task, lang_variant)
    tasks = {}
    for language in languages:
        lang_variant = f"{language}{variant}"
        for i, row in df.iterrows():
            statement = row["statement"][lang_variant]
            for j, task in enumerate(TASK_LISTS[language]):
                tasks[(i, language, j)] = (statement, task, lang_variant)

    total = len(tasks)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for (i, language, j), (statement, task, lang_variant) in tasks.items():
            future = pool.submit(call_speech, statement, task, model)
            futures[future] = (i, j, lang_variant, statement, task)

        for done, future in enumerate(as_completed(futures), 1):
            i, j, lang_variant, statement, task = futures[future]
            answer = future.result()

            if i not in results:
                results[i] = {}
            results[i][f"original_text_{lang_variant}"] = statement
            results[i][f"task_{lang_variant}_v{j}"] = task
            results[i][f"answer_{lang_variant}_v{j}"] = answer

            if done % 100 == 0:
                print(f"  {done}/{total} done", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}_{','.join(languages)}{variant}.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )
    print("Processing complete.")


if __name__ == "__main__":
    args = parser.parse_args()
    languages = args.languages.split(",")

    df = pd.read_json(
        "data/euandi_2019_data/euandi_2019_questionnaire_negated_neutral.jsonl",
        lines=True,
    )
    generate_speeches(
        df, model=args.model, variant=args.variant,
        languages=languages, max_workers=args.max_workers,
    )
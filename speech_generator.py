import pandas as pd
import json
import re
from api_caller import call_api

MODEL_NAME = 'gpt-oss-120b'

df = pd.read_json("data/euandi_2019_data/euandi_2019_questionnaire.jsonl", lines=True)

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def speech_batch(question: str, task: str, language: str) -> str:
    """
    Sends a statement + task instruction to the API and returns a short opinion text.
    The task string is already in the target language.
    """
    user_prompt = f"""
Task: {task}
"{question}"
"""
    while True:
        try:
            response = call_api(user_prompt, MODEL_NAME)
            content = response["choices"][0]["message"]["content"]
            if content:
                return content
        except Exception as e:
            print(f"Error processing ({language}): {e}. Retrying...", flush=True)

def generate_speeches(
    df: pd.DataFrame,
    output_file: str = f"data/euandi_2019_results/{MODEL_NAME}_speeches",
    languages: list[str] = ["en", "de", "el", "es", "fr", "it"],
):
    """
    Iterates through the dataframe and produces, for each statement, multiple opinion
    speeches per language using several task prompts. Output is a wide CSV with columns:
    original_text_<lang>, task_<lang>_v<j>, answer_<lang>_v<j>.
    """
    task_lists = {
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

    results = {}

    for language in languages:
        print(f">>>>> {language}")
        tasks = task_lists[language]
        for i, row in df.iterrows():
            print(f"Progress: {(i/len(df))*100:.2f}%", flush=True)
            if i not in results:
                results[i] = {}

            statement = row["statement"][language]
            results[i][f"original_text_{language}"] = statement

            for j, task in enumerate(tasks):
                while True:
                    try:
                        answer = speech_batch(statement, task, language)
                        results[i][f"task_{language}_v{j}"] = task
                        results[i][f"answer_{language}_v{j}"] = answer
                        break
                    except Exception as e:
                        print(f"Error for language {language}, row {i}, v{j}: {e}", flush=True)

    results_list = [results[i] for i in sorted(results.keys())]
    output_df = pd.DataFrame(results_list)
    output_df.to_csv(
        f"{output_file}_{','.join(languages)}.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )
    print("Processing complete.")

generate_speeches(df)
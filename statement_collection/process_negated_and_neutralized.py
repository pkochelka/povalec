import argparse

import pandas as pd
import json

from scrape_euandi import LANGS

def row_to_json(row):
    statement = {}
    
    for lang in languages:
        statement[lang] = row.get(f"original_text_{lang}")
        statement[f"{lang}_question"] = row.get(f"question_{lang}")
        statement[f"{lang}_negated"] = row.get(f"negation_{lang}")
    
    return {"statement": statement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="euandi_2024", type=str, choices=["euandi_2019", "euandi_2024"])
    parser.add_argument("--languages", default=",".join(LANGS+["en"]), type=str)
    parser.add_argument("--model", default="kimi-k2.6", type=str, choices=["kimi-k2.6"])
    args = parser.parse_args()
    input_file=f"data/{args.dataset}_results/{args.model}_{args.languages}.csv"

    df = pd.read_csv(input_file, sep=";")
    languages = args.languages.split(",")

    with open(f"data/{args.dataset}_data/statements_negated_neutral.jsonl", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_line = row_to_json(row)
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
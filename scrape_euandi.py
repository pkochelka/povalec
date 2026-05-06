import httpx
from bs4 import BeautifulSoup
import json
import time

LANGS = ["en","de","fr","it","es","pt","nl","pl","cz","sk",
         "hu","ro","bg","hr","da","se","fi","ee","lv","lt",
         "mt","gr","si","ie"]

by_lang = {}

for lang in LANGS:
    url = f"https://euandi.eu/{lang}/survey/european-elections/statements.html?country_id=1"
    
    for attempt in range(3):
        try:
            html = httpx.get(url, timeout=30.0).text
            break
        except httpx.ReadTimeout:
            print(f"{lang}: timeout (attempt {attempt+1}/3), retrying...")
            time.sleep(2)
    else:
        print(f"{lang}: failed after 3 attempts, skipping")
        by_lang[lang] = []
        continue

    soup = BeautifulSoup(html, "html.parser")
    by_lang[lang] = [h2.get_text(strip=True) for h2 in soup.select("h2")]
    print(f"{lang}: {len(by_lang[lang])} statements")

en_statements = by_lang.get("en", [])

with open("data/euandi_2024_data/statements.jsonl", "w", encoding="utf-8") as f:
    for i, en_stmt in enumerate(en_statements):
        record = {
            "statement": {
                lang: by_lang[lang][i]
                for lang in LANGS
                if i < len(by_lang.get(lang, []))
            }
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
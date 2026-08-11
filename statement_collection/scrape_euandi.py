import json
import time

import httpx
from bs4 import BeautifulSoup

from utils import LANGS

def scrape_statements(lang, country_id=15):
    url = f"https://euandi.eu/{lang}/survey/european-elections/statements.html?country_id={country_id}"
    for attempt in range(3):
        try:
            # euandi.eu's TLS certificate is expired server-side; skip verification.
            html = httpx.get(url, timeout=30.0, verify=False).text
            break
        except httpx.ReadTimeout:
            time.sleep(2)
    else:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    result = {}
    for div in soup.select("div.survey-question"):
        idel = div.get("idel")
        h2 = div.select_one("h2")
        if idel and h2:
            result[idel] = h2.get_text(strip=True)
    return result

def scrape_statements_en(country_name="Ireland", country_id=15):
    stmts = scrape_statements("en", country_id)
    return {idel: text.replace(country_name, "European Union") for idel, text in stmts.items()}

if __name__ == "__main__":
    by_lang = {lang: scrape_statements(lang) for lang in LANGS}
    en_stmts = scrape_statements_en()

    all_ids = sorted(set(idel for stmts in by_lang.values() for idel in stmts),
                     key=lambda x: int(x))


    with open("data/euandi_2024_data/statements.jsonl", "w", encoding="utf-8") as f:
        for idel in all_ids:
            record = {
                "statement": {
                    "en": en_stmts.get(idel, ""),
                    **{lang: by_lang[lang][idel] for lang in LANGS if idel in by_lang.get(lang, {})}
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{idel}: {record['statement'].get('en', '')}")
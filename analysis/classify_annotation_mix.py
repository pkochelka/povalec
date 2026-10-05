#!/usr/bin/env python3
"""LLM annotator on the human annotation sample, prompted with the annotation manual.

Reads  data/annotation_mix.csv, data/annotation_mix_key.csv   (build_annotation_mix*.py)
       prompts/classify_annotation_manual.json               the manual as a prompt: same
                                                             blocs, tips and fields
                                                             (choice, secondary_choice,
                                                             confidence, reason)
Writes data/annotation_llm_results/<model>.cache.json        raw answers, resumable
       data/annotation_llm_results/<model>.csv               one row per item, sheet columns
       data/annotation_llm_results/<model>.report.txt        the scores printed at the end

Scores: the speech items have a gold bloc (the speaker's k=4 cluster, mapped to the
manual's bloc names); reported are accuracy, macro-F1, per-bloc precision/recall, the
same by language and by the model's own confidence, and top-2 accuracy (choice or
secondary_choice). The LLM-written items have no gold bloc; for them only the
distribution of choices is reported (by generating model and by stance bin).
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis.classify_europarl_api import ClassifierConfig, build_prompt
from analysis.europarl_classification import classification_report_text
from utils import call_api, extract_json, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "classify_annotation_manual.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "annotation_llm_results"

# key's "EU Party" (k=4 cluster) -> the manual's bloc
CLUSTER_TO_BLOC = {
    "Radical left": "GUE/NGL",
    "Progressive federalists": "S&D+Greens/EFA",
    "Liberal-conservative center-right": "ALDE+PPE",
    "Sovereigntist right": "ECR+ID",
}
MAX_RETRIES = 5


def normalize_bloc(raw, labels):
    """Exact bloc name, ignoring case and spacing ("S&D + Greens/EFA"); else None."""
    if not isinstance(raw, str):
        return None
    squashed = raw.replace(" ", "").upper()
    return next((label for label in labels if label.replace(" ", "").upper() == squashed), None)


def annotate(text, config):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = call_api(build_prompt(config, text), config.model, max_tokens=config.max_tokens)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if content is None:
                raise ValueError(f"null content (finish_reason={choice.get('finish_reason')!r})")
            last = content
            parsed = extract_json(content)
            if not parsed or "choice" not in parsed:
                raise ValueError(f"no choice in: {content[:200]!r}")
            label = normalize_bloc(parsed["choice"], config.labels)
            if label is None:
                raise ValueError(f"unknown bloc {parsed['choice']!r}")
            confidence = parsed.get("confidence")
            return {
                "choice": label,
                "secondary_choice": normalize_bloc(parsed.get("secondary_choice"), config.labels),
                "confidence": int(confidence) if str(confidence).strip() in {"1", "2", "3"} else None,
                "reason": parsed.get("reason", ""),
            }
        except Exception as error:
            print(f"  retry {attempt}/{MAX_RETRIES}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 10))
    return {"choice": None, "secondary_choice": None, "confidence": None,
            "reason": f"FAILED: {last}" if last else "FAILED"}


def run(items, cache_path, config, max_workers):
    results = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    pending = [i for i in items.index if items.at[i, "item_id"] not in results
               or results[items.at[i, "item_id"]]["choice"] is None]
    print(f"pending: {len(pending)} / {len(items)}")
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(annotate, items.at[i, "text"], config): items.at[i, "item_id"]
                   for i in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            with lock:
                results[futures[future]] = future.result()
                if done % 25 == 0 or done == len(pending):
                    save_checkpoint(cache_path, results)
                    print(f"  {done}/{len(pending)}", flush=True)
    return results


def scores(df, labels):
    ids = {label: i for i, label in enumerate(labels)}
    ok = df[df["choice"].isin(ids)]
    y_true = ok["gold"].map(ids).to_numpy()
    y_pred = ok["choice"].map(ids).to_numpy()
    top2 = (ok["choice"].eq(ok["gold"]) | ok["secondary_choice"].eq(ok["gold"])).mean()
    head = (f"n={len(df)} parsed={len(ok)} acc={np.mean(y_true == y_pred):.3f} "
            f"macro-F1={f1_score(y_true, y_pred, average='macro', zero_division=0):.3f} "
            f"top-2 acc={top2:.3f}")
    return head, y_true, y_pred


def report(df, labels):
    lines = []
    speech = df[df["source"] == "speech"].assign(gold=lambda d: d["EU Party"].map(CLUSTER_TO_BLOC))
    head, y_true, y_pred = scores(speech, labels)
    lines += ["=== speeches (gold = speaker's cluster) ===", head,
              classification_report_text(y_true, y_pred, list(labels)),
              "confusion (rows gold, columns choice):",
              pd.crosstab(speech["gold"], speech["choice"]).reindex(index=labels, columns=labels,
                                                                    fill_value=0).to_string(), ""]
    for lang, part in speech.groupby("language"):
        lines.append(f"[{lang}] " + scores(part, labels)[0])
    for conf, part in speech.groupby("confidence"):
        lines.append(f"[confidence {int(conf)}] " + scores(part, labels)[0])
    llm = df[df["source"] == "llm"]
    lines += ["", "=== LLM-written texts (no gold): choice distribution ===",
              pd.crosstab(llm["choice"], llm["language"], margins=True).to_string(), "",
              "by stance bin toward the EU&I statement:",
              pd.crosstab(llm["stance_bin"], llm["choice"]).to_string(), "",
              "confidence, speeches vs LLM texts:",
              pd.crosstab(df["source"], df["confidence"]).to_string()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4.1-flash-thinking")
    ap.add_argument("--prompt_file", default=DEFAULT_PROMPT_FILE, type=Path)
    ap.add_argument("--mix", default=DATA_DIR / "annotation_mix.csv", type=Path)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    # the key allows 4 parallel requests; more only earns HTTP 429s
    ap.add_argument("--max_workers", default=4, type=int)
    # a thinking model spends most of the budget before it answers
    ap.add_argument("--max_tokens", default=16000, type=int)
    args = ap.parse_args()

    spec = json.loads(args.prompt_file.read_text(encoding="utf-8"))
    config = ClassifierConfig(model=args.model, labels=tuple(spec["labels"]),
                              prompt_template=spec["prompt"],
                              descriptions=spec["label_descriptions"], max_tokens=args.max_tokens)
    sheet = pd.read_csv(args.mix, encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(args.mix.with_name(args.mix.stem + "_key.csv"), encoding="utf-8-sig",
                      keep_default_na=False)
    items = sheet[["item_id", "language", "text"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # not Path.with_suffix: model names contain dots ("deepseek-v4.1-flash")
    stem = args.model.replace("/", "_")
    paths = {kind: args.output_dir / f"{stem}.{kind}" for kind in ("cache.json", "csv", "report.txt")}
    results = run(items, paths["cache.json"], config, args.max_workers)

    pred = pd.DataFrame.from_dict(results, orient="index").rename_axis("item_id").reset_index()
    out = items.merge(pred, on="item_id", how="left")
    out.drop(columns="text").to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    text = report(out.merge(key.drop(columns="language"), on="item_id"), config.labels)
    print(text)
    paths["report.txt"].write_text(f"model={args.model}\n{text}\n", encoding="utf-8")
    print(f"wrote {paths['csv']} and {paths['report.txt']}")


if __name__ == "__main__":
    main()

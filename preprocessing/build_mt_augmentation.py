#!/usr/bin/env python3
"""Machine-translated training speeches for every (language, cluster) cell.

Original-language EP text in a language comes only from that country's MEPs, so it
carries their party mix (original Slovak is 99% Sovereigntist right). The classifier
learned "Slovak that is not EP translationese -> Sov-right", and LLM-written or
LLM-translated Slovak falls on the native side of that line
(analysis/translation_probe.py: Czech/Polish Lib-con speeches machine-translated into
Slovak come out 96% Sov-right). This script puts counterexamples into that region: for
every target language and every cluster it machine-translates --per_cell native speeches
from OTHER languages into the target, with Kimi K3 (reasoning_effort="none").

Every cluster gets the same quota in every language, including the cluster a language
already has natively, so "machine-translated" is not itself a cluster cue.

Sources come from the train split only (the translation is a train row; dev/test stay
untouched). A source is an original-language speech (speaker + date present in <= 2
languages, as in build_language_matched_split.py), 20-600 words. Each source speech is
used for at most one target, and the sources of a cell are spread evenly over the
source languages, so neither one speech nor one source language dominates.

Work is done in rounds of --round_size per cell, so a partial run already has every cell
filled to the same depth. Translations are appended to <out_dir>/translations.jsonl as
they arrive; a rerun skips what is there. --assemble only writes the parquet.

Output: <out_dir>/train_mt.parquet with the split columns (date, EU Party, text,
language, speaker) plus source_language and origin="mt". Run clean-names on it before
training: MT can re-inflect party names that the cleaned source no longer contained.

Usage, from the repository root:
  python -m preprocessing.build_mt_augmentation
  python -m preprocessing.build_mt_augmentation --per_cell 500 --assemble
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import configure_stdout
from utils.api_caller import call_api

configure_stdout()

DEFAULT_TRAIN = "data/EuroParl Custom/clusters_k4_national_ni/train.parquet"
DEFAULT_OUT = "data/EuroParl Custom/clusters_k4_national_ni/mt_augmentation"
PARTY = "EU Party"
NAMES = {"bg": "Bulgarian", "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
         "es": "Spanish", "et": "Estonian", "fi": "Finnish", "fr": "French", "hu": "Hungarian", "it": "Italian",
         "lt": "Lithuanian", "lv": "Latvian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
         "sk": "Slovak", "sl": "Slovenian", "sv": "Swedish"}
PROMPT = ("Translate the following {src} speech from the European Parliament into {tgt}. Keep the meaning, "
          "tone and register; do not summarise, comment or add anything. Output only the translation.\n\n{text}")
MIN_WORDS, MAX_WORDS = 20, 600


def native_keys(train, max_languages):
    keys = pd.read_parquet(train, columns=["date", PARTY, "language", "speaker"])
    n_lang = keys.groupby(["speaker", "date"])["language"].transform("nunique")
    keys = keys[n_lang <= max_languages]
    keys.index.name = "row"
    return keys.reset_index()


def read_texts(train, rows):
    """Texts for the given global row numbers, one row group at a time."""
    rows = np.sort(np.asarray(rows))
    f = pq.ParquetFile(train)
    out, start = {}, 0
    for i in range(f.num_row_groups):
        n = f.metadata.row_group(i).num_rows
        sel = rows[(rows >= start) & (rows < start + n)]
        if len(sel):
            texts = f.read_row_group(i, columns=["text"]).column("text")
            out.update({int(r): texts[int(r - start)].as_py() for r in sel})
        start += n
    return out


def plan(keys, texts, targets, per_cell, seed):
    """[(target, cluster, row)] with every cell filled to per_cell, sources spread over languages."""
    rng = np.random.default_rng(seed)
    keys = keys[keys.row.isin(texts)]
    tasks = []
    for cluster, pool in keys.groupby(PARTY):
        queues = {lang: list(rng.permutation(g.row.to_numpy())) for lang, g in pool.groupby("language")}
        for target in rng.permutation(targets):
            picked, langs = [], [l for l in queues if l != target]
            while len(picked) < per_cell and any(queues[l] for l in langs):
                for lang in rng.permutation(langs):
                    if queues[lang] and len(picked) < per_cell:
                        picked.append(queues[lang].pop())
            if len(picked) < per_cell:
                print(f"  WARNING {target}/{cluster}: only {len(picked)} sources left")
            tasks += [(target, cluster, int(r), depth) for depth, r in enumerate(picked)]
    return tasks


def translate_one(text, src, tgt, model):
    last = ""
    for attempt in range(8):
        try:
            choice = call_api(PROMPT.format(src=NAMES[src], tgt=NAMES[tgt], text=text), model,
                              max_tokens=8000, reasoning_effort="none")["choices"][0]
        except RuntimeError as err:
            last = str(err)[:200]
            time.sleep(min(15 * (attempt + 1), 120) if "429" in last else 5 * (attempt + 1))
            continue
        out = (choice["message"].get("content") or "").strip()
        if out and choice.get("finish_reason") == "stop":
            return out
        last = f"finish_reason={choice.get('finish_reason')}, {len(out)} chars"
    print(f"  failed ({src}>{tgt}): {last}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--targets", default=",".join(NAMES))
    parser.add_argument("--per_cell", type=int, default=500)
    parser.add_argument("--round_size", type=int, default=100)
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--workers", type=int, default=4, help="the API key allows 4 parallel requests")
    parser.add_argument("--original_max_languages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--assemble", action="store_true", help="only write the parquet from what is translated")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl, plan_path = out_dir / "translations.jsonl", out_dir / "plan.parquet"

    if plan_path.exists():
        tasks = pd.read_parquet(plan_path)
        print(f"plan: {plan_path} ({len(tasks)} tasks)")
    else:
        keys = native_keys(args.train, args.original_max_languages)
        # length filter needs the text; read only native rows
        texts = read_texts(args.train, keys.row)
        texts = {r: t for r, t in texts.items() if isinstance(t, str) and MIN_WORDS <= len(t.split()) <= MAX_WORDS}
        print(f"{len(keys)} native train rows, {len(texts)} with {MIN_WORDS}-{MAX_WORDS} words")
        tasks = pd.DataFrame(plan(keys, texts, args.targets.split(","), args.per_cell, args.seed),
                             columns=["target", PARTY, "row", "depth"])
        tasks = tasks.merge(keys, on=["row", PARTY]).rename(columns={"language": "source_language"})
        tasks["text_src"] = tasks.row.map(texts)
        tasks.to_parquet(plan_path, index=False)
        print(f"plan written: {len(tasks)} tasks -> {plan_path}")

    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[(rec["target"], rec["row"])] = rec["text"]

    if not args.assemble:
        todo = tasks[[(t, r) not in done for t, r in zip(tasks.target, tasks.row)]]
        todo = todo.sort_values(["depth", "target", PARTY])  # rounds: every cell advances together
        print(f"{len(done)} done, {len(todo)} to translate with {args.model} ({args.workers} parallel)")
        lock, t0, n_new = threading.Lock(), time.time(), 0
        with open(jsonl, "a", encoding="utf-8") as sink:
            for depth_from in range(0, args.per_cell, args.round_size):
                chunk = todo[(todo.depth >= depth_from) & (todo.depth < depth_from + args.round_size)]
                if chunk.empty:
                    continue
                print(f"round {depth_from}-{depth_from + args.round_size}: {len(chunk)} texts")
                with ThreadPoolExecutor(args.workers) as pool:
                    futures = {pool.submit(translate_one, r.text_src, r.source_language, r.target, args.model): r
                               for r in chunk.itertuples()}
                    for fut in as_completed(futures):
                        r, out = futures[fut], fut.result()
                        if out is None:
                            continue
                        with lock:
                            sink.write(json.dumps({"target": r.target, "row": r.row, "text": out},
                                                  ensure_ascii=False) + "\n")
                            sink.flush()
                            done[(r.target, r.row)] = out
                            n_new += 1
                            if n_new % 200 == 0:
                                rate = n_new / (time.time() - t0) * 3600
                                left = len(tasks) - len(done)
                                print(f"  {len(done)}/{len(tasks)}  {rate:.0f}/h  ~{left / rate:.1f} h left",
                                      flush=True)

    tasks["text"] = [done.get((t, r)) for t, r in zip(tasks.target, tasks.row)]
    out = tasks[tasks.text.notna()]
    mt = pd.DataFrame({"date": out.date, PARTY: out[PARTY], "text": out.text, "language": out.target,
                       "speaker": out.speaker, "source_language": out.source_language, "origin": "mt"})
    mt.to_parquet(out_dir / "train_mt.parquet", index=False)
    print(f"{len(mt)} rows -> {out_dir / 'train_mt.parquet'}")
    print(pd.crosstab(mt.language, mt[PARTY]).to_string())


if __name__ == "__main__":
    main()

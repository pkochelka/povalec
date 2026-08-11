#!/usr/bin/env python3
"""Judge every generated text with the LLM stance judge, one file per model.

Covers both tracks produced for each model in euandi_2024_results:
  * speeches -- the creative-writing opinion pieces (speeches_*.csv, answer_
    columns), melted into (statement, speech) pairs.
  * reasons  -- the survey_processor_concurrent outputs ({langs}{framing}.csv,
    reason_/choice_/original_text_ columns), the short justifications each model
    gave for its own Likert answer, collected into (statement, reason) pairs.

Both are labeled on the 1-5 stance scale with the same judge and prompt as
llm_stance_judge, reading the raw (unscored) files so nothing depends on the NLI
pool. Output is one CSV per model per track (speeches_llm_stance.csv /
reasons_llm_stance.csv), judge-label-only, checkpointed and resumable: re-running
skips finished (model, track) pairs and resumes an interrupted one.
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils import ALL_LANGS_STR, likert_to_stance, load_dataframe, save_checkpoint, configure_stdout
from analysis.sample_speeches_for_labeling import melt_speeches
from analysis.llm_reason_stance_judge import collect_observations
from analysis.llm_stance_judge import judge_one, load_prompt, load_checkpoint

configure_stdout()

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "euandi_2024_results")
DEFAULT_MODELS = [
    "deepseek-v4-pro", "glm-5.2", "kimi-k2.7", "gemini3.5-flash",
    "gemma-4-31b", "gpt-oss-120b", "mistral-medium-3.5", "qwen3.5-122b"
]
# framing name -> filename infix carried by both tracks
FRAMINGS = {"base": "", "negated": "_negated"}
OUTPUT_NAME = {"speeches": "speeches_llm_stance.csv", "reasons": "reasons_llm_stance.csv"}
COLUMNS = ["model", "paraphrase", "language", "variant", "statement",
           "statement_text", "answer_text", "llm_choice", "llm_stance"]


def find_speech_file(model_dir, negated):
    """The raw speeches file for one framing: a speeches_*.csv that is neither
    scored/classified/llm_stance (those are derived) nor the question framing."""
    for path in sorted(glob.glob(os.path.join(model_dir, "speeches_*.csv"))):
        name = os.path.basename(path)
        if any(tag in name for tag in ("_scored", "_classified", "_llm_stance")):
            continue
        if ("_negated" in name) == negated:
            return path
    return None


def load_speech_pool(model):
    frames = []
    model_dir = os.path.join(DATA_DIR, model)
    for framing, suffix in FRAMINGS.items():
        path = find_speech_file(model_dir, suffix == "_negated")
        if not path:
            print(f"  [{model}/{framing} speeches] no raw file, skipping.")
            continue
        frames.append(melt_speeches(load_dataframe(path), model, framing))
    return finalize_pool(frames)


def load_reason_pool(model):
    languages = ALL_LANGS_STR.split(",")
    frames = []
    for framing, suffix in FRAMINGS.items():
        path = os.path.join(DATA_DIR, model, f"{ALL_LANGS_STR}{suffix}.csv")
        if not os.path.exists(path):
            print(f"  [{model}/{framing} reasons] no file at {path}, skipping.")
            continue
        observations = collect_observations(load_dataframe(path), languages, suffix)
        if observations.empty:
            continue
        frames.append(pd.DataFrame({
            "model": model,
            "paraphrase": framing,
            "language": observations["language"].values,
            "variant": observations["variant_idx"].values,
            "statement": observations["statement_idx"].values,
            "statement_text": observations["statement"].astype("string").str.strip().values,
            "answer_text": observations["reason"].astype("string").str.strip().values,
        }))
    return finalize_pool(frames)


def finalize_pool(frames):
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    pool = pd.concat(frames, ignore_index=True)
    pool["answer_text"] = pool["answer_text"].astype("string").str.strip()
    pool = pool.dropna(subset=["answer_text", "statement_text"])
    pool = pool[pool["answer_text"].str.len() > 0]
    # (statement, text): the same short reason under two statements is a distinct
    # judging task, so text alone is the wrong dedup key.
    pool = pool.drop_duplicates(subset=["statement_text", "answer_text"]).reset_index(drop=True)
    return pool


def judge_pool(model, track, prompt_template, args):
    loader = load_speech_pool if track == "speeches" else load_reason_pool
    pool = loader(model)
    output_path = os.path.join(DATA_DIR, model, OUTPUT_NAME[track])
    checkpoint_path = output_path + ".ckpt.json"

    if os.path.exists(output_path) and not os.path.exists(checkpoint_path) and not args.overwrite:
        print(f"[{model}/{track}] already done ({output_path}); skipping. Use --overwrite to redo.")
        return 0
    print(f"[{model}/{track}] {len(pool)} unique texts to judge "
          f"({pool['paraphrase'].value_counts().to_dict() if len(pool) else {}})", flush=True)
    if args.dry_run or len(pool) == 0:
        return len(pool)

    choices = load_checkpoint(checkpoint_path)
    if choices:
        print(f"[{model}/{track}] resuming: {len(choices)}/{len(pool)} already judged", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(judge_one, row["statement_text"], row["answer_text"],
                            args.judge_model, prompt_template): index
            for index, row in pool.iterrows() if index not in choices
        }
        for done, future in enumerate(as_completed(futures), 1):
            choices[futures[future]] = future.result()
            if done % 50 == 0:
                print(f"  [{model}/{track}] {done}/{len(futures)} judged", flush=True)
                save_checkpoint(checkpoint_path, choices)

    pool["llm_choice"] = pool.index.map(choices)
    labeled = pool.dropna(subset=["llm_choice"]).copy()
    labeled["llm_choice"] = labeled["llm_choice"].astype(int)
    labeled["llm_stance"] = likert_to_stance(labeled["llm_choice"]).astype("float32")
    labeled[COLUMNS].to_csv(output_path, sep=";", encoding="utf-8-sig", index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    failed = len(pool) - len(labeled)
    print(f"[{model}/{track}] wrote {len(labeled)}/{len(pool)} ({failed} failed) -> {output_path}")
    print(f"  choice distribution: {labeled['llm_choice'].value_counts().sort_index().to_dict()}",
          flush=True)
    return len(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge_model", default="deepseek-v4-pro")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model dirs to judge.")
    parser.add_argument("--tracks", default="speeches,reasons",
                        help="Comma-separated tracks to judge: speeches, reasons.")
    parser.add_argument("--max_workers", default=20, type=int)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-judge (model, track) pairs whose output already exists.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report per-model/-track counts and the total; make no API calls.")
    args = parser.parse_args()

    prompt_template = load_prompt()
    models = args.models.split(",")
    tracks = args.tracks.split(",")
    total = 0
    for model in models:
        for track in tracks:
            total += judge_pool(model, track, prompt_template, args)
    if args.dry_run:
        print(f"\n[dry run] {total} texts across {len(models)} models x {tracks}; "
              f"no API calls issued.")


if __name__ == "__main__":
    main()

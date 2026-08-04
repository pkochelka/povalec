import argparse
import subprocess
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR, VARIANTS

LANGUAGES = ALL_LANGS_STR
MAX_RETRIES = 1
MAX_WORKERS = 40
# Per-key worker count for the second provider. The actual pool fans out to
# this many workers for *each* configured API key (KEY1, KEY2, ... in .env.local),
# so total concurrency is MAX_WORKERS_SECOND_PROVIDER * number_of_keys.
MAX_WORKERS_SECOND_PROVIDER = 4
DATASETS = [#"euandi_2019",
    "euandi_2024"]

MODELS = [
    #{"model": "gemma4:12b",      "model_dir":"gemma-4-12b", "second_provider": False},
    #{"model": "google/gemini-3.5-flash",      "model_dir":"gemini3.5-flash", "second_provider": False},    
    #{"model": "x-ai/grok-4.5",      "model_dir":"grok-4.5", "second_provider": False},    
    #{"model": "openai/gpt-5.6-luna", "model_dir": "gpt-5.6-luna", "second_provider": False},
    {"model": "kimi-k3",      "model_dir":"kimi-k3", "second_provider": True},
    #{"model": "gemma4",      "model_dir":"gemma-4-31b", "second_provider": True},
    #{"model": "phi4:14b-q8_0",      "model_dir":"phi-4-14b", "second_provider": True},
    #{"model": "gemma4:31b-it-q8_0",      "model_dir":"gemma-4-31b-it", "second_provider": False},
    #{"model": "deepseek/deepseek-v4-pro",      "model_dir":"deepseek-v4-pro", "second_provider": False},
    #{"model": "mistralai/mistral-small-2603",      "model_dir":"mistral-small-2603", "second_provider": False},
    #{"model": "x-ai/grok-4.3",      "model_dir":"grok-4.3", "second_provider": False},

    #{"model": "deepseek-v4-pro-thinking", "model_dir": "deepseek-v4-pro", "second_provider": True},
    #{"model": "glm-5.2", "model_dir": "glm-5.2", "second_provider": True},
    #{"model": "mistral-medium-3.5", "model_dir": "mistral-medium-3.5", "second_provider": True},
    #{"model": "command-a", "model_dir": "command-a", "second_provider": True},
    #{"model": "qwen3.5-122b",              "model_dir": "qwen3.5-122b",  "second_provider": True},
    #{"model": "kimi-k2.7", "model_dir": "kimi-k2.7", "second_provider": True},
    #{"model": "gpt-oss-120b", "model_dir": "gpt-oss-120b", "second_provider": True},
]

_DIR = os.path.dirname(os.path.abspath(__file__))

def build_cmd(script, model, model_dir, variant, dataset, second_provider, patch,
              overwrite_changed=False, dry_run=False):
    cmd = [sys.executable, os.path.join(_DIR, script),
           "--model", model, "--model_dir", model_dir,
           "--variant", variant, "--dataset", dataset]
    if script == "survey_processor_concurrent.py":
        cmd += ["--languages", LANGUAGES]
    if second_provider:
        cmd.append("--second_provider")
        cmd += ["--max_workers", str(MAX_WORKERS_SECOND_PROVIDER)]
    else:
        cmd += ["--max_workers", str(MAX_WORKERS)]
    if patch:
        cmd.append("--patch")
    if overwrite_changed:
        cmd.append("--overwrite_changed")
    if dry_run:
        cmd.append("--dry_run")
    return cmd

SCRIPTS = ["speech_generator.py", "survey_processor_concurrent.py"]

def run_model(cfg, patch, overwrite_changed=False, dry_run=False):
    model, model_dir, second_provider = cfg["model"], cfg["model_dir"], cfg["second_provider"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for script in SCRIPTS:
                label = f"{script}: model={model}, variant='{variant}', dataset='{dataset}'"
                print(f"Running: {label}", flush=True)

                for attempt in range(1, MAX_RETRIES + 1):
                    result = subprocess.run(build_cmd(script, model, model_dir, variant, dataset,
                                                      second_provider, patch, overwrite_changed, dry_run))

                    if result.returncode == 0:
                        print(f"✓  Succeeded: {label}", flush=True)
                        break
                    else:
                        print(f"✗ Attempt {attempt}/{MAX_RETRIES} failed: {label}", flush=True)
                        if attempt < MAX_RETRIES:
                            time.sleep(2)
                else:
                    print(f"✗ All retries exhausted for: {label}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Patch only failed/refused responses in existing outputs; forwarded as --patch to both generator scripts.",
    )
    parser.add_argument(
        "--overwrite_changed",
        action="store_true",
        help="Overwrite in place only the rows whose statement text was rewritten; forwarded to both generator scripts.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="With --overwrite_changed, only report what each model would regenerate; make no API calls.",
    )
    parser.add_argument(
        "--sequential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run models one after another (default). Use --no-sequential to run them concurrently in threads.",
    )
    args = parser.parse_args()

    if args.sequential:
        for cfg in MODELS:
            run_model(cfg, args.patch, args.overwrite_changed, args.dry_run)
    else:
        threads = [threading.Thread(target=run_model, args=(cfg, args.patch, args.overwrite_changed, args.dry_run),
                                    name=cfg["model_dir"]) for cfg in MODELS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR

VARIANTS = ["", "_question", "_negated"]
LANGUAGES = ALL_LANGS_STR
MAX_RETRIES = 5
MAX_WORKERS = 3
MAX_WORKERS_SECOND_PROVIDER = 4
DATASETS = [#"euandi_2019",
    "euandi_2024"]

MODELS = [
    {"model": "gemma4:12b",      "model_dir":"gemma-4-12b", "second_provider": False},
    #{"model": "gemma4:31b-it-q8_0",      "model_dir":"gemma-4-31b-it", "second_provider": False},
    #{"model": "deepseek/deepseek-v4-pro",      "model_dir":"deepseek-v4-pro", "second_provider": False},
    #{"model": "mistralai/mistral-small-2603",      "model_dir":"mistral-small-2603", "second_provider": False},
    #{"model": "x-ai/grok-4.3",      "model_dir":"grok-4.3", "second_provider": False},

#    {"model": "qwen3.5-122b",              "model_dir": "qwen3.5-122b",  "second_provider": False},
    #{"model": "gpt-oss-120b", "model_dir": "gpt-oss-120b", "second_provider": True},
    #{"model": "mistral-medium-3.5", "model_dir": "mistral-medium-3.5", "second_provider": True},
    #{"model": "kimi-k2.6", "model_dir": "kimi-k2.6", "second_provider": True},
    #{"model": "deepseek-v4-pro-thinking", "model_dir": "deepseek-v4-pro", "second_provider": True},
    #{"model": "glm-5", "model_dir": "glm-5", "second_provider": True},
]

_DIR = os.path.dirname(os.path.abspath(__file__))

def build_cmd(script, model, model_dir, variant, dataset, second_provider, patch):
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
    return cmd

SCRIPTS = ["speech_generator.py", "survey_processor_concurrent.py"]

def run_model(cfg, patch):
    model, model_dir, second_provider = cfg["model"], cfg["model_dir"], cfg["second_provider"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for script in SCRIPTS:
                label = f"{script}: model={model}, variant='{variant}', dataset='{dataset}'"
                print(f"Running: {label}", flush=True)

                for attempt in range(1, MAX_RETRIES + 1):
                    result = subprocess.run(build_cmd(script, model, model_dir, variant, dataset, second_provider, patch))

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
    args = parser.parse_args()

    threads = [threading.Thread(target=run_model, args=(cfg, args.patch), name=cfg["model_dir"]) for cfg in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()

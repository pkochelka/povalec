import subprocess
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statement_collection.scrape_euandi import LANGS

VARIANTS = ["", "_question", "_negated"]
LANGUAGES = ",".join(LANGS+["en"])
MAX_RETRIES = 3
DATASETS = [#"euandi_2019",
    "euandi_2024"]

MODELS = [
    {"model": "google/gemma-4-31b-it",      "model_dir":"gemma-4-31b-it", "second_provider": False},
    {"model": "deepseek/deepseek-v4-pro",      "model_dir":"deepseek-v4-pro", "second_provider": False},
#    {"model": "qwen3.5-122b",              "model_dir": "qwen3.5-122b",  "second_provider": False},
#    {"model": "gpt-oss-120b", "model_dir": "gpt-oss-120b", "second_provider": False},
]

_DIR = os.path.dirname(os.path.abspath(__file__))

def build_cmd(script, model, model_dir, variant, dataset, second_provider):
    cmd = [sys.executable, os.path.join(_DIR, script),
           "--model", model, "--model_dir", model_dir,
           "--variant", variant, "--dataset", dataset]
    if script == "survey_processor_concurrent.py":
        cmd += ["--languages", LANGUAGES]
    if second_provider:
        cmd.append("--second_provider")
    return cmd

SCRIPTS = ["speech_generator.py", "survey_processor_concurrent.py"]

def run_model(cfg):
    model, model_dir, second_provider = cfg["model"], cfg["model_dir"], cfg["second_provider"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for script in SCRIPTS:
                label = f"{script}: model={model}, variant='{variant}', dataset='{dataset}'"
                print(f"Running: {label}", flush=True)

                for attempt in range(1, MAX_RETRIES + 1):
                    result = subprocess.run(build_cmd(script, model, model_dir, variant, dataset, second_provider))

                    if result.returncode == 0:
                        print(f"✓  Succeeded: {label}", flush=True)
                        break
                    else:
                        print(f"✗ Attempt {attempt}/{MAX_RETRIES} failed: {label}", flush=True)
                        if attempt < MAX_RETRIES:
                            time.sleep(2)
                else:
                    print(f"✗ All retries exhausted for: {label}", flush=True)

threads = [threading.Thread(target=run_model, args=(cfg,), name=cfg["model_dir"]) for cfg in MODELS]
for t in threads:
    t.start()
for t in threads:
    t.join()

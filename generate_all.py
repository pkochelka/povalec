import subprocess
import sys
import time

from scrape_euandi import LANGS

MODELS = ["gpt-oss-120b", "qwen3.5-122b"]
VARIANTS = ["", "_question", "_negated"]
LANGUAGES = ",".join(LANGS)
MAX_RETRIES = 3
DATASETS = [#"euandi_2019", 
    "euandi_2024"]

SCRIPTS = [
    #("speech_generator.py", lambda model, variant, dataset: [
    #    sys.executable, "speech_generator.py",
    #    "--model", model, "--variant", variant, "--dataset", dataset,
    #]),
    ("survey_processor_concurrent.py", lambda model, variant, dataset: [
        sys.executable, "survey_processor_concurrent.py",
        "--model", model, "--variant", variant, "--dataset", dataset,
        "--languages", LANGUAGES,
    ]),
]

for dataset in DATASETS:
    for model in MODELS:
        for variant in VARIANTS:
            for script_name, build_cmd in SCRIPTS:
                label = f"{script_name}: model={model}, variant='{variant}', dataset='{dataset}'"
                print(f"Running: {label}")

                for attempt in range(1, MAX_RETRIES + 1):
                    result = subprocess.run(build_cmd(model, variant, dataset))

                    if result.returncode == 0:
                        print(f"✓ Succeeded: {label}")
                        break
                    else:
                        print(f"✗ Attempt {attempt}/{MAX_RETRIES} failed: {label}")
                        if attempt < MAX_RETRIES:
                            time.sleep(2)
                else:
                    print(f"✗ All retries exhausted for: {label}")

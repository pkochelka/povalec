import argparse
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ALL_LANGS_STR

VARIANTS = ["", "_negated"]
LANGUAGES = ALL_LANGS_STR
MAX_RETRIES = 1
DATASETS = [#"euandi_2019",
    "euandi_2024"]

MODEL_DIRS = [
    #"gemma-4-12b",
    "gemma-4-31b",
    "deepseek-v4-pro",
    "qwen3.5-122b",
    "gpt-oss-120b",
    "kimi-k2.7",
    "mistral-medium-3.5",
    "glm-5.2"
]

_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTTING_DIR = os.path.join(_DIR, "plotting")

def _results_dir(dataset, model_dir):
    return os.path.join("data", f"{dataset}_results", model_dir)


PER_VARIANT_SCRIPTS = [
    #("agreement_scoring.py", _DIR, "--llm", [],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"speeches_{LANGUAGES}{variant}_scored.csv")),
    #("evaluate_cronbach_speeches.py", _DIR, "--model_dir", [],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"cronbach_speeches{variant}_{LANGUAGES}.csv")),
    #("evaluate_euandi.py", _DIR, "--model_dir", ["--collapse-ecr-id"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"vaa{variant}_{LANGUAGES}.csv")),
    #("evaluate_euandi.py", _DIR, "--model_dir", ["--source", "speeches", "--collapse-ecr-id"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"vaa_speeches{variant}_{LANGUAGES}.csv")),
    #("evaluate_cronbach.py", _DIR, "--model_dir", [],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"cronbach{variant}_{LANGUAGES}.csv")),
    ("classify_speeches.py", _DIR, "--llm",
     ["--source", "speeches", "--model_dir", "./mmBERT-base-multilabel-collapsed"],
     lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"speeches_{LANGUAGES}{variant}_classified.csv")),
    ("classify_speeches.py", _DIR, "--llm",
     ["--source", "reasons", "--model_dir", "./mmBERT-base-multilabel-collapsed"],
     lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"{LANGUAGES}{variant}_classified.csv")),
]

PER_DATASET_PLOTTING_SCRIPTS = [
    #("plot_vaa_per_language.py",      _PLOTTING_DIR),
    #("plot_vaa_variants_grouped.py",  _PLOTTING_DIR),
    #("plot_vaa_speeches.py",          _PLOTTING_DIR),
    #("plot_consistency.py",           _PLOTTING_DIR),
    #("plot_cronbach.py",              _PLOTTING_DIR),
    #("plot_speeches.py",              _PLOTTING_DIR),
    ("plot_political_bias.py",        _PLOTTING_DIR),
    ("plot_classified_parties.py",    _PLOTTING_DIR),
    #("plot_likert_distributions.py",  _PLOTTING_DIR),
]


def build_per_variant_cmd(script, script_dir, model_arg_name, extra_args, model_dir, variant, dataset, override):
    cmd = [
        sys.executable, os.path.join(script_dir, script),
        model_arg_name, model_dir,
        "--variant", variant,
        "--dataset", dataset,
        "--languages", LANGUAGES,
        *extra_args,
    ]
    if override:
        cmd.append("--override")
    return cmd


def build_plotting_cmd(script, script_dir, dataset):
    return [
        sys.executable, os.path.join(script_dir, script),
        "--dataset", dataset,
    ]


def run_with_retries(cmd, label):
    for attempt in range(1, MAX_RETRIES + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✓  Succeeded: {label}", flush=True)
            return
        print(f"✗ Attempt {attempt}/{MAX_RETRIES} failed: {label}", flush=True)
        if attempt < MAX_RETRIES:
            time.sleep(2)
    print(f"✗ All retries exhausted for: {label}", flush=True)


def should_skip(output_path, dataset, model_dir, variant, override):
    return not override and os.path.exists(output_path(dataset, model_dir, variant))


def run_per_variant_scripts(override):
    for dataset in DATASETS:
        for model_dir in MODEL_DIRS:
            for variant in VARIANTS:
                for script, script_dir, model_arg_name, extra_args, output_path in PER_VARIANT_SCRIPTS:
                    label = f"{script} {' '.join(extra_args)}: model={model_dir}, variant='{variant}', dataset='{dataset}'"
                    if should_skip(output_path, dataset, model_dir, variant, override):
                        print(f"Skipping (output exists): {label}", flush=True)
                        continue
                    print(f"Running: {label}", flush=True)
                    cmd = build_per_variant_cmd(script, script_dir, model_arg_name, extra_args, model_dir, variant, dataset, override)
                    run_with_retries(cmd, label)


def run_plotting_scripts():
    for dataset in DATASETS:
        for script, script_dir in PER_DATASET_PLOTTING_SCRIPTS:
            label = f"{script}: dataset='{dataset}'"
            print(f"Running: {label}", flush=True)
            cmd = build_plotting_cmd(script, script_dir, dataset)
            run_with_retries(cmd, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--override",
        action="store_true",
        help="Recompute and overwrite existing outputs; forwarded as --override to the non-plotting scripts.",
    )
    args = parser.parse_args()

    run_per_variant_scripts(args.override)
    run_plotting_scripts()


if __name__ == "__main__":
    main()
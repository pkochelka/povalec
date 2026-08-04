import argparse
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The progress lines carry check/cross marks; without this a redirected stdout falls
# back to the console codepage on Windows and the run dies on the first success.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import ALL_LANGS_STR, VARIANTS

LANGUAGES = ALL_LANGS_STR
MAX_RETRIES = 1
# Whose euandi answers stand for an EP group. evaluate_euandi.py writes vaa*.csv without
# the basis in the filename and the plotting scripts read those back, so every step below
# has to be run on one basis -- change it here (or pass --positions) and rerun with
# --override, otherwise the existing vaa*.csv files are left in place and the plots keep
# reading the old basis. The script lists carry a placeholder that main() fills in.
POSITIONS = "ep-group"
POSITIONS_PLACEHOLDER = "@positions@"
TABLES_PLACEHOLDER = "@tables@"


def _tables_dir(dataset):
    return os.path.join("data", f"{dataset}_results", "tables")


def resolve_args(extra_args, positions, dataset):
    """Fill the placeholders the script lists carry: the positions basis and the
    dataset's tables directory."""
    return [
        arg.replace(POSITIONS_PLACEHOLDER, positions)
           .replace(TABLES_PLACEHOLDER, _tables_dir(dataset))
        for arg in extra_args
    ]
DATASETS = [#"euandi_2019",
    "euandi_2024"]

MODEL_DIRS = [
    #"gemma-4-12b",
    #"gemini3.5-flash",
    #"grok-4.5",
    #"gpt-5.6-luna",
    #"granite-4.1-8b",
    #"gemma-4-31b",
    #"deepseek-v4-pro",
    #"qwen3.5-122b",
    #"gpt-oss-120b",
    #"kimi-k2.7",
    #"mistral-medium-3.5",
    #"glm-5.2",
    "kimi-k3"
]

_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTTING_DIR = os.path.join(_DIR, "plotting")

def _results_dir(dataset, model_dir):
    return os.path.join("data", f"{dataset}_results", model_dir)


PER_VARIANT_SCRIPTS = [
    #("agreement_scoring.py", _DIR, "--llm", ["--source", "speeches"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"speeches_{LANGUAGES}{variant}_scored.csv")),
    #("agreement_scoring.py", _DIR, "--llm", ["--source", "reasons"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"{LANGUAGES}{variant}_scored.csv")),
    #("evaluate_cronbach_speeches.py", _DIR, "--model_dir", [],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"cronbach_speeches{variant}_{LANGUAGES}.csv")),
    ("evaluate_euandi.py", _DIR, "--model_dir", ["--collapse-ecr-id", "--positions", POSITIONS_PLACEHOLDER],
     lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"vaa{variant}_{LANGUAGES}.csv")),
    ("evaluate_euandi.py", _DIR, "--model_dir", ["--source", "speeches", "--collapse-ecr-id", "--positions", POSITIONS_PLACEHOLDER],
     lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"vaa_speeches{variant}_{LANGUAGES}.csv")),
    #("evaluate_cronbach.py", _DIR, "--model_dir", [],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"cronbach{variant}_{LANGUAGES}.csv")),
    #("classify_speeches.py", _DIR, "--llm",
    # ["--source", "speeches", "--model_dir", "./mmBERT-base-balanced-collapsed"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"speeches_{LANGUAGES}{variant}_classified.csv")),
    #("classify_speeches.py", _DIR, "--llm",
    # ["--source", "reasons", "--model_dir", "./mmBERT-base-balanced-collapsed"],
    # lambda dataset, model_dir, variant: os.path.join(_results_dir(dataset, model_dir), f"{LANGUAGES}{variant}_classified.csv")),
]

# (script, directory, extra args). Scripts that read the euandi positions take
# --positions and must be given the same basis the vaa*.csv files were written with.
_POSITIONS_ARGS = ["--positions", POSITIONS_PLACEHOLDER]

PER_DATASET_PLOTTING_SCRIPTS = [
    #("plot_vaa_per_language.py",      _PLOTTING_DIR, []),
    ("plot_vaa_variants_grouped.py",  _PLOTTING_DIR, []),
    ("plot_vaa_speeches.py",          _PLOTTING_DIR, []),
    #("plot_consistency.py",           _PLOTTING_DIR, []),
    #("plot_speeches.py",              _PLOTTING_DIR, []),
    # plot_political_bias.py spells the same choice --parties (it also accepts 'none').
    ("plot_political_bias.py",        _PLOTTING_DIR, ["--parties", POSITIONS_PLACEHOLDER]),
    ("plot_classified_parties.py",    _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_topical_parties.py",       _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_argmax_shares.py",         _PLOTTING_DIR, _POSITIONS_ARGS),
    #("plot_party_axis_compass.py",    _PLOTTING_DIR, _POSITIONS_ARGS),
    #("plot_likert_distributions.py",  _PLOTTING_DIR, []),
]

# The table builders. They read the same vaa*.csv / scored CSVs as the plots, so they
# belong to the same basis; compare_position_bases.py is the exception -- it walks the
# bases itself and takes --bases instead of --positions.
_MODELS_ARG = ["--models", ",".join(MODEL_DIRS)]

PER_DATASET_TABLE_SCRIPTS = [
    #("rank_consistency_tables.py", _DIR, _POSITIONS_ARGS),
    #("axis_position_tables.py", _DIR, _POSITIONS_ARGS),
    #("vaa_agreement_ci.py", _DIR,
    # [*_POSITIONS_ARGS, *_MODELS_ARG, "--collapse-ecr-id",
    #  "--languages", LANGUAGES,
    #  "--output", os.path.join(TABLES_PLACEHOLDER, "vaa_agreement_ci.md")]),
    #("compare_position_bases.py", _DIR,
    # [*_MODELS_ARG, "--collapse-ecr-id", "--languages", LANGUAGES,
    #  "--output", os.path.join(TABLES_PLACEHOLDER, "position_bases_comparison.md")]),
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


def build_plotting_cmd(script, script_dir, dataset, extra_args):
    return [
        sys.executable, os.path.join(script_dir, script),
        "--dataset", dataset,
        *extra_args,
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


def selected(script, only):
    return not only or any(fragment in script for fragment in only)


def run_per_variant_scripts(override, only, positions):
    for dataset in DATASETS:
        for model_dir in MODEL_DIRS:
            for variant in VARIANTS:
                for script, script_dir, model_arg_name, raw_args, output_path in PER_VARIANT_SCRIPTS:
                    if not selected(script, only):
                        continue
                    extra_args = resolve_args(raw_args, positions, dataset)
                    label = f"{script} {' '.join(extra_args)}: model={model_dir}, variant='{variant}', dataset='{dataset}'"
                    if should_skip(output_path, dataset, model_dir, variant, override):
                        print(f"Skipping (output exists): {label}", flush=True)
                        continue
                    print(f"Running: {label}", flush=True)
                    cmd = build_per_variant_cmd(script, script_dir, model_arg_name, extra_args, model_dir, variant, dataset, override)
                    run_with_retries(cmd, label)


def run_per_dataset_scripts(scripts, only, positions):
    """The plotting and table builders: one run per dataset, no per-model loop."""
    for dataset in DATASETS:
        os.makedirs(_tables_dir(dataset), exist_ok=True)
        for script, script_dir, raw_args in scripts:
            if not selected(script, only):
                continue
            extra_args = resolve_args(raw_args, positions, dataset)
            label = f"{script} {' '.join(extra_args)}: dataset='{dataset}'"
            print(f"Running: {label}", flush=True)
            cmd = build_plotting_cmd(script, script_dir, dataset, extra_args)
            run_with_retries(cmd, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--override",
        action="store_true",
        help="Recompute and overwrite existing outputs; forwarded as --override to the non-plotting scripts.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated script-name fragments; run only the matching scripts "
             "(e.g. 'evaluate_euandi' to redo the VAA CSVs without re-scoring). "
             "Default: every enabled script.",
    )
    parser.add_argument(
        "--positions",
        default=POSITIONS,
        choices=["ep-group", "national", "group-mean"],
        help=f"Whose euandi answers stand for an EP group, for evaluate_euandi.py and "
             f"every plotting script that reads the positions back. Default: {POSITIONS}. "
             f"Changing it needs --override, since the vaa*.csv filenames do not carry "
             f"the basis.",
    )
    args = parser.parse_args()
    only = [fragment for fragment in (args.only or "").split(",") if fragment]
    print(f"EP group positions basis: {args.positions}\n", flush=True)

    run_per_variant_scripts(args.override, only, args.positions)
    run_per_dataset_scripts(PER_DATASET_PLOTTING_SCRIPTS, only, args.positions)
    run_per_dataset_scripts(PER_DATASET_TABLE_SCRIPTS, only, args.positions)


if __name__ == "__main__":
    main()
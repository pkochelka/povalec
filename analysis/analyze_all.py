import argparse
import json
import subprocess
import sys
import os
import time

from utils import ALL_LANGS_STR, VARIANTS, configure_stdout
from analysis.core import classified_name, responses_name, scored_name, speeches_name, vaa_name
from analysis.core.positions import POSITION_CHOICES, load_party_positions, positions_path

configure_stdout()

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
CLASSIFIER_PLACEHOLDER = "@classifier@"


def _tables_dir(dataset):
    return os.path.join("data", f"{dataset}_results", "tables")


def resolve_args(extra_args, positions, dataset, classifier=None):
    """Fill the placeholders the script lists carry: the positions basis, the dataset's
    tables directory and the classifier checkpoint."""
    return [
        arg.replace(POSITIONS_PLACEHOLDER, positions)
           .replace(TABLES_PLACEHOLDER, _tables_dir(dataset))
           .replace(CLASSIFIER_PLACEHOLDER, classifier or CLASSIFIER_DIR)
        for arg in extra_args
    ]
DATASETS = ["euandi_2024"]

MODEL_DIRS = [
    "gemma-4-12b",
    "gemini3.5-flash",
    "grok-4.5",
    "gpt-5.6-luna",
    "granite-4.1-8b",
    "gemma-4-31b",
    "deepseek-v4-pro",
    "qwen3.5-122b",
    "gpt-oss-120b",
    "kimi-k2.7",
    "mistral-medium-3.5",
    "glm-5.2",
    "kimi-k3"
]

_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTTING_DIR = os.path.join(_DIR, "plotting")

def _results_dir(dataset, model_dir):
    return os.path.join("data", f"{dataset}_results", model_dir)


def _result(dataset, model_dir, filename):
    return os.path.join(_results_dir(dataset, model_dir), filename)


# Checkpoints are inputs too: retraining one has to invalidate everything scored with it.
# That is the case this staleness check exists for -- the classifier was retrained and the
# *_classified.csv files from the previous checkpoint were left in place for a week.
CROSSENCODER_DIR = "mmbert-small-stance-crossencoder"
# The default EP-group (ECR+ID) classifier; --classifier swaps in another checkpoint,
# e.g. a cluster-track one (runs/<track>-k4-<trainer>/model) for --positions cluster.
CLASSIFIER_DIR = "mmBERT-base-balanced-collapsed"


# Each entry is (script, directory, model-arg name, extra args, output, inputs).
#
# `output` and `inputs` are what should_skip() compares: a step re-runs when its output is
# missing OR older than anything it derives from. Both are built with the shared filename
# builders in analysis.core.paths, the same ones the scripts themselves write through, so
# the driver cannot predict a path the child does not produce.
PER_VARIANT_SCRIPTS = [
    ("agreement_scoring.py", _DIR, "--llm", ["--source", "speeches"],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, scored_name(LANGUAGES, variant, "speeches")),
     lambda dataset, model_dir, variant, positions: [_result(dataset, model_dir, speeches_name(LANGUAGES, variant)), CROSSENCODER_DIR]),
    ("agreement_scoring.py", _DIR, "--llm", ["--source", "reasons"],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, scored_name(LANGUAGES, variant, "reasons")),
     lambda dataset, model_dir, variant, positions: [_result(dataset, model_dir, responses_name(LANGUAGES, variant)), CROSSENCODER_DIR]),
    ("evaluate_euandi.py", _DIR, "--model_dir", ["--collapse-ecr-id", "--positions", POSITIONS_PLACEHOLDER],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, vaa_name(LANGUAGES, variant, "reasons")),
     lambda dataset, model_dir, variant, positions: [
         _result(dataset, model_dir, responses_name(LANGUAGES, variant)), positions_path(positions)]),
    ("evaluate_euandi.py", _DIR, "--model_dir", ["--source", "speeches", "--collapse-ecr-id", "--positions", POSITIONS_PLACEHOLDER],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, vaa_name(LANGUAGES, variant, "speeches")),
     lambda dataset, model_dir, variant, positions: [
         _result(dataset, model_dir, scored_name(LANGUAGES, variant, "speeches")), positions_path(positions)]),
    ("classify_speeches.py", _DIR, "--llm",
     ["--source", "speeches", "--model_dir", CLASSIFIER_PLACEHOLDER],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, classified_name(LANGUAGES, variant, "speeches")),
     lambda dataset, model_dir, variant, positions: [_result(dataset, model_dir, speeches_name(LANGUAGES, variant)), CLASSIFIER_PLACEHOLDER]),
    ("classify_speeches.py", _DIR, "--llm",
     ["--source", "reasons", "--model_dir", CLASSIFIER_PLACEHOLDER],
     lambda dataset, model_dir, variant: _result(dataset, model_dir, classified_name(LANGUAGES, variant, "reasons")),
     lambda dataset, model_dir, variant, positions: [_result(dataset, model_dir, responses_name(LANGUAGES, variant)), CLASSIFIER_PLACEHOLDER]),
]

# (script, directory, extra args). Scripts that read the euandi positions take
# --positions and must be given the same basis the vaa*.csv files were written with.
_POSITIONS_ARGS = ["--positions", POSITIONS_PLACEHOLDER]

PER_DATASET_PLOTTING_SCRIPTS = [
    ("plot_vaa_per_language.py",      _PLOTTING_DIR, []),
    ("plot_vaa_variants_grouped.py",  _PLOTTING_DIR, []),
    ("plot_vaa_speeches.py",          _PLOTTING_DIR, []),
    ("plot_speeches.py",              _PLOTTING_DIR, []),
    # plot_political_bias.py spells the same choice --parties (it also accepts 'none').
    ("plot_political_bias.py",        _PLOTTING_DIR, ["--parties", POSITIONS_PLACEHOLDER]),
    ("plot_classified_parties.py",    _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_topical_parties.py",       _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_argmax_shares.py",         _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_party_axis_compass.py",    _PLOTTING_DIR, _POSITIONS_ARGS),
    ("plot_likert_distributions.py",  _PLOTTING_DIR, []),
]

# The table builders. They read the same vaa*.csv / scored CSVs as the plots, so they
# belong to the same basis; compare_position_bases.py is the exception -- it walks the
# bases itself and takes --bases instead of --positions.
_MODELS_ARG = ["--models", ",".join(MODEL_DIRS)]

PER_DATASET_TABLE_SCRIPTS = [
    ("rank_consistency_tables.py", _DIR, _POSITIONS_ARGS),
    ("axis_position_tables.py", _DIR, _POSITIONS_ARGS),
    ("vaa_agreement_ci.py", _DIR,
     [*_POSITIONS_ARGS, *_MODELS_ARG, "--collapse-ecr-id",
      "--languages", LANGUAGES,
      "--output", os.path.join(TABLES_PLACEHOLDER, "vaa_agreement_ci.md")]),
    ("compare_position_bases.py", _DIR,
     [*_MODELS_ARG, "--collapse-ecr-id", "--languages", LANGUAGES,
      "--output", os.path.join(TABLES_PLACEHOLDER, "position_bases_comparison.md")]),    # Reads plots/argmax_share_methods.csv, which plot_argmax_shares.py writes above.
    ("argmax_share_table.py", os.path.join(_DIR, "tables"),
     ["--output", os.path.join(TABLES_PLACEHOLDER, "argmax_share_table.tex")]),
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


def newest_mtime(path):
    """Modification time of a file, or of the newest file directly inside a directory.

    Checkpoints are directories, and a retrained one keeps its directory mtime from
    whenever it was created on some systems, so the files inside are what count.
    """
    if os.path.isdir(path):
        times = [entry.stat().st_mtime for entry in os.scandir(path) if entry.is_file()]
        return max(times) if times else None
    return os.path.getmtime(path) if os.path.exists(path) else None


def skip_reason(output_path, inputs, dataset, model_dir, variant, positions, classifier, override):
    """Why this step can be skipped, or None if it must run.

    Skipping used to be `os.path.exists(output)`, which never re-derived anything: a
    regenerated CSV or a retrained checkpoint left every downstream artefact in place,
    silently and indefinitely. A step now also has to be *newer* than its inputs.
    """
    if override:
        return None
    out = output_path(dataset, model_dir, variant)
    out_mtime = newest_mtime(out)
    if out_mtime is None:
        return None

    stale = []
    for source in resolve_args(inputs(dataset, model_dir, variant, positions),
                               positions, dataset, classifier):
        source_mtime = newest_mtime(source)
        if source_mtime is None:
            continue          # a missing input is the step's own problem to report
        if source_mtime > out_mtime:
            stale.append(os.path.basename(source.rstrip(os.sep)) or source)
    if stale:
        print(f"  Out of date -- newer input(s): {', '.join(stale)}", flush=True)
        return None
    return "output exists and is newer than its inputs"


def selected(script, only):
    return not only or any(fragment in script for fragment in only)


def run_per_variant_scripts(override, only, positions, classifier):
    for dataset in DATASETS:
        for model_dir in MODEL_DIRS:
            for variant in VARIANTS:
                for script, script_dir, model_arg_name, raw_args, output_path, inputs in PER_VARIANT_SCRIPTS:
                    if not selected(script, only):
                        continue
                    extra_args = resolve_args(raw_args, positions, dataset, classifier)
                    label = f"{script} {' '.join(extra_args)}: model={model_dir}, variant='{variant}', dataset='{dataset}'"
                    reason = skip_reason(output_path, inputs, dataset, model_dir, variant,
                                         positions, classifier, override)
                    if reason:
                        print(f"Skipping ({reason}): {label}", flush=True)
                        continue
                    # The child scripts carry the same existence check, so a rerun of a
                    # stale output has to tell them to overwrite it or they skip in turn.
                    replacing = os.path.exists(output_path(dataset, model_dir, variant))
                    print(f"Running: {label}", flush=True)
                    cmd = build_per_variant_cmd(script, script_dir, model_arg_name, extra_args,
                                                model_dir, variant, dataset, override or replacing)
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
            if not os.path.exists(os.path.join(script_dir, script)):
                print(f"Skipping (script not in this checkout): {label}", flush=True)
                continue
            print(f"Running: {label}", flush=True)
            cmd = build_plotting_cmd(script, script_dir, dataset, extra_args)
            run_with_retries(cmd, label)


def check_classifier_labels(classifier, positions):
    """Fail early when the classifier predicts a different set of groups than the
    positions basis scores against -- a cluster checkpoint with EP-group positions, or
    the reverse. Nothing downstream would raise: the tables and figures would just
    compare the two methods over their (empty or partial) intersection."""
    with open(os.path.join(classifier, "manifest.json"), encoding="utf-8") as f:
        labels = set(json.load(f)["label2id"])
    groups = load_party_positions(positions_path(positions), positions)["ep_group"]
    groups = set(groups.dropna().replace({"ECR": "ECR+ID", "ID": "ECR+ID"}))
    if labels != groups:
        raise SystemExit(
            f"classifier {classifier} predicts {sorted(labels)}, but the {positions!r} "
            f"positions basis has groups {sorted(groups)}. Pair a cluster checkpoint with "
            f"--positions cluster and an EP-group one with the other bases.")


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
        choices=POSITION_CHOICES,
        help=f"Whose euandi answers stand for an EP group, for evaluate_euandi.py and "
             f"every plotting script that reads the positions back. Default: {POSITIONS}. "
             f"Changing it needs --override, since the vaa*.csv filenames do not carry "
             f"the basis.",
    )
    parser.add_argument(
        "--classifier",
        default=CLASSIFIER_DIR,
        help=f"Party-classifier checkpoint for classify_speeches.py (a directory with "
             f"manifest.json). Its labels must be the positions basis's groups: a "
             f"cluster-track checkpoint needs --positions cluster. Default: {CLASSIFIER_DIR}. "
             f"The *_classified.csv filenames do not carry the checkpoint either, so a "
             f"new one needs --override (or a checkpoint newer than those files).",
    )
    args = parser.parse_args()
    only = [fragment for fragment in (args.only or "").split(",") if fragment]
    if not os.path.exists(positions_path(args.positions)):
        builder = "build_cluster_positions.py" if args.positions == "cluster" else "build_group_positions.py"
        raise SystemExit(f"{positions_path(args.positions)} not found; build it with analysis/{builder}")
    if selected("classify_speeches.py", only):
        check_classifier_labels(args.classifier, args.positions)
    print(f"EP group positions basis: {args.positions}", flush=True)
    print(f"Classifier: {args.classifier}\n", flush=True)

    run_per_variant_scripts(args.override, only, args.positions, args.classifier)
    run_per_dataset_scripts(PER_DATASET_PLOTTING_SCRIPTS, only, args.positions)
    run_per_dataset_scripts(PER_DATASET_TABLE_SCRIPTS, only, args.positions)


if __name__ == "__main__":
    main()

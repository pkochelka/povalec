"""Run the whole classifier-track preprocessing pipeline, in order.

The seven scripts in preprocessing/ have to run in a fixed sequence, each one
reading what the previous wrote. This driver runs them for you, skips steps
whose outputs already exist, and handles the one interpreter split in the chain
(the fastText language-ID step; see PYTHON VERSIONS below).

    python preprocessing/run_preprocessing.py --dry-run
    python preprocessing/run_preprocessing.py
    python preprocessing/run_preprocessing.py --from lid-filter
    python preprocessing/run_preprocessing.py --languages en de fr   # small test run

Order
-----
    1  fetch        fetch_raw_data.py           download ~9 GB of corpora
    2  rdf-query    europarl_rdf_query.py       LinkedEP *.ttl -> multi-europarl.csv
    3  lid-filter   europarl_lid_filter.py      -> multi-europarl-lang_id.csv   [fastText]
    4  preprocess   preprocess_data.py          three corpora -> parquet
    5  split        split_preprocessed_data.py  group-disjoint, balanced splits
    6  clean-names  clean_party_names.py        -> cleaned/{split}.parquet
    7  collapse     build_collapsed_splits.py   merge ECR+ID, re-split from scratch

PYTHON VERSIONS
---------------
Everything here runs on Python 3.10-3.14 EXCEPT step 3, `lid-filter`, which
needs a fastText binding:

    fasttext-wheel     prebuilt wheels through 3.12
    fasttext-predict   prediction-only, wheels through 3.13
    fasttext           builds from C++ source, needs a toolchain

None of them has a 3.14 wheel. If the interpreter running this driver cannot
import a binding, the run stops before step 3 and tells you what to do. Two
ways forward:

    pip install fasttext-wheel        # if you are on 3.10-3.12
    pip install fasttext-predict      # if you are on 3.13

or point the one step at another interpreter that has one, and keep running
everything else on your normal one:

    python preprocessing/run_preprocessing.py --fasttext-python C:\\Python312\\python.exe

`--skip lid-filter` is the escape hatch if you already have
multi-europarl-lang_id.csv from elsewhere.

Costs
-----
Step 1 downloads ~9 GB. Step 2 loads every Turtle dump into one rdflib.Graph:
many hours and a very large RAM requirement at full scale. Use `--languages`
to build the whole chain on a couple of language dumps first; it is threaded
through to both steps that accept it.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EUROPARL_DIR = DATA_DIR / "EuroParl Custom"

SPLITS = ("train", "dev", "test")

# Bindings load_lid_model() in europarl_lid_filter.py tries, in its order.
FASTTEXT_MODULES = ("fasttext", "fasttext_predict")


class Step:
    """One pipeline stage: a script to run plus the outputs that prove it ran.

    `outputs` is what --force-less runs check for; a step whose every output
    already exists is skipped. `takes_languages` marks the two steps that accept
    the --languages passthrough. `needs_fasttext` marks the one step that may
    have to run on a different interpreter.
    """

    def __init__(self, name, script, description, outputs,
                 takes_languages=False, needs_fasttext=False):
        self.name = name
        self.script = script
        self.description = description
        self.outputs = outputs
        self.takes_languages = takes_languages
        self.needs_fasttext = needs_fasttext

    def satisfied(self):
        return bool(self.outputs) and all(path.exists() for path in self.outputs)


STEPS = [
    Step(
        "fetch", "fetch_raw_data.py",
        "download ParlEE, EU Debates, LinkedEP and lid.176.bin (~9 GB)",
        outputs=[
            DATA_DIR / "ParlEE" / "ParlEE_EP_plenary_speeches.csv",
            DATA_DIR / "EU Debates" / "train.jsonl",
            EUROPARL_DIR / "lid.176.bin",
        ],
        takes_languages=True,
    ),
    Step(
        "rdf-query", "europarl_rdf_query.py",
        "LinkedEP *.ttl -> multi-europarl.csv (slow, memory-hungry)",
        outputs=[EUROPARL_DIR / "multi-europarl.csv"],
        takes_languages=True,
    ),
    Step(
        "lid-filter", "europarl_lid_filter.py",
        "blank cells whose language fastText disagrees with",
        outputs=[EUROPARL_DIR / "multi-europarl-lang_id.csv"],
        needs_fasttext=True,
    ),
    Step(
        "preprocess", "preprocess_data.py",
        "EuroParl + ParlEE + EU Debates -> parquet",
        outputs=[
            EUROPARL_DIR / "preprocessed.parquet",
            EUROPARL_DIR / "preprocessed_parlee.parquet",
            EUROPARL_DIR / "preprocessed_eu_debates.parquet",
        ],
    ),
    Step(
        "split", "split_preprocessed_data.py",
        "group-disjoint, class-balanced train/dev/test",
        outputs=[EUROPARL_DIR / f"{split}.parquet" for split in SPLITS],
    ),
    Step(
        "clean-names", "clean_party_names.py",
        "repair detached accents; strip EP group names and titled person names",
        outputs=[EUROPARL_DIR / "cleaned" / f"{split}.parquet" for split in SPLITS],
    ),
    Step(
        "collapse", "build_collapsed_splits.py",
        "merge ECR+ID and re-split from scratch",
        outputs=[EUROPARL_DIR / "collapsed" / f"{name}.parquet"
                 for name in SPLITS + ("train_balanced",)],
    ),
]

STEP_NAMES = [step.name for step in STEPS]


def find_fasttext(interpreter):
    """Return the fastText module `interpreter` can import, or None."""
    probe = (
        "import importlib, sys\n"
        f"for name in {FASTTEXT_MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        continue\n"
        "    print(name)\n"
        "    break\n"
    )
    try:
        result = subprocess.run([str(interpreter), "-c", probe],
                                capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"warning: could not probe {interpreter}: {error}")
        return None
    return result.stdout.strip() or None


def interpreter_version(interpreter):
    try:
        result = subprocess.run(
            [str(interpreter), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, timeout=120, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def fasttext_advice(version):
    """The install line that actually has wheels for `version`."""
    if version in ("3.10", "3.11", "3.12"):
        return "pip install fasttext-wheel"
    if version == "3.13":
        return "pip install fasttext-predict   (fasttext-wheel has no 3.13 wheel)"
    return ("neither binding publishes wheels for Python %s -- run this step on "
            "3.10-3.13 via --fasttext-python, or build plain 'fasttext' from source"
            % (version or "?"))


def check_fasttext(interpreter, label):
    """Print what `interpreter` can do for the LID step; return True if it can."""
    version = interpreter_version(interpreter)
    module = find_fasttext(interpreter)
    if module:
        print(f"  lid-filter will use {label} (Python {version or '?'}), binding: {module}")
        return True
    print(f"  {label} (Python {version or '?'}) has no fastText binding installed.")
    print(f"    -> {fasttext_advice(version)}")
    print(f"    -> or run the step elsewhere: --fasttext-python <path-to-python>")
    print(f"    -> or --skip lid-filter if multi-europarl-lang_id.csv already exists")
    return False


def select_steps(args):
    """Apply --only / --from / --to / --skip to the step list."""
    if args.only:
        chosen = [step for step in STEPS if step.name in set(args.only)]
    else:
        start = STEP_NAMES.index(args.start) if args.start else 0
        end = STEP_NAMES.index(args.end) + 1 if args.end else len(STEPS)
        if start > end:
            raise SystemExit(f"--from {args.start} comes after --to {args.end}")
        chosen = STEPS[start:end]
    skipped = set(args.skip or [])
    return [step for step in chosen if step.name not in skipped]


def build_command(step, args, interpreter):
    command = [str(interpreter), str(PREPROCESSING / step.script)]
    if step.takes_languages and args.languages:
        command += ["--languages", *args.languages]
    command += args.extra.get(step.name, [])
    return command


def run_step(step, args, interpreter):
    command = build_command(step, args, interpreter)
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    if args.dry_run:
        print(f"  [dry-run] {printable}")
        return

    print(f"  $ {printable}")
    # cwd=PROJECT_ROOT: preprocess_data.py, split_preprocessed_data.py and
    # clean_party_names.py resolve their paths relative to the repo root.
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{step.script} exited with code {result.returncode}")


def parse_extra(values):
    """--extra 'step:--flag value' pairs, into {step: [args]}."""
    import shlex

    extra = {}
    for value in values or []:
        name, separator, rest = value.partition(":")
        if not separator or name not in STEP_NAMES:
            raise SystemExit(f"--extra needs 'step:args'; step must be one of {STEP_NAMES}")
        extra.setdefault(name, []).extend(shlex.split(rest))
    return extra


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STEP_NAMES,
                        help="start at this step")
    parser.add_argument("--to", dest="end", choices=STEP_NAMES,
                        help="stop after this step")
    parser.add_argument("--only", nargs="+", choices=STEP_NAMES,
                        help="run just these steps (ignores --from/--to)")
    parser.add_argument("--skip", nargs="+", choices=STEP_NAMES,
                        help="drop these steps from the run")
    parser.add_argument("--languages", nargs="+", metavar="CODE",
                        help="2-letter codes, passed to fetch and rdf-query; use this "
                             "to rehearse the chain on a subset before the full run")
    parser.add_argument("--fasttext-python", type=Path, default=None,
                        help="interpreter to run lid-filter with (default: this one). "
                             "Use when your main Python has no fastText wheel.")
    parser.add_argument("--force", action="store_true",
                        help="re-run steps whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the commands; run nothing")
    parser.add_argument("--extra", action="append", metavar="STEP:ARGS",
                        help="repeatable passthrough, e.g. --extra 'fetch:--dry-run' "
                             "or --extra 'collapse:--per-party 50000'")
    args = parser.parse_args()
    args.extra = parse_extra(args.extra)
    return args


def main():
    args = parse_args()
    steps = select_steps(args)
    if not steps:
        raise SystemExit("nothing to run")

    lid_python = args.fasttext_python or Path(sys.executable)
    lid_label = "this interpreter" if args.fasttext_python is None else str(lid_python)

    print(f"Repo root:   {PROJECT_ROOT}")
    print(f"Interpreter: {sys.executable} (Python {'.'.join(map(str, sys.version_info[:2]))})")
    print("\nPlan:")
    pending = []
    for index, step in enumerate(steps, start=1):
        skip = step.satisfied() and not args.force
        mark = "skip" if skip else "run "
        print(f"  [{mark}] {index}. {step.name:<14} {step.description}")
        if not skip:
            pending.append(step)
    if not args.force:
        print("  (steps whose outputs already exist are skipped; --force re-runs them)")

    needs_lid = any(step.needs_fasttext for step in pending)
    if needs_lid:
        print("\nfastText check:")
        if not check_fasttext(lid_python, lid_label) and not args.dry_run:
            raise SystemExit(
                "\nStopping before anything runs: lid-filter cannot work with the "
                "interpreter it was given. Fix one of the above, or --skip lid-filter."
            )

    if not pending:
        print("\nEverything is already built. Use --force to redo a step.")
        return

    print()
    for index, step in enumerate(pending, start=1):
        interpreter = lid_python if step.needs_fasttext else Path(sys.executable)
        print(f"=== [{index}/{len(pending)}] {step.name} - {step.description} ===")
        started = time.monotonic()
        try:
            run_step(step, args, interpreter)
        except (RuntimeError, OSError) as error:
            print(f"\nFAILED at step '{step.name}': {error}")
            print("Fix the cause and resume with "
                  f"--from {step.name} (earlier steps will skip themselves).")
            raise SystemExit(1)
        elapsed = time.monotonic() - started
        if not args.dry_run:
            print(f"  done in {elapsed / 60:.1f} min")
        print()

    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to execute.")
    else:
        print("Preprocessing complete. Next: analysis/classifier_training.py "
              "(or classifier_training_for_balanced_collapsed.py for the collapsed track).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Logit-adjusted cluster classifier with a translation-consistency loss.

The MT-augmented classifier still changes its label when a speech is only translated:
32-42% of sk/ro LLM speeches get another cluster once machine-translated into English
(classifier_argument_analysis memory, translation probe). Adding translations as extra
training rows (TRAINER=langmatched_mt) teaches each translation its label, but nothing
ties a speech and its translation to the same prediction. This trainer does.

Every training step appends CONSISTENCY_PAIRS pairs of texts that are the same speech in
two languages to the ordinary batch (one forward pass, so DDP and gradient checkpointing
see a single graph) and adds

    loss = logit-adjusted CE(batch) + w(t) * symmetric KL(p(a) || p(b)) over the pairs

with w ramping linearly from 0 to CONSISTENCY_WEIGHT over the first CONSISTENCY_WARMUP
of training. The CE part, its rows and its class / language mix are those of the chosen
train file, unchanged; the pairs only add the consistency term.

Pairs come from the train split only, in two kinds of groups (a group = one speech):
  ep   a speaker-day with exactly one speech per language in train.parquet, available in
       >= 2 languages: the original and its official EP translations
  mt   a speaker-day with one DeepSeek translation per language in train_mt.parquet
       (preprocessing/build_mt_train.py), plus its train.parquet versions when that key
       holds a single speech: machine translations, and original <-> MT pairs, which is
       where the "native-sounding Slovak = Sov-right" shortcut lives
A pair is two different versions of one group; CONSISTENCY_MT_SHARE of the pairs come
from mt groups. Texts are tokenized per step, so the pool costs only its strings.

After training (main process) the model is scored for consistency on held-out pairs:
human EP translation groups from the dev and test splits, and the LLM sk/ro texts of
data/translation_probe/translate_test.csv against their English MT. The same scoring runs
on any existing model for a baseline:

    python -m analysis.classifier_training_consistency --evaluate runs/<run>/model \\
        --data-dir "data/EuroParl Custom/clusters_k4_national_ni_eval25k"

Training goes through analysis/train_cluster_track.py --consistency-weight W (any logit-
adjusted trainer; not combined with --adversary); recipe O in slurm_train_cluster_track.sh.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from analysis import classifier_training as base

# train_cluster_track.py sets these on this module; main() hands them to classifier_training.
DATA_DIR = base.DATA_DIR
OUTPUT_DIR = base.OUTPUT_DIR
TRAIN_SPLIT = base.TRAIN_SPLIT
DEV_SPLIT = base.DEV_SPLIT
TEST_SPLIT = base.TEST_SPLIT

CONSISTENCY_WEIGHT = 1.0
CONSISTENCY_PAIRS = 8          # pairs per process per step, on top of the ordinary batch
CONSISTENCY_MT_SHARE = 0.5     # share of pairs drawn from the machine-translation groups
CONSISTENCY_WARMUP = 0.1       # fraction of training over which the weight ramps up from 0
CONSISTENCY_MAX_EP_GROUPS = 50000
EVAL_GROUPS = 2000             # held-out EP translation groups scored after training
PAIRS_CSV = os.path.join(base.PROJECT_ROOT, "data", "translation_probe", "translate_test.csv")
KEY = ["speaker", "date"]


# --------------------------------------------------------------------------- #
# pair pool
# --------------------------------------------------------------------------- #

def single_speech(df):
    """Rows whose speaker-day holds one speech per language (so its versions are one speech)."""
    per_language = df.groupby(KEY + ["language"]).size()
    single = per_language.groupby(level=[0, 1]).max() == 1
    keys = single[single].index
    return df.set_index(KEY).loc[lambda d: d.index.isin(keys)].reset_index()


class PairPool:
    """Groups of texts that are the same speech; draws two versions of one group."""

    def __init__(self, texts, groups, mt_share):
        self.texts = texts
        self.groups = {kind: g for kind, g in groups.items() if len(g)}
        self.mt_share = mt_share if "mt" in self.groups else 0.0

    def sample(self, n, rng):
        a, b = [], []
        for _ in range(n):
            kind = "mt" if rng.random() < self.mt_share else "ep"
            members = self.groups[kind][rng.integers(len(self.groups[kind]))]
            i, j = rng.choice(len(members), 2, replace=False)
            a.append(self.texts[members[i]])
            b.append(self.texts[members[j]])
        return a, b

    def describe(self):
        return ", ".join(f"{kind}: {len(g)} groups, {sum(len(m) for m in g)} texts" for kind, g in self.groups.items())


def build_pair_pool(data_dir, seed):
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)
    train = pd.read_parquet(data_dir / "train.parquet", columns=KEY + ["language", "text"]).dropna(subset=["text"])
    train = single_speech(train)
    mt_path = data_dir / "train_mt.parquet"
    mt = (single_speech(pd.read_parquet(mt_path, columns=KEY + ["language", "text"]).dropna(subset=["text"]))
          if mt_path.exists() else pd.DataFrame(columns=KEY + ["language", "text"]))
    if mt.empty:
        print(f"[consistency] no {mt_path}: human EP translation groups only")
    texts, groups = [], {"ep": [], "mt": []}

    def add(frame, kind):
        if len(frame) >= 2:
            start = len(texts)
            texts.extend(frame.text.astype(str).tolist())
            groups[kind].append(np.arange(start, len(texts)))

    # mt groups: the DeepSeek versions plus the train.parquet versions of the same speech
    in_mt = train.set_index(KEY).index.isin(pd.MultiIndex.from_frame(mt[KEY].drop_duplicates()))
    originals = dict(tuple(train[in_mt].groupby(KEY)))
    for key, g in mt.groupby(KEY):
        add(pd.concat([g, originals[key]]) if key in originals else g, "mt")
    print(f"[consistency] {len(originals)} of {mt.groupby(KEY).ngroups} MT speeches also have train.parquet versions")
    # ep groups: speeches with official translations and no MT, capped
    ep_groups = [g for _, g in train[~in_mt].groupby(KEY) if len(g) >= 2]
    if len(ep_groups) > CONSISTENCY_MAX_EP_GROUPS:
        keep = rng.choice(len(ep_groups), CONSISTENCY_MAX_EP_GROUPS, replace=False)
        ep_groups = [ep_groups[i] for i in np.sort(keep)]
    for g in ep_groups:
        add(g, "ep")
    return PairPool(texts, groups, CONSISTENCY_MT_SHARE)


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #

POOL = None   # set in main() before classifier_training builds the trainer


class ConsistencyTrainer(base.LogitAdjustedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.adversary is not None:
            raise SystemExit("the consistency trainer is not combined with an adversary")
        self.pool = POOL
        self.rng = np.random.default_rng(base.SEED + 1000 * base.LOCAL_RANK)
        self.cons_stats = np.zeros(3)   # summed symmetric KL, agreeing pairs, pairs since the last print

    def consistency_weight(self):
        progress = self.state.global_step / max(self.state.max_steps, 1)
        return CONSISTENCY_WEIGHT * min(1.0, progress / CONSISTENCY_WARMUP) if CONSISTENCY_WARMUP else CONSISTENCY_WEIGHT

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training or self.pool is None or CONSISTENCY_WEIGHT <= 0 or CONSISTENCY_PAIRS <= 0:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        extra = set(inputs) - {"input_ids", "attention_mask"}
        if extra:
            raise SystemExit(f"unexpected model inputs for the consistency trainer: {sorted(extra)}")
        a, b = self.pool.sample(CONSISTENCY_PAIRS, self.rng)
        pairs = self.processing_class(a + b, truncation=True, max_length=base.MAX_LEN, padding=True,
                                      return_tensors="pt")
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        length = max(ids.shape[1], pairs["input_ids"].shape[1])
        pad_id = self.processing_class.pad_token_id

        def pad(t, value):
            return F.pad(t, (0, length - t.shape[1]), value=value)

        merged = {"input_ids": torch.cat([pad(ids, pad_id), pad(pairs["input_ids"].to(ids.device), pad_id)]),
                  "attention_mask": torch.cat([pad(mask, 0), pad(pairs["attention_mask"].to(mask.device), 0)])}
        outputs = model(**merged)
        n = labels.shape[0]
        logits = outputs.logits
        loss = F.cross_entropy(logits[:n] + self.logit_adjustment, labels)
        la = F.log_softmax(logits[n:n + CONSISTENCY_PAIRS].float(), dim=-1)
        lb = F.log_softmax(logits[n + CONSISTENCY_PAIRS:].float(), dim=-1)
        skl = 0.5 * (F.kl_div(la, lb, log_target=True, reduction="batchmean")
                     + F.kl_div(lb, la, log_target=True, reduction="batchmean"))
        loss = loss + self.consistency_weight() * skl
        with torch.no_grad():
            self.cons_stats += [float(skl) * CONSISTENCY_PAIRS, float((la.argmax(-1) == lb.argmax(-1)).sum()),
                                CONSISTENCY_PAIRS]
        step = self.state.global_step
        if step % self.args.logging_steps == 0 and self.cons_stats[2]:
            if base.IS_MAIN_PROCESS:
                print(f"[consistency] step {step}: weight {self.consistency_weight():.3f}; pair symmetric KL "
                      f"{self.cons_stats[0] / self.cons_stats[2]:.4f}, same argmax "
                      f"{self.cons_stats[1] / self.cons_stats[2]:.3f}")
            self.cons_stats[:] = 0
        outputs.logits = logits[:n]
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# held-out consistency
# --------------------------------------------------------------------------- #

def deployed_labels(texts, classifier, device, batch=64):
    """argmax(logits + manifest biases), the labels the analysis uses."""
    from analysis.classify_speeches import predict_class_logits
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.empty(len(texts), dtype=int)
    for start in range(0, len(texts), batch):
        idx = order[start:start + batch]
        logits = predict_class_logits([texts[i] for i in idx], classifier, device)
        if classifier.biases is not None:
            logits = logits + classifier.biases
        out[idx] = logits.argmax(1)
    return out


def kappa(a, b):
    a, b = pd.Series(a), pd.Series(b)
    agree = float((a.to_numpy() == b.to_numpy()).mean())
    chance = float((a.value_counts(normalize=True) * b.value_counts(normalize=True)).fillna(0).sum())
    return agree, (agree - chance) / (1 - chance) if chance < 1 else float("nan")


def evaluate_consistency(model_dir, data_dir, pairs_csv=PAIRS_CSV, n_groups=EVAL_GROUPS, seed=base.SEED):
    """Label agreement between versions of one speech, for any classifier with a manifest."""
    from analysis.classify_speeches import load_classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = load_classifier(str(model_dir), device)
    names = np.array(clf.labels)
    lines = [f"translation consistency of {model_dir} (deployed labels: argmax(logits + biases))"]
    rng = np.random.default_rng(seed)

    # human EP translation groups from dev and test
    frames = [pd.read_parquet(Path(data_dir) / f"{s}.parquet", columns=KEY + ["language", "text"])
              for s in ("dev", "test") if (Path(data_dir) / f"{s}.parquet").exists()]
    if frames:
        held = single_speech(pd.concat(frames, ignore_index=True).dropna(subset=["text"]))
        keys = held.groupby(KEY).size()
        keys = keys[keys >= 2].index
        pick = keys[np.sort(rng.choice(len(keys), min(n_groups, len(keys)), replace=False))]
        held = held.set_index(KEY).loc[pick].reset_index()
        held["label"] = names[deployed_labels(held.text, clf, device)]
        pair_agree, majority = [], []
        for _, g in held.groupby(KEY):
            counts = g.label.value_counts()
            k = len(g)
            pair_agree.append(float((counts * (counts - 1)).sum() / (k * (k - 1))))
            majority.append(counts.iloc[0] / k)
        lines.append(f"  EP translation groups (dev+test, {len(pick)} speeches, {len(held)} versions, "
                     f"{held.language.nunique()} languages): pairwise label agreement {np.mean(pair_agree):.1%}, "
                     f"versions with the speech's majority label {np.mean(majority):.1%}")
        en = held[held.language == "en"].set_index(KEY).label
        other = held[held.language != "en"]
        other = other[other.set_index(KEY).index.isin(en.index)]
        if len(other):
            other = other.assign(en_label=en.reindex(other.set_index(KEY).index).to_numpy())
            agree, k = kappa(other.label, other.en_label)
            by_lang = other.groupby("language").apply(lambda g: (g.label == g.en_label).mean())
            lines.append(f"  same speech in language X vs its English version: agreement {agree:.1%} (kappa {k:.2f}); "
                         f"lowest {by_lang.nsmallest(3).round(2).to_dict()}")
    # LLM texts vs their English MT
    if pairs_csv and Path(pairs_csv).exists():
        p = pd.read_csv(pairs_csv, encoding="utf-8-sig").dropna(subset=["text", "text_en"])
        p["orig"] = names[deployed_labels(p.text, clf, device)]
        p["mt"] = names[deployed_labels(p.text_en, clf, device)]
        for (track, lang), g in p.groupby(["track", "language"]):
            agree, k = kappa(g.orig, g.mt)
            lines.append(f"  LLM {track} {lang} vs English MT (n={len(g)}): label kept {agree:.1%} (kappa {k:.2f})")
    for line in lines:
        print(line)
    return lines


# --------------------------------------------------------------------------- #

def consistency_manifest():
    return {"weight": CONSISTENCY_WEIGHT, "pairs_per_step_per_process": CONSISTENCY_PAIRS,
            "mt_share": CONSISTENCY_MT_SHARE, "warmup": CONSISTENCY_WARMUP,
            "max_ep_groups": CONSISTENCY_MAX_EP_GROUPS, "loss": "symmetric KL of softmax(logits)",
            "pool": POOL.describe() if POOL is not None else None}


def main():
    global POOL
    for name in ("DATA_DIR", "OUTPUT_DIR", "TRAIN_SPLIT", "DEV_SPLIT", "TEST_SPLIT"):
        setattr(base, name, globals()[name])
    if base.ADVERSARY:
        raise SystemExit("--consistency-weight is not combined with --adversary")
    if not base.EVALUATE_ONLY:
        POOL = build_pair_pool(DATA_DIR, base.SEED)
        print(f"[consistency] weight {CONSISTENCY_WEIGHT}, {CONSISTENCY_PAIRS} pairs/step/process "
              f"(mt share {CONSISTENCY_MT_SHARE}), warm-up {CONSISTENCY_WARMUP:.0%}; pool {POOL.describe()}")
        base.LogitAdjustedTrainer = ConsistencyTrainer   # train_and_evaluate builds it by this name
    base.main()
    if not base.IS_MAIN_PROCESS:
        return
    dirs = [OUTPUT_DIR] + [base.snapshot_dir(e) for e in sorted(base.SNAPSHOT_EPOCHS)]
    for model_dir in dirs:
        manifest_path = Path(model_dir) / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["consistency"] = consistency_manifest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = evaluate_consistency(model_dir, DATA_DIR)
        out = f"results_consistency_{Path(model_dir).name}.txt"
        Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved {out}")


def cli():
    parser = argparse.ArgumentParser(description="translation consistency of an existing classifier")
    parser.add_argument("--evaluate", required=True, help="model dir with manifest.json")
    parser.add_argument("--data-dir", required=True, help="track dir with dev.parquet / test.parquet")
    parser.add_argument("--pairs", default=PAIRS_CSV)
    parser.add_argument("--groups", type=int, default=EVAL_GROUPS)
    parser.add_argument("--out", default="", help="default: results_consistency_<run>.txt")
    args = parser.parse_args()
    lines = evaluate_consistency(args.evaluate, args.data_dir, args.pairs, args.groups)
    out = args.out or f"results_consistency_{Path(args.evaluate).resolve().parent.name}.txt"
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    cli()

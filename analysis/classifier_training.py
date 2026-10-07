#!/usr/bin/env python3
"""Train a single mmBERT EU-party classifier that is robust to class imbalance.

Runs on the ECR+ID-collapsed split track (data/EuroParl Custom/collapsed), whose
train.parquet keeps the natural, imbalanced priors while dev/test come out uniform
across parties by construction -- exactly the setup this loss is written for.

Train is heavily imbalanced, but dev/test are uniform across parties (and
language-stratified). Logit-adjusted cross-entropy subtracts the train log-priors
during the loss, so the model is optimised for a uniform label distribution: at
inference the plain argmax of the logits is Bayes-optimal for the uniform dev/test.
The best checkpoint is picked on the uniform dev split and evaluated once on test.
"""
import functools
import inspect
import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (registers pa.compute)
import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainingArguments, set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

# accelerate probes for the optional lomo_optim package on EVERY optimizer
# step; with the package absent, importlib re-lists site-packages whenever its
# finder cache is stale. On a network-mounted venv a single transient I/O
# error there kills the whole training run, so cache the probe to one call.
try:
    import accelerate.optimizer
    accelerate.optimizer.is_lomo_available = functools.cache(
        accelerate.optimizer.is_lomo_available)
except (ImportError, AttributeError):
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analysis.europarl_classification import (
    MIN_LANGUAGE_SAMPLES,
    PARTY_COLUMN,
    attach_labels,
    build_label_maps,
    classification_report_text,
    fit_uniform_bias,
    load_split,
    per_language_f1_report,
)

MODEL_NAME = "jhu-clsp/mmBERT-base"
MODEL_SLUG = MODEL_NAME.split("/")[-1]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "EuroParl Custom", "collapsed")
# train_cluster_track.py --trainer langmatched points these at the *_langmatched files.
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"
TEST_SPLIT = "test"
OUTPUT_DIR = f"{MODEL_SLUG}-logitadj-collapsed"
MAX_LEN = 512
SEED = 42
MAX_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2
LOGIT_ADJUSTMENT_TAU = 1.0
EPSILON = 1e-12
BEST_METRIC = "f1_macro_mean_lang"

# Speed settings. The defaults reproduce the original runs; train_cluster_track.py
# overrides them per run. Gradient checkpointing trades ~30% step time for memory a
# 48 GB card does not need at batch 32 x 512. Length grouping batches texts of similar
# length so less of each batch is padding (logit-adjusted trainer only).
GRADIENT_CHECKPOINTING = True
GROUP_BY_LENGTH = False
DATALOADER_WORKERS = 2
TOKENIZE_PROC = None
# Epochs after which an extra copy of the model is kept, as <OUTPUT_DIR>_epoch<N>, with
# its own dev-fitted biases, manifest and test results (logit-adjusted trainer only).
SNAPSHOT_EPOCHS = ()
# Multi-GPU (torchrun, one process per GPU): the global batch stays 32, split over the
# processes, so a 4-GPU run optimizes like the 1-GPU one. Eval batches change no result.
GLOBAL_TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
# bf16 instead of fp16 autocast: same speed on Ampere+, no loss scaling, no overflow skips.
BF16 = False
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_MAIN_PROCESS = int(os.environ.get("RANK", "0")) == 0
# Skip training and only evaluate the existing SNAPSHOT_EPOCHS snapshots (biases, manifest,
# results), e.g. a mid-run checkpoint copied to <OUTPUT_DIR>_epoch<N>.
EVALUATE_ONLY = False
# Origin adversary (train_cluster_track.py --adversary origin). Original-language EP text
# comes only from that country's MEPs (original Slovak is 99% Sov-right), and AI-written or
# machine-translated text reads as original, so "native-sounding Slovak" became a cluster
# cue. A small MLP on the classifier input predicts, within each text's language, whether
# it is EP-translated, original or machine-translated; a gradient-reversal layer between
# it and the encoder trains the encoder to make that impossible. The reversal strength
# ramps from 0 to ADVERSARY_WEIGHT (Ganin et al. 2016). Not conditioned on the label:
# original Slovak has no non-Sov-right rows, so a label-conditioned adversary would only
# act inside Sov-right. analysis/leace_probe.py tests the linear, no-retrain version.
ADVERSARY = ""
ADVERSARY_WEIGHT = 0.3
ADVERSARY_HIDDEN = 256
# A language takes part only with >= 2 origins of at least this many train rows each.
ADVERSARY_MIN_ROWS = 200
ORIGINS = ("translated", "original", "mt")


@dataclass
class TrainingData:
    train: Dataset
    dev: Dataset
    test: Dataset
    collator: DataCollatorWithPadding
    label2id: dict
    id2label: dict
    num_labels: int
    target_names: list
    dev_langs: np.ndarray
    test_langs: np.ndarray
    train_lengths: list
    adv_languages: list = None        # language index -> code, for the adversary's cells
    adv_weights: np.ndarray = None    # per cell: balances origins within each language


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class OriginAdversary(torch.nn.Module):
    """Origin logits for every language from the classifier input; the text's language row is read.

    Attached as model.origin_adversary, so the optimizer, DDP and checkpoints include it,
    and run from a forward hook on model.classifier, so it runs inside the (DDP) forward.
    """

    def __init__(self, dim, n_languages, n_origins, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                       torch.nn.Linear(hidden, n_languages * n_origins))
        self.shape = (n_languages, n_origins)
        self.scale = 0.0
        self.logits = None

    def capture(self, module, inputs, output):
        if self.training:
            self.logits = self.net(GradientReversal.apply(inputs[0], self.scale)).view(-1, *self.shape)


def attach_adversary(model, data, device):
    adversary = OriginAdversary(model.classifier.in_features, len(data.adv_languages), len(ORIGINS),
                                ADVERSARY_HIDDEN).to(device)
    model.origin_adversary = adversary
    model._adversary_hook = model.classifier.register_forward_hook(adversary.capture)
    return adversary


def detach_adversary(model):
    """Drop the adversary before the final save: the classifier alone is the deliverable."""
    model._adversary_hook.remove()
    del model.origin_adversary, model._adversary_hook


def origin_column(df, data_dir):
    """'original' / 'translated' per row as in build_language_matched_split.origin_of, and
    'mt' for rows train.parquet lacks (the machine-translated rows of train_langmatched_mt).

    Counted on the full train.parquet: a downsampled train file keeps only some language
    versions of a speech, which would make its translations look original."""
    from preprocessing.build_language_matched_split import origin_of
    key = ["speaker", "date", "language"]
    full = pd.read_parquet(os.path.join(data_dir, "train.parquet"), columns=key)
    full["origin"] = origin_of(full, 2)
    full = full.drop_duplicates(key)
    return df[key].merge(full, on=key, how="left")["origin"].fillna("mt").to_numpy()


def adversary_cells(origins, languages):
    """Per row: language index * len(ORIGINS) + origin index, or -100 where that origin has
    fewer than ADVERSARY_MIN_ROWS rows or the language has fewer than two such origins.
    Weights per cell make every usable origin weigh the same within its language."""
    lang_codes = pd.Categorical(languages)
    lang_ids = lang_codes.codes.astype(np.int64)
    origin_ids = np.array([ORIGINS.index(o) for o in origins], dtype=np.int64)
    counts = np.zeros((len(lang_codes.categories), len(ORIGINS)))
    np.add.at(counts, (lang_ids, origin_ids), 1)
    usable = counts >= ADVERSARY_MIN_ROWS
    usable &= (usable.sum(1) >= 2)[:, None]
    weights = np.where(usable, counts.sum(1, keepdims=True) / np.maximum(counts, 1)
                       / np.maximum(usable.sum(1, keepdims=True), 1), 0.0)
    cells = lang_ids * len(ORIGINS) + origin_ids
    cells[~usable[lang_ids, origin_ids]] = -100
    table = pd.DataFrame(counts.astype(int), index=lang_codes.categories, columns=ORIGINS)
    table["adversary"] = ["/".join(o for o, u in zip(ORIGINS, row) if u) or "-" for row in usable]
    print(f"Origin adversary: train rows per (language, origin); cells with < {ADVERSARY_MIN_ROWS} rows "
          f"are ignored, {int((cells >= 0).sum())}/{len(cells)} rows take part\n{table.to_string()}")
    return cells, list(lang_codes.categories), weights.reshape(-1)


class LanguageAwareMetrics:
    def __init__(self):
        self.current_languages = None

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        top2 = np.argsort(logits, axis=-1)[:, -2:]

        results = {
            "accuracy":      float((preds == labels).mean()),
            "top2_accuracy": float(np.mean([label in t for label, t in zip(labels, top2)])),
            "f1_macro":      f1_score(labels, preds, average="macro",    zero_division=0),
            "f1_weighted":   f1_score(labels, preds, average="weighted", zero_division=0),
        }

        if self.current_languages is None:
            return results

        f1_by_language = {}
        for language in np.unique(self.current_languages):
            mask = self.current_languages == language
            if mask.sum() < MIN_LANGUAGE_SAMPLES:
                continue
            f1_by_language[language] = f1_score(
                labels[mask], preds[mask], average="macro", zero_division=0
            )
            results[f"f1_macro_{language}"] = f1_by_language[language]

        if f1_by_language:
            results["f1_macro_mean_lang"]     = float(np.mean(list(f1_by_language.values())))
            results["f1_macro_min_lang"]      = float(min(f1_by_language.values()))
            results["f1_macro_min_lang_name"] = min(f1_by_language, key=f1_by_language.get)

        return results


class LogitAdjustedTrainer(Trainer):
    def __init__(self, logit_adjustment, train_lengths=None, adversary=None, adversary_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.logit_adjustment = logit_adjustment
        self.train_lengths = train_lengths
        self.adversary = adversary
        self.adversary_weights = adversary_weights
        self.adversary_stats = np.zeros(3)   # weighted loss, weighted hits, weight since the last print

    def _set_signature_columns_if_needed(self):
        # The Trainer drops dataset columns that model.forward does not take; keep the
        # adversary's cell id, which compute_loss pops before the forward.
        super()._set_signature_columns_if_needed()
        if self.adversary is not None and "adv_cell" not in self._signature_columns:
            self._signature_columns.append("adv_cell")

    def adversary_scale(self):
        progress = self.state.global_step / max(self.state.max_steps, 1)
        return float(ADVERSARY_WEIGHT * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))

    def adversary_loss(self, cells):
        logits, self.adversary.logits = self.adversary.logits, None
        valid = cells >= 0
        safe = cells.clamp_min(0)
        n_origins = logits.shape[-1]
        rows = logits[torch.arange(len(cells), device=cells.device), safe // n_origins].float()
        origin = safe % n_origins
        weight = self.adversary_weights[safe] * valid
        ce = F.cross_entropy(rows, origin, reduction="none")
        loss = (ce * weight).sum() / weight.sum().clamp_min(EPSILON)
        with torch.no_grad():
            hits = ((rows.argmax(-1) == origin).float() * weight).sum()
            self.adversary_stats += [float((ce * weight).sum()), float(hits), float(weight.sum())]
        step = self.state.global_step
        if step % self.args.logging_steps == 0 and self.adversary_stats[2] > 0:
            if IS_MAIN_PROCESS:
                total, hit, w = self.adversary_stats
                print(f"[adversary] step {step}: reversal {self.adversary.scale:.3f}, origin loss {total / w:.3f}, "
                      f"origin-balanced accuracy {hit / w:.3f}")
            self.adversary_stats[:] = 0
        return loss

    def _get_train_sampler(self, *args, **kwargs):
        # Built here rather than through TrainingArguments: the flag was renamed between
        # transformers 4 and 5, and the Trainer drops a "length" column before sampling,
        # which would make LengthGroupedSampler re-read every row to measure it.
        if self.train_lengths is None:
            return super()._get_train_sampler(*args, **kwargs)
        # Every process must draw the same shuffle: accelerate then hands each one its
        # own share of the batches. A seeded generator guarantees that; one process
        # keeps the global RNG, as in the earlier runs.
        generator = torch.Generator().manual_seed(SEED) if WORLD_SIZE > 1 else None
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=self.train_lengths,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        cells = inputs.pop("adv_cell", None)   # train batches only
        if self.adversary is not None:
            self.adversary.scale = self.adversary_scale()
        outputs = model(**inputs)
        adjusted_logits = outputs.logits + self.logit_adjustment
        loss = F.cross_entropy(adjusted_logits, labels)
        if cells is not None and self.adversary is not None and self.adversary.logits is not None:
            loss = loss + self.adversary_loss(cells)
        return (loss, outputs) if return_outputs else loss


def select_device():
    # Under torchrun every process owns GPU LOCAL_RANK; tensors built outside the Trainer
    # (the logit adjustment) must live there too, not on cuda:0.
    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"--- Device: {device} (process {LOCAL_RANK + 1}/{WORLD_SIZE}) ---")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    return device


def logit_adjustment_for(dataset, num_labels, device):
    counts = np.bincount(np.asarray(dataset["labels"]), minlength=num_labels)
    priors = counts / counts.sum()
    return torch.tensor(
        LOGIT_ADJUSTMENT_TAU * np.log(priors + EPSILON), dtype=torch.float, device=device,
    )


def prepare_training_data(data_dir, tokenizer, train_split="train", dev_split="dev", test_split="test"):
    """Load the three splits and tokenize them.

    `train_split` names the file the train set comes from: "train" keeps the
    natural priors (for the logit-adjusted loss here), "train_balanced" the
    party- and language-balanced downsample (for the plain-CE trainers).
    `dev_split`/`test_split` likewise, e.g. the *_langmatched eval files.
    """
    raw_splits = {
        "train": load_split(train_split, data_dir),
        "dev": load_split(dev_split, data_dir),
        "test": load_split(test_split, data_dir),
    }

    label_list = sorted(raw_splits["train"][PARTY_COLUMN].unique().tolist())
    label2id, id2label = build_label_maps(label_list)
    num_labels = len(label_list)
    print(f"{num_labels} classes: {label2id}")

    splits = {name: attach_labels(df, label2id) for name, df in raw_splits.items()}
    adv_languages = adv_weights = None
    if ADVERSARY:
        # attach_labels keeps only text/labels/language, in the order of the rows it keeps.
        kept = raw_splits["train"][raw_splits["train"][PARTY_COLUMN].isin(label2id)]
        splits["train"]["adv_cell"], adv_languages, adv_weights = adversary_cells(
            origin_column(kept, data_dir), kept["language"].to_numpy())
    print("Train class counts:\n", splits["train"]["labels"].value_counts().sort_index().to_dict())
    print("Train languages:\n",    splits["train"]["language"].value_counts().to_dict())

    dataset = DatasetDict({
        "train":      Dataset.from_pandas(splits["train"], preserve_index=False),
        "validation": Dataset.from_pandas(splits["dev"],   preserve_index=False),
        "test":       Dataset.from_pandas(splits["test"],  preserve_index=False),
    })
    for split in dataset:
        dataset[split] = dataset[split].cast_column("labels", ClassLabel(num_classes=num_labels))
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text", "language"],
        num_proc=TOKENIZE_PROC,
    )
    train_lengths = pa.compute.list_value_length(tokenized["train"].data.column("input_ids")).to_pylist()

    return TrainingData(
        train=tokenized["train"],
        dev=tokenized["validation"],
        test=tokenized["test"],
        collator=DataCollatorWithPadding(tokenizer=tokenizer),
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        target_names=[id2label[i] for i in range(num_labels)],
        dev_langs=splits["dev"]["language"].to_numpy(),
        test_langs=splits["test"]["language"].to_numpy(),
        train_lengths=train_lengths,
        adv_languages=adv_languages,
        adv_weights=adv_weights,
    )


def build_model(data, hf_token, device):
    set_seed(SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, token=hf_token,
        num_labels=data.num_labels, id2label=data.id2label, label2id=data.label2id,
    ).to(device)


def warmup_kwargs(ratio):
    # transformers 5 dropped warmup_ratio; a float < 1 in warmup_steps is read as a ratio.
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def training_arguments(output_dir, num_epochs, evaluate_each_epoch):
    return TrainingArguments(
        output_dir=output_dir,
        fp16=not BF16,
        bf16=BF16,
        eval_strategy="epoch" if evaluate_each_epoch else "no",
        save_strategy="epoch" if evaluate_each_epoch else "no",
        logging_steps=500,
        per_device_train_batch_size=GLOBAL_TRAIN_BATCH_SIZE // WORLD_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        ddp_find_unused_parameters=False if WORLD_SIZE > 1 else None,
        gradient_accumulation_steps=1,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        dataloader_num_workers=DATALOADER_WORKERS,
        num_train_epochs=num_epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        **warmup_kwargs(0.06),
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        load_best_model_at_end=evaluate_each_epoch,
        metric_for_best_model=BEST_METRIC if evaluate_each_epoch else None,
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        seed=SEED,
    )


def release(model, trainer, device):
    del model, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def snapshot_dir(epoch):
    return f"{OUTPUT_DIR}_epoch{epoch}"


class EpochSnapshot(TrainerCallback):
    """Keep a copy of the model as it is at the end of the listed epochs.

    save_total_limit=1 deletes earlier epoch checkpoints, and load_best_model_at_end
    replaces the final weights with the best-dev ones, so neither leaves the
    epoch-N model behind on its own.
    """

    def __init__(self, epochs, tokenizer):
        self.epochs = set(epochs)
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch in self.epochs and state.is_world_process_zero:
            model.save_pretrained(snapshot_dir(epoch))
            self.tokenizer.save_pretrained(snapshot_dir(epoch))
            print(f"Saved epoch-{epoch} snapshot to {snapshot_dir(epoch)}")


def fit_biases_and_test(trainer, data, metrics_fn):
    """Dev-fitted per-class bias, then test predictions with that bias applied."""
    metrics_fn.current_languages = data.dev_langs
    dev_predictions = trainer.predict(data.dev)
    dev_shares = np.bincount(dev_predictions.label_ids, minlength=data.num_labels)
    biases = fit_uniform_bias(dev_predictions.predictions, data.num_labels, target=dev_shares)
    print("Dev-marginal bias (fit on dev): "
          + ", ".join(f"{name}={b:+.3f}" for name, b in zip(data.target_names, biases)))
    metrics_fn.current_languages = data.test_langs
    predictions = trainer.predict(data.test)
    return biases, predictions.label_ids, np.argmax(predictions.predictions + biases, axis=-1)


def evaluate_snapshot(epoch, data, tokenizer, device):
    """Biases, manifest and results file for one epoch snapshot, like the main model."""
    path = snapshot_dir(epoch)
    if not os.path.isdir(path):
        print(f"No epoch-{epoch} snapshot (training stopped earlier); skipping.")
        return
    print(f"\n{'=' * 60}\n  Evaluating epoch-{epoch} snapshot\n{'=' * 60}")
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    metrics_fn = LanguageAwareMetrics()
    trainer = Trainer(
        model=model,
        args=training_arguments(os.path.join(path, "_eval"), 1, evaluate_each_epoch=False),
        data_collator=data.collator,
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
    )
    biases, y_test, y_pred = fit_biases_and_test(trainer, data, metrics_fn)
    release(model, trainer, device)
    report_and_save(path, data, epoch, biases, y_test, y_pred, tag_extra=f"_snapshot-ep{epoch}")


def train_and_evaluate(data, tokenizer, hf_token, device):
    print(f"\n{'=' * 60}\n  Training on imbalanced train, selecting on uniform dev\n{'=' * 60}")
    model = build_model(data, hf_token, device)
    metrics_fn = LanguageAwareMetrics()
    metrics_fn.current_languages = data.dev_langs

    callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    if SNAPSHOT_EPOCHS:
        callbacks.append(EpochSnapshot(SNAPSHOT_EPOCHS, tokenizer))
    adversary = attach_adversary(model, data, device) if ADVERSARY else None
    if adversary is not None:
        print(f"Origin adversary on: max reversal {ADVERSARY_WEIGHT}, hidden {ADVERSARY_HIDDEN}, "
              f"{len(data.adv_languages)} languages x {ORIGINS}")
    trainer = LogitAdjustedTrainer(
        logit_adjustment=logit_adjustment_for(data.train, data.num_labels, device),
        train_lengths=data.train_lengths if GROUP_BY_LENGTH else None,
        adversary=adversary,
        adversary_weights=(torch.tensor(data.adv_weights, dtype=torch.float, device=device)
                           if adversary is not None else None),
        model=model,
        args=training_arguments(OUTPUT_DIR, MAX_EPOCHS, evaluate_each_epoch=True),
        train_dataset=data.train,
        eval_dataset=data.dev,
        data_collator=data.collator,
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )
    trainer.train()  # load_best_model_at_end restores the best-dev checkpoint
    if adversary is not None:
        detach_adversary(model)

    scored = [entry for entry in trainer.state.log_history if f"eval_{BEST_METRIC}" in entry]
    best = max(scored, key=lambda entry: entry[f"eval_{BEST_METRIC}"])
    best_epoch = max(1, round(best["epoch"]))
    print(f"Best dev {BEST_METRIC}={best[f'eval_{BEST_METRIC}']:.4f} at epoch {best_epoch}")

    # Per-class bias that matches the argmax marginal to the dev label shares, fit on dev only.
    biases, y_test, y_pred = fit_biases_and_test(trainer, data, metrics_fn)
    trainer.save_model(OUTPUT_DIR)
    release(model, trainer, device)
    return best_epoch, biases, y_test, y_pred


def save_manifest(output_dir, data, num_epochs, biases):
    manifest = {
        "model_name": MODEL_NAME,
        "seed": SEED,
        "max_len": MAX_LEN,
        "num_epochs": num_epochs,
        "logit_adjustment_tau": LOGIT_ADJUSTMENT_TAU,
        "global_train_batch": GLOBAL_TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "precision": "bf16" if BF16 else "fp16",
        "label2id": data.label2id,
        "biases": [float(b) for b in biases],
        "inference": "argmax(model logits + biases)",
        "data_dir": str(DATA_DIR),
        "splits": {"train": TRAIN_SPLIT, "dev": DEV_SPLIT, "test": TEST_SPLIT},
        "adversary": ({"target": ADVERSARY, "weight": ADVERSARY_WEIGHT, "hidden": ADVERSARY_HIDDEN,
                       "min_rows": ADVERSARY_MIN_ROWS, "origins": list(ORIGINS), "languages": data.adv_languages}
                      if ADVERSARY else None),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def write_results_file(output_tag, num_epochs, test_f1, overall_report,
                       language_summary, per_language_reports):
    path = f"results_{output_tag}.txt"
    with open(path, "w") as f:
        f.write(output_tag + "\n" + "-" * 40 + "\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Logit adjustment tau: {LOGIT_ADJUSTMENT_TAU}\n")
        f.write(f"Best epoch from dev search: {num_epochs}\n")
        f.write(f"Test f1_macro: {test_f1:.4f}\n\n")
        f.write("=== OVERALL ===\n" + overall_report)
        f.write(language_summary)
        f.write("\n".join(per_language_reports))
    print(f"\nSaved {path}")


def main():
    if GLOBAL_TRAIN_BATCH_SIZE % WORLD_SIZE:
        raise SystemExit(f"batch {GLOBAL_TRAIN_BATCH_SIZE} does not split over {WORLD_SIZE} GPUs")
    if "EVAL_CHECKPOINT" in os.environ and not EVALUATE_ONLY:
        # Set (even empty) by slurm_train_cluster_track.sh; never fall through to training.
        raise SystemExit(f"EVAL_CHECKPOINT={os.environ['EVAL_CHECKPOINT']!r} but EVALUATE_ONLY is not set "
                         "(empty path, or a stale analysis/train_cluster_track.py): refusing to train")
    if EVALUATE_ONLY:
        print(f"--- EVALUATE_ONLY: no training, evaluating epoch snapshots {sorted(SNAPSHOT_EPOCHS)} ---")
    device = select_device()
    load_dotenv(os.path.expanduser("~/.env.local"))
    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    data = prepare_training_data(DATA_DIR, tokenizer, train_split=TRAIN_SPLIT,
                                 dev_split=DEV_SPLIT, test_split=TEST_SPLIT)

    if not EVALUATE_ONLY:
        best_epoch, biases, y_test, y_pred = train_and_evaluate(data, tokenizer, hf_token, device)
        report_and_save(OUTPUT_DIR, data, best_epoch, biases, y_test, y_pred)

    for epoch in sorted(SNAPSHOT_EPOCHS):
        evaluate_snapshot(epoch, data, tokenizer, device)


def report_and_save(output_dir, data, num_epochs, biases, y_test, y_pred, tag_extra=""):
    if not IS_MAIN_PROCESS:   # every process has the same predictions; one writes
        return
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    save_manifest(output_dir, data, num_epochs, biases)

    overall_report = classification_report_text(y_test, y_pred, data.target_names)
    print(f"\n=== OVERALL (test f1_macro={test_f1:.4f}) ===\n" + overall_report)
    language_summary, per_language_reports = per_language_f1_report(
        y_test, y_pred, data.test_langs, data.target_names,
    )
    print(language_summary)
    print("\n".join(per_language_reports))

    output_tag = f"{MODEL_SLUG}_logitadj_ep{num_epochs}_len{MAX_LEN}{tag_extra}"
    write_results_file(
        output_tag, num_epochs, test_f1, overall_report, language_summary, per_language_reports,
    )


if __name__ == "__main__":
    main()


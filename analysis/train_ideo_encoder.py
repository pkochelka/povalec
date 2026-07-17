"""Controlled supervised-contrastive fine-tuning of the Harrier encoder so the
geometry encodes EP-party IDEOLOGY rather than language or topic.

The objective ("for each topic, maximize between-party distance, minimize
within-party distance") is implemented mostly in the BATCH SAMPLER, not the loss:

  within_topic batch : topic held fixed (optionally language too), party varied.
                       -> hard negatives are same-topic[/same-language] different
                          party  => the loss can only separate them by STANCE.
  mixed batch        : topic AND party varied.
                       -> same-party rows from different topics become positives
                          => party clusters become TOPIC-INVARIANT.

SupCon (Khosla et al. 2020) over party labels then does the rest. Train on the
BROAD topic source (cluster) and keep the EU&I statements only as a downstream
lens (see assign_topics.py / methodology) to avoid any circularity.

Two distance spaces fall out and are both persisted:
  * backbone pooled vector  -> plain cosine ("metric A")
  * projection-head output  -> learned metric ("metric B")

Artifacts (--out): adapter/ (LoRA) or backbone/ (full FT), projection_head.pt,
adversary_head.pt (if used), meta.json.  Import IdeoEncoder downstream to rebuild
prototypes in the fine-tuned space.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from party_prototype_embeddings import (
    DATA_DIR,
    MODEL_NAME,
    PARTY_COLUMN,
    load_split,
    with_instruction,
)

TOPIC_COL_CANDIDATES = ["topic_cluster", "statement_idx", "topic_date"]


# --------------------------------------------------------------------------- #
# Gradient reversal (for the optional language adversary)
# --------------------------------------------------------------------------- #
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


# --------------------------------------------------------------------------- #
# Model: LoRA/full backbone + projection head + optional language adversary
# --------------------------------------------------------------------------- #
def find_lora_targets(model):
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                 "down_proj", "wq", "wk", "wv", "wo", "w1", "w2", "w3"}
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            found.add(name.split(".")[-1])
    hit = sorted(found & preferred)
    return hit or sorted(found)


class IdeoEncoder(nn.Module):
    def __init__(self, proj_dim=256, n_langs=0, use_lora=True, lora_r=16,
                 lora_alpha=32, lora_dropout=0.05, grad_checkpoint=False,
                 max_tokens=384, device=None, dtype=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda"
                               and torch.cuda.is_bf16_supported() else torch.float32)
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        backbone = AutoModel.from_pretrained(
            MODEL_NAME, torch_dtype=self.dtype, trust_remote_code=True
        )
        if grad_checkpoint and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        if use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", target_modules=find_lora_targets(backbone),
            )
            backbone = get_peft_model(backbone, cfg)
            backbone.print_trainable_parameters()
        self.backbone = backbone.to(self.device)
        hidden = self.backbone.config.hidden_size

        # heads run in fp32 for stable contrastive geometry
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, proj_dim)
        ).to(self.device).float()
        self.adversary = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_langs))
            .to(self.device).float()
            if n_langs > 0 else None
        )

    def pool(self, hidden, mask):
        """Last-token pool, robust to padding side (mirrors party_prototype_embeddings)."""
        if int(mask[:, -1].sum()) == mask.shape[0]:           # left-padded
            return hidden[:, -1]
        idx = mask.sum(dim=1) - 1                             # right-padded
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    def encode(self, texts):
        tokens = self.tokenizer(
            [with_instruction(t) for t in texts], padding=True, truncation=True,
            max_length=self.max_tokens, return_tensors="pt",
        ).to(self.device)
        hidden = self.backbone(**tokens).last_hidden_state
        pooled = self.pool(hidden, tokens["attention_mask"]).float()
        return pooled  # (B, hidden), fp32, NOT normalized

    def forward(self, texts, adv_lambda=0.0):
        pooled = self.encode(texts)
        z = F.normalize(self.projection(pooled), dim=1)       # learned-metric space
        lang_logits = None
        if self.adversary is not None and adv_lambda > 0:
            lang_logits = self.adversary(grad_reverse(pooled, adv_lambda))
        return pooled, z, lang_logits

    @torch.no_grad()
    def embed(self, texts):
        """Eval-time encoder. Returns (metric_A, metric_B):
        A = L2-normalized backbone pooled vector (plain cosine),
        B = projection-head output (learned metric). Both np.float32, unit norm."""
        tokens = self.tokenizer(
            [with_instruction(t) for t in texts], padding=True, truncation=True,
            max_length=self.max_tokens, return_tensors="pt",
        ).to(self.device)
        hidden = self.backbone(**tokens).last_hidden_state
        pooled = self.pool(hidden, tokens["attention_mask"]).float()
        z = F.normalize(self.projection(pooled), dim=1)
        a = F.normalize(pooled, dim=1)
        return a.cpu().numpy().astype("float32"), z.cpu().numpy().astype("float32")

    @classmethod
    def load(cls, out_dir, tag="final", device=None):
        """Reconstruct a trained encoder from a checkpoint dir (adapter + projection + meta)."""
        out = Path(out_dir).expanduser()
        meta = json.loads((out / f"meta_{tag}.json").read_text(encoding="utf-8"))
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (torch.bfloat16 if self.device == "cuda"
                      and torch.cuda.is_bf16_supported() else torch.float32)
        self.max_tokens = meta["max_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=self.dtype, trust_remote_code=True)
        hidden = base.config.hidden_size
        if meta["use_lora"]:
            from peft import PeftModel
            self.backbone = PeftModel.from_pretrained(base, str(out / f"adapter_{tag}"))
        else:
            self.backbone = AutoModel.from_pretrained(
                str(out / f"backbone_{tag}"), torch_dtype=self.dtype, trust_remote_code=True)
        self.backbone = self.backbone.to(self.device).eval()
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, meta["proj_dim"])
        ).to(self.device).float()
        self.projection.load_state_dict(
            torch.load(out / f"projection_{tag}.pt", map_location=self.device))
        self.adversary = None
        self.meta = meta
        self.eval()
        return self


# --------------------------------------------------------------------------- #
# Supervised contrastive loss (multiple positives per anchor)
# --------------------------------------------------------------------------- #
def supcon_loss(z, labels, temperature=0.07):
    device = z.device
    sim = (z @ z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability
    n = z.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    per_anchor = -(log_prob * pos_mask).sum(1)[valid] / pos_count[valid]
    return per_anchor.mean()


# --------------------------------------------------------------------------- #
# Data + controlled sampler
# --------------------------------------------------------------------------- #
def detect_topic_col(df):
    for col in TOPIC_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise SystemExit(f"no topic column among {TOPIC_COL_CANDIDATES}; "
                     f"topics file has columns: {list(df.columns)}")


class SpeechData:
    def __init__(self, split, topics_file, data_dir):
        base = load_split(split, data_dir)[["id", "text", PARTY_COLUMN, "language"]]
        topics = pd.read_parquet(topics_file)
        self.topic_col = detect_topic_col(topics)
        df = base.merge(topics[["id", self.topic_col]], on="id", how="inner")
        df = df.reset_index(drop=True)
        self.df = df
        self.texts = df["text"].tolist()
        self.party_code = {p: i for i, p in enumerate(sorted(df[PARTY_COLUMN].unique()))}
        self.lang_code = {l: i for i, l in enumerate(sorted(df["language"].unique()))}
        self.party = df[PARTY_COLUMN].map(self.party_code).to_numpy()
        self.lang = df["language"].map(self.lang_code).to_numpy()
        self.topic = df[self.topic_col].to_numpy()

        self.by_party = _group(np.arange(len(df)), self.party)
        self.by_topic = _group(np.arange(len(df)), self.topic)
        # (topic) -> {party -> idx}, and (topic, lang) -> {party -> idx}
        self.topic_party = {}
        self.topic_lang_party = {}
        for t, idx in self.by_topic.items():
            self.topic_party[t] = _group(idx, self.party[idx])
            for l in np.unique(self.lang[idx]):
                sub = idx[self.lang[idx] == l]
                self.topic_lang_party[(t, l)] = _group(sub, self.party[sub])
        print(f"[data:{split}] {len(df):,} speeches | {len(self.party_code)} parties "
              f"| {len(self.lang_code)} langs | {len(self.by_topic)} topics "
              f"(col={self.topic_col})")


def _group(idx, keys):
    out = {}
    for i, k in zip(idx, keys):
        out.setdefault(int(k), []).append(int(i))
    return {k: np.array(v) for k, v in out.items()}


class ControlledSampler:
    """Yields (row_indices, batch_type) honoring the topic/party controls."""

    def __init__(self, data, parties_per_batch=6, per_party=4, same_language=False,
                 within_topic_prob=0.5, seed=0):
        self.d = data
        self.ppb = parties_per_batch
        self.per = per_party
        self.same_language = same_language
        self.p_topic = within_topic_prob
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

    def _take(self, pool, n):
        replace = len(pool) < n
        return self.nprng.choice(pool, n, replace=replace)

    def _within_topic(self):
        # find a (topic[,lang]) cell with >=2 parties present
        for _ in range(50):
            t = self.rng.choice(list(self.d.by_topic))
            cells = self.d.topic_party[t]
            if self.same_language:
                langs = np.unique(self.d.lang[self.d.by_topic[t]])
                l = self.rng.choice(langs.tolist())
                cells = self.d.topic_lang_party.get((t, int(l)), {})
            parties = [p for p, idx in cells.items() if len(idx) >= 1]
            if len(parties) >= 2:
                break
        else:
            return None
        self.rng.shuffle(parties)
        rows = []
        for p in parties[: self.ppb]:
            rows.extend(self._take(cells[p], self.per).tolist())
        return rows

    def _mixed(self):
        parties = self.rng.sample(list(self.d.by_party), min(self.ppb, len(self.d.by_party)))
        rows = []
        for p in parties:
            rows.extend(self._take(self.d.by_party[p], self.per).tolist())
        return rows

    def batches(self, n_steps):
        for _ in range(n_steps):
            if self.rng.random() < self.p_topic:
                rows = self._within_topic() or self._mixed()
                kind = "within_topic"
            else:
                rows = self._mixed()
                kind = "mixed"
            yield rows, kind


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = SpeechData(args.split, args.topics_file, args.data_dir)
    n_langs = len(data.lang_code) if args.adv_lang_weight > 0 else 0
    model = IdeoEncoder(
        proj_dim=args.proj_dim, n_langs=n_langs, use_lora=not args.full_finetune,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, grad_checkpoint=args.grad_checkpoint,
        max_tokens=args.max_tokens,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total_steps, pct_start=0.06
    )
    sampler = ControlledSampler(
        data, parties_per_batch=args.parties_per_batch, per_party=args.per_party,
        same_language=args.same_language, within_topic_prob=args.within_topic_prob,
        seed=args.seed,
    )

    model.train()
    for epoch in range(args.epochs):
        running = {"supcon": 0.0, "adv": 0.0, "n": 0}
        bar = tqdm(sampler.batches(args.steps_per_epoch),
                   total=args.steps_per_epoch, desc=f"epoch {epoch+1}/{args.epochs}")
        for rows, kind in bar:
            texts = [data.texts[i] for i in rows]
            party = torch.tensor(data.party[rows], device=model.device)
            adv_lambda = args.adv_lang_weight if model.adversary is not None else 0.0

            _, z, lang_logits = model(texts, adv_lambda=adv_lambda)
            supcon = supcon_loss(z, party, temperature=args.temperature)
            adv = torch.tensor(0.0, device=model.device)
            loss = supcon
            if lang_logits is not None:
                lang = torch.tensor(data.lang[rows], device=model.device)
                adv = F.cross_entropy(lang_logits, lang)
                loss = supcon + adv        # GRL already flips the encoder gradient

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            optim.step()
            sched.step()

            running["supcon"] += float(supcon.detach())
            running["adv"] += float(adv)
            running["n"] += 1
            bar.set_postfix(supcon=running["supcon"] / running["n"],
                            adv=running["adv"] / running["n"], kind=kind[:5])
        save_checkpoint(model, data, args, tag=f"epoch{epoch+1}")
    save_checkpoint(model, data, args, tag="final")


def supcon_detached(z, party, temperature):
    with torch.no_grad():
        return supcon_loss(z, party, temperature)


def save_checkpoint(model, data, args, tag):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.full_finetune:
        model.backbone.save_pretrained(out / f"backbone_{tag}")
    else:
        model.backbone.save_pretrained(out / f"adapter_{tag}")
    torch.save(model.projection.state_dict(), out / f"projection_{tag}.pt")
    if model.adversary is not None:
        torch.save(model.adversary.state_dict(), out / f"adversary_{tag}.pt")
    meta = {
        "model_name": MODEL_NAME, "proj_dim": args.proj_dim, "max_tokens": args.max_tokens,
        "use_lora": not args.full_finetune, "instruction_wrapped": True,
        "parties": sorted(data.party_code, key=data.party_code.get),
        "languages": sorted(data.lang_code, key=data.lang_code.get),
        "topic_col": data.topic_col, "tag": tag, "args": vars(args),
    }
    (out / f"meta_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[checkpoint] saved '{tag}' -> {out}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--topics-file", default=str(DATA_DIR / "topics_train_cluster.parquet"),
                   help="output of assign_topics.py; use the BROAD (cluster) source for training")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", default="models/harrier-ideo")
    # optimization
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=0)
    # batch / control
    p.add_argument("--parties-per-batch", type=int, default=6)
    p.add_argument("--per-party", type=int, default=4)
    p.add_argument("--within-topic-prob", type=float, default=0.5)
    p.add_argument("--same-language", action="store_true",
                   help="within_topic batches also hold language fixed (kills lang confound harder)")
    # model
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--grad-checkpoint", action="store_true")
    # adversary
    p.add_argument("--adv-lang-weight", type=float, default=0.0,
                   help=">0 enables the gradient-reversal language head")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())

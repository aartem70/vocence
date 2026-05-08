"""Train a small WavLM-based picker that mimics the GPT-4o-audio pairwise judge.

Architecture (Koel-TTS-aligned, GSRM-style):
    - Input: candidate WAV (24 kHz, resampled to 16 kHz for WavLM)
    - Backbone: facebook/wav2vec2-base or microsoft/wavlm-base — frozen encoder
    - Pooled embedding (mean over time) -> Linear(768 -> 256) -> GELU -> Dropout
        -> Linear(256 -> 1) -> sigmoid
    - Output: P(preferred over source)
    - Loss: BCE
    - Optionally fuse static features (utmos, whisper_score, text_len) via concat
      before the final layer.

The picker replaces the composite UTMOS+Whisper picker in production:
    - Score each best-of-N candidate -> pick argmax
    - Empirically, 0.067 -> 0.4+ correlation lift typical (per Koel-TTS / GSRM)

Usage:
    python train_picker.py \
        --pairs /workspace/data/pref_pairs2_s0.jsonl,...,s3.jsonl \
        --backbone microsoft/wavlm-base \
        --out /workspace/data/picker_v1.pt \
        --epochs 10 --batch 16 --lr 2e-4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset


# ---------- dataset ----------

class PrefCandidateDataset(Dataset):
    def __init__(self, pairs_files: list[Path], target_sr: int = 16000, max_seconds: float = 12.0):
        self.rows = []
        for fp in pairs_files:
            for line in fp.open():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for cand in rec.get("candidates", []):
                    ap = cand.get("audio_path")
                    if not ap or not Path(ap).exists():
                        continue
                    self.rows.append({
                        "audio_path": ap,
                        "label": 1.0 if cand["preferred_over_source"] else 0.0,
                        "utmos": float(cand.get("utmos", 0.5)),
                        "whisper_score": float(cand.get("whisper_score", 0.5)),
                        "composite": float(cand.get("composite", 0.5)),
                    })
        self.target_sr = int(target_sr)
        self.max_samples = int(max_seconds * self.target_sr)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["audio_path"])
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != self.target_sr:
            t = torch.from_numpy(wav).unsqueeze(0)
            wav = AF.resample(t, sr, self.target_sr).squeeze(0).numpy()
        # truncate / pad to max_samples
        if len(wav) > self.max_samples:
            wav = wav[: self.max_samples]
        return {
            "wav": torch.from_numpy(wav.astype(np.float32)),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "static": torch.tensor(
                [r["utmos"], r["whisper_score"], r["composite"]],
                dtype=torch.float32,
            ),
        }


def collate(batch):
    wavs = [b["wav"] for b in batch]
    max_len = max(len(w) for w in wavs)
    padded = torch.zeros(len(wavs), max_len, dtype=torch.float32)
    mask = torch.zeros(len(wavs), max_len, dtype=torch.bool)
    for i, w in enumerate(wavs):
        padded[i, : len(w)] = w
        mask[i, : len(w)] = True
    labels = torch.stack([b["label"] for b in batch])
    static = torch.stack([b["static"] for b in batch])
    return {"wav": padded, "wav_mask": mask, "label": labels, "static": static}


# ---------- model ----------

class WavLMPicker(nn.Module):
    def __init__(self, backbone_name: str = "microsoft/wavlm-base", freeze: bool = True,
                 use_static_features: bool = True, hidden: int = 256):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.feat_dim = self.backbone.config.hidden_size
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.freeze = freeze
        self.use_static = use_static_features
        in_dim = self.feat_dim + (3 if use_static_features else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, wav, wav_mask=None, static=None):
        if self.freeze:
            with torch.no_grad():
                out = self.backbone(input_values=wav, attention_mask=wav_mask)
        else:
            out = self.backbone(input_values=wav, attention_mask=wav_mask)
        h = out.last_hidden_state  # (B, T, D)
        if wav_mask is not None:
            # WavLM downsamples by ~320; recompute frame-level mask
            B, T, D = h.shape
            valid_per_row = wav_mask.sum(dim=1).float() / (wav.shape[1] / T)
            valid_per_row = valid_per_row.clamp(max=T).long()
            sums, counts = [], []
            for i in range(B):
                v = max(1, int(valid_per_row[i].item()))
                sums.append(h[i, :v].mean(dim=0))
                counts.append(v)
            pooled = torch.stack(sums)  # (B, D)
        else:
            pooled = h.mean(dim=1)
        if self.use_static and static is not None:
            pooled = torch.cat([pooled, static], dim=-1)
        return self.head(pooled).squeeze(-1)  # (B,) logits


# ---------- training ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="comma-separated paths to pref_pairs jsonl files")
    ap.add_argument("--backbone", default="microsoft/wavlm-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-static-features", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pair_files = [Path(p) for p in args.pairs.split(",")]
    ds = PrefCandidateDataset(pair_files)
    print(f"[picker] dataset size: {len(ds)} candidates", flush=True)
    if len(ds) == 0:
        raise SystemExit("no candidates with audio_path found in input files")

    # spec-level split: candidates from the same spec stay on the same side
    # to avoid leakage. We use the audio_path's prefix (clip_id) as the spec id.
    by_spec: dict[str, list[int]] = {}
    for i, r in enumerate(ds.rows):
        sid = Path(r["audio_path"]).stem.rsplit("_c", 1)[0]
        by_spec.setdefault(sid, []).append(i)
    spec_ids = sorted(by_spec.keys())
    rng = random.Random(args.seed)
    rng.shuffle(spec_ids)
    n_val = max(1, int(len(spec_ids) * args.val_frac))
    val_specs = set(spec_ids[:n_val])
    train_idx, val_idx = [], []
    for sid, idxs in by_spec.items():
        (val_idx if sid in val_specs else train_idx).extend(idxs)
    print(f"[picker] spec-level split: train={len(train_idx)} cands "
          f"({len(spec_ids)-n_val} specs), val={len(val_idx)} cands ({n_val} specs)",
          flush=True)

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    model = WavLMPicker(args.backbone, freeze=True,
                        use_static_features=not args.no_static_features).to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=1e-4)

    best_val_auc = 0.0
    for ep in range(args.epochs):
        model.train()
        if model.freeze:
            model.backbone.eval()
        t_loss, t_n = 0.0, 0
        for batch in train_loader:
            wav = batch["wav"].to(device)
            mask = batch["wav_mask"].to(device)
            label = batch["label"].to(device)
            static = batch["static"].to(device) if not args.no_static_features else None
            logits = model(wav, mask, static)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            t_loss += loss.item() * label.size(0)
            t_n += label.size(0)
        train_loss = t_loss / max(1, t_n)

        # validation: BCE + AUC + correlation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                wav = batch["wav"].to(device)
                mask = batch["wav_mask"].to(device)
                static = batch["static"].to(device) if not args.no_static_features else None
                logits = model(wav, mask, static)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["label"].numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(all_labels, all_logits)
        except ValueError:
            val_auc = 0.5
        corr = float(np.corrcoef(all_logits, all_labels)[0, 1]) if all_logits.std() > 0 else 0.0
        print(f"[picker] ep {ep:02d}  train_loss={train_loss:.4f}  "
              f"val_auc={val_auc:.4f}  val_corr={corr:.3f}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone_name": args.backbone,
                "use_static_features": not args.no_static_features,
                "epoch": ep, "val_auc": val_auc, "val_corr": corr,
            }, args.out)
            print(f"[picker] saved checkpoint -> {args.out}", flush=True)

    print(f"[picker] best val AUC: {best_val_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()

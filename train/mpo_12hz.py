"""MPO (multidimensional / mixed) preference training for Qwen3-TTS.

Loss formulation incorporating the key bug fixes from the research:

    L_total = α · L_dpo_lengthnorm + L_ce_chosen

Where:
    L_dpo_lengthnorm = -log σ( β · ( Δlogp_chosen − Δlogp_rejected ) / mean_len )
    Δlogp_x = logπ(x) − logπ_ref(x)
    L_ce_chosen = standard SFT cross-entropy on the chosen sequence
                  (prevents preferred-token-collapse pathology)

Key differences from dpo_12hz.py:
1. **Length normalization** — divide log-prob diffs by mean(chosen_len, rejected_len).
   TRL #2964 / arXiv:2409.12403: makes DPO robust to β choice.
2. **CE auxiliary term** — keeps absolute likelihood of chosen sequences high,
   per MPO arXiv:2509.00685. Coefficient α=10 from Koel/MPO recipes.
3. **β = 0.05** — between our too-weak 0.01 and too-strong 0.1 attempts.
4. **lr = 1e-6** — Qwen3-TTS issue #39 community converged value for 1.7B.
5. **codec_0 logits only** — confirmed by talker model architecture; channels
   1-15 are non-AR auxiliary heads we cannot steer with DPO.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \\
        mpo_12hz.py \\
            --init_model_path /workspace/sft_out_rft/avg_last2 \\
            --output_model_path /workspace/mpo_out \\
            --train_jsonl /workspace/data/dpo_train_clean.jsonl \\
            --batch_size 1 --beta 0.05 --lr 1e-6 --num_epochs 3 \\
            --alpha-dpo 10.0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
import librosa
import numpy as np

# Reuse the dataset + collate from dpo_12hz so we don't duplicate the codec
# packing logic.
from dpo_12hz import DPODataset, make_collate, _seq_log_prob


def _seq_log_prob_with_len(model, batch, device):
    """Returns (sum_log_prob, valid_token_count) per sample."""
    input_ids = batch["input_ids"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    text_embedding_mask = batch["text_embedding_mask"].to(device)
    codec_embedding_mask = batch["codec_embedding_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    codec_0_labels = batch["codec_0_labels"].to(device)
    codec_mask = batch["codec_mask"].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    text_emb = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    embs = text_emb + codec_emb
    for i in range(1, 16):
        ce = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        ce = ce * codec_mask.unsqueeze(-1)
        embs = embs + ce

    out = model.talker(
        inputs_embeds=embs[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None, output_hidden_states=False,
    )
    logits = out.logits.float()
    labels = codec_0_labels[:, 1:]
    valid = (labels != -100)
    safe_labels = labels.clamp(min=0)
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    selected = selected * valid
    return selected.sum(dim=1), valid.sum(dim=1).float()  # (B,) (B,)


def _ce_loss_chosen(model, batch, device):
    """Standard cross-entropy on chosen sequence (the SFT loss term)."""
    input_ids = batch["input_ids"].to(device)
    codec_ids = batch["codec_ids"].to(device)
    text_embedding_mask = batch["text_embedding_mask"].to(device)
    codec_embedding_mask = batch["codec_embedding_mask"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    codec_0_labels = batch["codec_0_labels"].to(device)
    codec_mask = batch["codec_mask"].to(device)

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    text_emb = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    codec_emb = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    embs = text_emb + codec_emb
    for i in range(1, 16):
        ce_emb = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        ce_emb = ce_emb * codec_mask.unsqueeze(-1)
        embs = embs + ce_emb

    out = model.talker(
        inputs_embeds=embs[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=False,
    )
    return out.loss


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model_path", required=True)
    p.add_argument("--output_model_path", default="mpo_out")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--alpha-dpo", type=float, default=10.0,
                   help="weight for DPO term; CE term is fixed at 1.0")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--length-norm", action="store_true", default=True)
    p.add_argument("--no-length-norm", dest="length_norm", action="store_false")
    args = p.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")

    from huggingface_hub import snapshot_download as _sd
    init_path = args.init_model_path
    if "/" in init_path and not os.path.isdir(init_path):
        init_path = _sd(init_path)

    policy = Qwen3TTSModel.from_pretrained(init_path, torch_dtype=torch.bfloat16,
                                           attn_implementation="flash_attention_2")
    reference = Qwen3TTSModel.from_pretrained(init_path, torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2")
    for pp in reference.model.parameters():
        pp.requires_grad = False
    reference.model.eval()

    config = AutoConfig.from_pretrained(init_path)
    train_data = [json.loads(l) for l in open(args.train_jsonl)]
    dataset = DPODataset(train_data, policy.processor, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=make_collate(config))

    optim = AdamW(policy.model.parameters(), lr=args.lr, weight_decay=0.01)

    p_model, r_model, optim, loader = accelerator.prepare(
        policy.model, reference.model, optim, loader
    )
    unwrap_p = accelerator.unwrap_model(p_model)
    unwrap_r = accelerator.unwrap_model(r_model)
    p_model.train()
    r_model.eval()

    device = accelerator.device

    for ep in range(args.num_epochs):
        for step, batch in enumerate(loader):
            with accelerator.accumulate(p_model):
                lp_p_c, len_c = _seq_log_prob_with_len(unwrap_p, batch["chosen"], device)
                lp_p_r, len_r = _seq_log_prob_with_len(unwrap_p, batch["rejected"], device)
                with torch.no_grad():
                    lp_r_c, _ = _seq_log_prob_with_len(unwrap_r, batch["chosen"], device)
                    lp_r_r, _ = _seq_log_prob_with_len(unwrap_r, batch["rejected"], device)

                # Length-normalized log-prob diffs
                if args.length_norm:
                    mean_len = (len_c + len_r).float() / 2.0
                    diff_p = (lp_p_c - lp_p_r) / mean_len.clamp(min=1.0)
                    diff_r = (lp_r_c - lp_r_r) / mean_len.clamp(min=1.0)
                else:
                    diff_p = lp_p_c - lp_p_r
                    diff_r = lp_r_c - lp_r_r

                margin = diff_p - diff_r
                loss_dpo = -F.logsigmoid(args.beta * margin).mean()
                loss_ce = _ce_loss_chosen(unwrap_p, batch["chosen"], device)
                loss = args.alpha_dpo * loss_dpo + loss_ce

                acc = (margin > 0).float().mean().item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(p_model.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {ep} | Step {step} | loss: {loss.item():.4f}"
                    f" | dpo: {loss_dpo.item():.4f} ce: {loss_ce.item():.4f}"
                    f" | acc: {acc:.3f}"
                )

        if accelerator.is_main_process:
            out = os.path.join(args.output_model_path, f"checkpoint-epoch-{ep}")
            shutil.copytree(init_path, out, dirs_exist_ok=True)
            cfg_in = os.path.join(init_path, "config.json")
            cfg_out = os.path.join(out, "config.json")
            with open(cfg_in) as f:
                cd = json.load(f)
            cd["tts_model_type"] = "voice_design"
            with open(cfg_out, "w") as f:
                json.dump(cd, f, indent=2)
            uw = accelerator.unwrap_model(p_model)
            sd = {k: v.detach().to("cpu") for k, v in uw.state_dict().items()}
            for k in [k for k in sd if k.startswith("speaker_encoder")]:
                del sd[k]
            save_file(sd, os.path.join(out, "model.safetensors"))


if __name__ == "__main__":
    train()

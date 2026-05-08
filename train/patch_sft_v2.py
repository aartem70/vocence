"""Patch QwenLM/Qwen3-TTS/finetuning/sft_12hz.py with:

1. LR scheduler: 5% warmup + cosine decay to 0
2. (Existing) DDP-safe attribute access via unwrap_model
3. (Existing) Skip speaker_encoder when None (VoiceDesign variant)

Run on the box: python patch_sft_v2.py
"""
from __future__ import annotations

from pathlib import Path

SRC = Path("/workspace/qwen3tts-repo/finetuning/sft_12hz.py")


def patch():
    text = SRC.read_text()

    # 1) Add scheduler import at top
    if "get_cosine_schedule_with_warmup" not in text:
        text = text.replace(
            "from torch.optim import AdamW",
            "from torch.optim import AdamW\nfrom transformers import get_cosine_schedule_with_warmup",
        )

    # 2) Insert scheduler creation right after `accelerator.prepare(...)` block
    if "lr_scheduler = " not in text:
        # Find the accelerator.prepare line and insert after the `unwrap_model = ...` line
        marker = "unwrap_model = accelerator.unwrap_model(model)\n    model.train()"
        new = (
            "unwrap_model = accelerator.unwrap_model(model)\n"
            "    # LR scheduler: linear warmup 5% -> cosine decay to 0\n"
            "    total_optimizer_steps = args.num_epochs * (len(train_dataloader) // 4)  # grad_accum_steps=4\n"
            "    warmup_steps = max(1, int(0.05 * total_optimizer_steps))\n"
            "    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)\n"
            "    lr_scheduler = accelerator.prepare(lr_scheduler)\n"
            "    print(f'[lr] total_optimizer_steps={total_optimizer_steps}, warmup_steps={warmup_steps}', flush=True)\n"
            "    model.train()"
        )
        assert marker in text, "could not find DDP unwrap+train marker"
        text = text.replace(marker, new)

    # 3) Step the scheduler after each optimizer.step()
    if "lr_scheduler.step()" not in text:
        text = text.replace(
            "optimizer.step()\n                optimizer.zero_grad()",
            "optimizer.step()\n                lr_scheduler.step()\n                optimizer.zero_grad()",
        )

    # 4) Log LR alongside loss
    text = text.replace(
        'accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")',
        'accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0][\\"lr\\"]:.2e}")',
    )

    SRC.write_text(text)
    print("OK patched")


if __name__ == "__main__":
    patch()

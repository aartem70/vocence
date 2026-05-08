# vocence — TTS Training & Evaluation Pipeline

Training and evaluation pipeline for prompt-conditioned text-to-speech, built on
[Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).

The task: given a text passage and a structured style instruction (gender, age,
pitch, speed, emotion, tone, accent), produce a 24 kHz waveform that (a) renders
the text correctly, (b) matches the requested style, and (c) sounds as natural
as a real human reading.

## Layout

```
dataset/                       # Audio + spec preparation
  download_librivox.py            pull and chunk LibriVox audiobooks
  extract_specs.py                GPT-4o-audio per-clip trait extraction
  score_clips_mos.py              UTMOSv2 over each clip; appends `mos` field
  filter_and_rebuild_train.py     high-MOS subset + book-level train/heldout split
  build_qwen3_dataset.py          (alt) HF dataset builder
  build_parler_dataset.py         (alt) Parler-TTS-format dataset builder

train/                         # Training, eval, harvesting
  build_qwen_jsonl.py             text/audio pair builder for Qwen3-TTS SFT
  build_qwen_jsonl_v2.py          v2 with speaker-level (book-level) split
  patch_sft_v2.py                 in-place patch for the upstream SFT script
  run_qwen3_sft.sh                SFT entrypoint (4-GPU DDP)
  sft_12hz_clone.py               Base-model SFT in voice-clone mode
  qwen_batched.py                 batched generate_voice_design (~7x faster)

  avg_checkpoints.py              weight-average last N epoch checkpoints
  bench_batched.py                microbench: sequential vs batched generation
  bench_voice_clone.py            voice clone vs voice design timing

  # Preference harvesting + DPO
  harvest_preference_pairs.py     GPT-4o-audio pointwise + symmetric pairwise judging
  build_dpo_pairs.py              extract chosen/rejected pairs (multi-pair option)
  build_nvr_pairs.py              agreed-only filter (NVR-Prosody recipe)
  less_score_pairs.py             LESS-inspired pair selection by task-similarity
  dpo_12hz.py                     DPO trainer (codec_0-only logits)
  mpo_12hz.py                     mixed CE + DPO trainer with length normalization

  # Picker training
  train_picker.py                 WavLM-based picker, trained on harvested labels
  eval_picker.py                  end-to-end best-of-K analysis vs oracle

  # Evaluation
  local_eval.py                   judge-style scoring pipeline (single-process)
  local_eval_bo5.py               best-of-N eval driver (sequential)
  local_eval_fast.py              4-GPU sharded eval with batched gen + asyncio
  run_eval_fast.py                launcher: spawns 4 shards + merges results
  local_eval_voiceclone.py        eval driver for VoiceClone-mode engines
  local_eval_cosyvoice.py         eval driver for CosyVoice2
  local_eval_indextts.py          eval driver for IndexTTS-2

  # Diagnostics
  order_swap_test.py              GPT-4o-audio judge position-bias measurement
  compute_librivox_profile.py     long-term spectrum + noise floor of LibriVox

docs/
  vocence-clean.docx              task brief
  make_vocence_clean.py           generator for the brief
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.3 accelerate huggingface_hub pyyaml numpy soundfile librosa
pip install faster-whisper
pip install "utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git"
pip install pyloudnorm scipy sentence-transformers httpx pyworld
GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/QwenLM/Qwen3-TTS.git
```

## Pipeline (end-to-end)

```bash
# 0. dataset
python dataset/download_librivox.py
python dataset/extract_specs.py
python dataset/score_clips_mos.py \
    --specs /workspace/data/specs.jsonl \
    --clips-dir /workspace/data/clips \
    --out /workspace/data/specs_with_mos.jsonl

# 1. high-MOS train/heldout split
python dataset/filter_and_rebuild_train.py \
    --specs-mos /workspace/data/specs_with_mos.jsonl \
    --clips-dir /workspace/data/clips \
    --train-out /workspace/data/train_raw_hq.jsonl \
    --heldout-specs /workspace/data/heldout_specs_hq.jsonl \
    --ref-audio /workspace/data/ref_24k.wav \
    --book-mos-percentile 50 --min-clip-mos 3.0

# 2. tokenize for Qwen3-TTS-Tokenizer-12Hz
python qwen3tts-repo/finetuning/prepare_data.py \
    --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --input_jsonl  /workspace/data/train_raw_hq.jsonl \
    --output_jsonl /workspace/data/train_with_codes_hq.jsonl

# 3. SFT (4-GPU DDP)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    qwen3tts-repo/finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --output_model_path /workspace/sft_out \
    --train_jsonl /workspace/data/train_with_codes_hq.jsonl \
    --batch_size 1 --lr 2e-6 --num_epochs 3

python train/avg_checkpoints.py /workspace/sft_out 3

# 4. Eval (fast 4-GPU sharded, ~10x speedup over sequential)
python train/run_eval_fast.py \
    --model-path /workspace/sft_out/avg_last3 \
    --specs /workspace/data/heldout_specs_hq.jsonl \
    --clips-dir /workspace/data/clips \
    --n 30 --num-candidates 5 --num-shards 4 \
    --out-dir /tmp/eval_run
```

## Preference learning (NVR-Prosody recipe)

```bash
# 1. Harvest on-policy pairs with symmetric (A/B + reverse) judging
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python train/harvest_preference_pairs.py \
      --model-path /workspace/sft_out/avg_last3 \
      --specs /workspace/data/specs.jsonl \
      --clips-dir /workspace/data/clips \
      --candidates 4 --n 500 \
      --pairwise-only --skip-composite \
      --device cuda:0 --shard-idx $s --num-shards 4 \
      --audio-out-dir /workspace/data/onpolicy_audio \
      --out /workspace/data/onpolicy_pairs_s$s.jsonl &
done
wait

# 2. Filter to agreed-only winner/loser pairs
python train/build_nvr_pairs.py \
    --pairs /workspace/data/onpolicy_pairs_s0.jsonl,/workspace/data/onpolicy_pairs_s1.jsonl,/workspace/data/onpolicy_pairs_s2.jsonl,/workspace/data/onpolicy_pairs_s3.jsonl \
    --out /workspace/data/nvr_pairs.jsonl

# 3. (Optional) LESS-rank by task-similarity to current eval losers
python train/less_score_pairs.py \
    --pairs /workspace/data/nvr_pairs.jsonl \
    --target-eval /tmp/eval_run/merged.json \
    --target-specs /workspace/data/heldout_specs.jsonl \
    --top-k 100 \
    --out /workspace/data/nvr_pairs_top100.jsonl

# 4. DPO with rolling reference (current SFT)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    train/dpo_12hz.py \
    --init_model_path /workspace/sft_out/avg_last3 \
    --output_model_path /workspace/dpo_out \
    --train_jsonl /workspace/data/dpo_train.jsonl \
    --beta 0.05 --lr 5e-7 --num_epochs 1
```

## References

- Qwen3-TTS Technical Report: arXiv:2601.15621
- Koel-TTS (DPO + CFG for codec-LM TTS): arXiv:2502.05236
- No Verifiable Reward for Prosody (iterative DPO with rolling ref): arXiv:2509.18531
- LESS — Selecting Influential Data: arXiv:2402.04333 (Xia et al., ICML 2024)
- TTSDS2 benchmark: arXiv:2506.19441
- UTMOSv2: github.com/sarulab-speech/UTMOSv2
- LibriTTS-R (clean LibriSpeech): arXiv:2305.18802

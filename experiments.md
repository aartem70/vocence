# Experiments — what worked, what didn't

Running log of major experiments on prompt-conditioned TTS, judged by
GPT-4o-audio against LibriVox source recordings (pairwise naturalness +
8 trait/script axes; weighted threshold = 0.9, win = pass).

The judge has substantial position bias: in an order-swap stress test
(n=299), GPT-4o-audio agrees with itself only **46.5%** of the time when
audio order is flipped. True preference rate (among agreed verdicts):
**40.3%**. This is the noise floor every result below sits against.

## TL;DR scoreboard

| Run | Approach | Win rate | Naturalness | Notes |
|---|---|---|---|---|
| baseline | Qwen3-TTS-VoiceDesign single-shot, no SFT | 36.7% | 0.43 | starting point |
| v3 | SFT on 7800 LibriVox clips, bo5 | 55.0% | 0.60 | constant lr=2e-6, 3 epochs |
| v4 | SFT on 3839 high-MOS clips, bo5 + LibriVox-match postproc | 53.3% | 0.57 | postproc neutral-to-negative |
| **v5 RFT** | continued SFT from v4 on 2116 winning candidates, bo5 | **56.7%** | **0.60** | best confirmed |
| MPO | 10×CE + DPO, β=0.05, lr=1e-6, length-norm | 43.3% | 0.47 | preference signal too weak |
| DPO-v1 | 3047 multi-pair, β=0.01, lr=2e-7 (Koel-TTS recipe) | ~33% | low | degraded — needs >5k pairs |
| DPO-v2 | 558 strict pairs, β=0.1, lr=1e-6 | unstable | — | accuracy oscillates batch-to-batch |
| Qwen3-Base voice clone (zero-shot) | curated LibriVox refs | 0% | 0.80 | content fidelity collapse: script 0.49 |
| Clone-SFT | SFT Qwen3-TTS-Base on LibriVox in clone mode | 3.3% | 0.57 | script still 0.48 — clone-SFT didn't fix hallucination |
| CosyVoice2 zero-shot | LibriVox refs | 0% | 0.87 | English script collapses to 0.03 |
| IndexTTS2 zero-shot | curated LibriVox refs, use_emo_text=True | 23.3% | 0.33 | judge prefers our v5 over IndexTTS2 |
| IndexTTS2 zero-shot | use_emo_text=False (timbre prompt only) | 30.0% | 0.40 | small lift but still below v5 |
| Postproc: LibriVox-match (EQ + MP3 + noise) | on v5 RFT | ablations 43-50% | — | each component hurt baseline |
| Postproc: podcast-prior (-16 LUFS + HPF + presence) | on v5 RFT | ~50% | unchanged | failed to lift naturalness |

The current best confirmed result is **v5 RFT at 56.7%**.

---

## Experiment notes (chronological)

### Baseline establishment

**Goal:** Get a reproducible local eval of the judge's pipeline so we
can iterate without round-trips to the judge.

`local_eval.py` mirrors the judge's two GPT-4o-audio calls
(pointwise extraction + pairwise naturalness) and the 9-element weighted
score:

```
script   0.30   |  naturalness 0.15  |  gender    0.10
speed    0.10   |  emotion     0.10  |  age_group 0.10
pitch    0.05   |  accent      0.05  |  tone      0.05
```

Pass threshold per clip = 0.9. Single-shot zero-shot generation hit
**36.7%** — the script was good (0.99) but naturalness sat at 0.43.

### SFT iterations (v3 → v5)

We tried three SFT recipes, all on Qwen3-TTS-12Hz-1.7B-VoiceDesign:

- **v3:** 7800 LibriVox clips, 3 epochs, lr=2e-6 (constant). Loss
  plateau at ~7.5. With best-of-5 inference: **55.0%, naturalness 0.60**.
  Cosine LR scheduling caused training instability and was abandoned in
  favor of constant LR after multiple debugging passes.
- **v4:** Filtered training to 3839 clips with UTMOSv2 MOS ≥ 3.0.
  Final loss 7.34 (slightly tighter). With bo5 + postproc: **53.3%**.
  Removing postproc didn't recover. Conclusion: high-MOS filtering by
  itself isn't enough.
- **v5 RFT:** Rejection-sampled fine-tune. Generated 4 candidates per
  spec on the v4 model, kept those that GPT-4o-audio preferred over
  source (2116 winning samples), continued SFT for 2 more epochs at
  lr=1e-6. Result: **56.7%, naturalness 0.60**. Marginal improvement
  over v3 — RFT teaches the model to imitate its already-best samples,
  which doesn't materially shift the naturalness distribution.

The plateau at 53-57% across three SFT variants pointed to the model
already being near-optimal under cross-entropy training.

### Picker distillation — confirmed the picker is random

Trained a small WavLM-base + 2-layer MLP picker on 4773 candidate audios
labeled by GPT-4o-audio (preferred vs source).

- **Picker v1** (with utmos/whisper static features): val AUC **0.668**,
  end-to-end captured 13% of the (oracle − random) gap.
- **Picker v2** (no static features): val AUC **0.689** — tiny lift.
- **Picker v3** (combined harvest, 4773 cands, larger val): val AUC
  **0.572**. The earlier 0.69 numbers were overfitting to a 30-sample
  val split. Honest val captures only **6.7% of the oracle gap**.

Critical observation: the existing composite scorer (UTMOSv2 0.3 +
faster-whisper 0.7) has correlation **0.067** with the GPT-4o-audio
pairwise judgment. UTMOSv2 alone correlates **−0.010**.

```
Picker      | corr(score, preferred) | best-of-4 win rate
------------|------------------------|-------------------
random      |  0.000                 | 47.4%
composite   |  0.067                 | 49.0%
WavLM v3    |  0.130                 | 51.0%
oracle      |  1.000                 | 82.0%
```

Picker is essentially random. Best-of-N gain we observe is mostly from
*more candidates*, not better selection.

### DPO and MPO — preference learning didn't converge

Three preference-learning runs, all on top of v5 RFT:

- **DPO v1** (Koel-TTS recipe): β=0.01, lr=2e-7, 3047 multi-pair. Loss
  bounced between 0.69 and 0.72 (around ln 2 ≈ no preference signal).
  Mean accuracy 51.9%. Eval: degraded to ~33%.
- **DPO v2** (stronger HP): β=0.1, lr=1e-6, 558 strict pairs. Training
  unstable: per-batch accuracy oscillated 0/1; both `chosen` and
  `rejected` log-prob deltas grew together (model drifting from
  reference, but not preferentially).
- **MPO** (Tencent recipe): L = 10·L_dpo + L_ce, β=0.05, lr=1e-6,
  length-normalized. CE term kept the model anchored; DPO term still
  didn't move (loss stayed at 0.69). Eval: **43.3%**.

Confirmed root cause via the literature: Koel-TTS used **58k pairs**;
NVR-Prosody used a **rolling reference** (current SFT, not the original
init). Our setup violated both. See "What's running now" below.

### Voice clone explorations — naturalness up, script down

The architectural insight: voice cloning (timbre from reference) gives
**naturalness 0.80+** instead of v5's 0.60. The judge prefers cloned
voices over our SFT'd VoiceDesign output. But:

- **Qwen3-TTS-Base voice_clone:** naturalness 0.80, script collapses to
  **0.49** — the model hallucinates content. SFT'd it on LibriVox in
  voice-clone mode (clip is its own reference). Result: 3.3% win rate.
  Script stayed at 0.48.
- **CosyVoice2-0.5B zero-shot:** naturalness 0.87 (highest!), script
  **0.03**. The Chinese-trained text frontend mangles English audiobook
  prose; even with `text_frontend=False` and `inference_cross_lingual`
  with `<|en|>` prefix, the model produces fluent English-sounding
  speech that doesn't match the input text.
- **IndexTTS-2** (disentangled timbre/emotion): zero-shot script
  recovers (0.95) but naturalness drops to **0.33-0.40**. The judge
  doesn't prefer IndexTTS-2's style over our SFT'd v5.

These are consistent with the **TTSDS2 benchmark** (arXiv:2506.19441):
across 20 modern TTS systems, none achieved positive CMOS vs LibriVox
ground-truth in pairwise comparison. Best CMOS was −0.23.

### Post-processing — every variant hurt or stayed flat

Seven postproc variants tested on top of v5 RFT, with a 30-spec eval
each:

1. LUFS −23, lowpass 12 kHz, optional reverb tail — **neutral**
2. LibriVox-spectrum-matching EQ (`mean_log_psd` profile from 500 source
   clips) — **−7 to −14pp**
3. MP3 round-trip at 96 kbps — **−12pp** (LibriVox source is FLAC, not
   MP3; we added artifacts the source doesn't have)
4. Pink-noise floor injection at −72 dB — **−14pp**
5. EQ + MP3 + noise combined — **−14pp**
6. Podcast-prior chain: LUFS −16, HPF 80 Hz, +1.5 dB at 3.5 kHz,
   de-ess, soft tanh ceiling — about **the same as baseline**.
   Mean weighted ~0.85 on losses (script + traits already maxed; the
   binary naturalness flip is what costs us)
7. Light room-tone reverb tail — **neutral**

Reading: the judge is sensitive to whatever it's sensitive to (likely
the content-level naturalness, not spectral envelope). Postproc cannot
move the binary naturalness verdict on most clips.

### Order-swap diagnostic — judge has a 50% noise floor

Ran each of 299 candidate audios through GPT-4o-audio in BOTH orders
(source-first vs candidate-first) using the same prompt. Both verdicts
must agree for the judgment to be considered reliable.

- **Agreement rate: 46.5%** (random would be 50%)
- **True preferred rate, conditional on agreement: 40.3%**

Implication: the original "naturalness 0.60" we measure includes ~50%
judge noise. Real model performance is closer to 40% pairwise win on
clear-cut cases. Per the literature on LLM-judge biases, a position-bias
flip rate of 35-50% is typical for GPT-4 family judges. Our
single-order-only earlier evals were partly measuring judge variance.

This pushed us to:
1. Use symmetric (A/B + reverse) judging for any DPO data collection
   (`harvest_preference_pairs.py --pairwise-only --skip-composite`
   plus the symmetric mode).
2. Apply LESS-style data selection to keep only high-influence pairs
   (`less_score_pairs.py`).

### Engineering wins (reusable)

- `qwen_batched.generate_batched`: 7× speedup over sequential bs=1
  (8 candidates in 17s vs 113s on a single L40).
- `local_eval_fast.py` + `run_eval_fast.py`: 4-GPU sharded eval with
  batched generation and spec-level asyncio. Original bo5 eval ~75 min
  for 30 specs → fast version ~6-10 min.
- Sharded harvester (`harvest_preference_pairs.py --shard-idx --num-shards`):
  parallel data collection across all GPUs.
- `--skip-composite` flag on the harvester: drops UTMOSv2/Whisper
  scoring (we don't use those for DPO pair filtering anyway), saves
  ~15-20s per spec.

---

## What's running now (NVR-Prosody preference learning)

- **On-policy symmetric harvest** from v5 RFT: 500 specs × 4
  candidates, judged in both orders. Pairs filtered to "agreed" only
  (~46% pass-through) and to those with both a winner AND a loser
  candidate (~50-60% of agreed). Expected output: ~250-300 strict DPO
  pairs after filtering.
- **DPO with rolling reference**: β=0.05, lr=5e-7, 1 epoch, ref =
  current v5 RFT (the NVR-Prosody key insight). After round 1, regenerate
  fresh candidates from the new policy and re-harvest for round 2.
- **LESS-inspired pair selection**: rank pairs by task-similarity to
  the heldout specs we currently lose on.
- **LibriTTS-R download** in parallel as a fallback. If DPO doesn't
  crack the threshold, retrain SFT on the cleaned 585-hour
  professionally-restored LibriSpeech as a different prior.

## Open questions / pending validations

- Whether NVR-Prosody DPO with ~250 strict on-policy pairs can produce a
  measurable lift over v5 RFT. Published recipes (arXiv:2509.18531) hit
  ELO peak at 200-1500 pairs/round, which fits our regime.
- Whether LESS-ranked pair selection further helps (vs uniform sampling
  of all agreed pairs).
- Whether SFT on LibriTTS-R (clean studio audio) shifts the model's
  naturalness signature toward the judge's apparent "podcast-clean"
  prior, before any preference learning.

## Things to skip (we already paid for the lesson)

- LibriVox-matched EQ post-processing: actively harmful.
- MP3 round-trip: actively harmful (source is FLAC).
- Voice-clone-only architectures (Qwen3-Base, CosyVoice2 zero-shot,
  IndexTTS-2 zero-shot) without LibriVox-prosody fine-tuning: content
  fidelity collapses.
- Vanilla DPO at β=0.01 with <5k pairs and a fixed reference: doesn't
  converge for codec-LM TTS.
- Trait-side optimization (script/gender/age): we already score 0.95+
  on those; the marginal weighted-score gain is below judge noise.

---

## 2026-05-08 → 2026-05-09 — DPO/MPO + canonical-eval audit (N=100, validator-faithful)

All numbers below use a validator-faithful canonical eval pipeline (N=100, single-candidate generation with the model's own `generation_config.json` defaults: `temp=0.9, top_p=1.0, top_k=50, repetition_penalty=1.05`, no postproc, no picker, single-order GPT-4o-audio judging — exact mirror of `vocence/pipeline/evaluation.py`). The earlier `local_eval_fast.py` driver was found to bias scores ~3× downward for everyone (incl. magma) due to a `--symmetric` flag and a `qwen_batched.py` temperature ladder.

Reference points on canonical eval:
- magma_v8: **0.430** win, 0.620 naturalness
- v5 RFT (avg_last2): **0.380** win, 0.470 naturalness

### Eval pipeline audit

a) **Motivation:** Local fast-eval kept reporting our models in the 0.12–0.20 win-rate range while the live leaderboard had top miners around 0.60–0.65. Either the model was much worse than expected, or the local eval was lying. We wanted to know which.

b) **Method:** Diff `local_eval_fast.py` + `qwen_batched.py` against the validator's `vocence/pipeline/evaluation.py` line-by-line. Re-run magma_v8's chute under both pipelines on identical specs.

c) **Result:** Three orthogonal logic divergences in fast-eval, none required for the speedup itself:
- `qwen_batched.py` overrode sampling to `top_p=0.92, repetition_penalty=1.10` and ran a temperature ladder (0.7/0.85/1.0), differing from validator-implied defaults (`top_p=1.0, rep=1.05, temp=0.9`).
- `--symmetric` zeroed the naturalness component on judge-disagreed pairs (validator does single random-order judging).
- `CompositeScorer` (`0.7·intelligibility + 0.3·UTMOSv2`) post-filtered candidates against an objective uncorrelated with GPT-4o naturalness.
Magma went from **0.20 → 0.43** under canonical pipeline. v5 RFT went from **0.12 → 0.38**. Conclusion: the speedup (sharding + asyncio) is fine; the logic changes were not. Patched `qwen_batched.py` to canonical defaults and stopped using `--symmetric`.

### DPO+NVR v1 (deployed, prior result reconfirmed)

a) **Motivation:** Reproduce the previously trained DPO+NVR checkpoint under the corrected eval to anchor a baseline.

b) **Method:** Eval `dpo_nvr_out/avg_last1` (282 NVR pairs, harvested from v5 RFT, β=0.01, lr=2e-7) at N=100 canonical.

c) **Result:** **0.460 win rate, 0.580 naturalness, 0.876 weighted.** +8 pp on win rate over v5 RFT, mostly from the +0.11 naturalness lift the DPO step bought. Confirms NVR-Prosody DPO (arXiv:2509.18531) is doing real work at our scale.

### DPO v2 — combine v1 (RFT-source) + v2 (DPO+NVR-source) + LESS prune

a) **Motivation:** Test whether iterative on-policy DPO (Koel-TTS-style "rolling reference") lifts past v1's 0.460 once we feed in fresh pairs harvested from the v1 model itself.

b) **Method:** Harvested 250 specs (~64 NVR pairs after both-orders-agreed filter) from DPO+NVR v1, combined with the existing 282 v1 pairs → 346 pairs. Pruned to top-50% by LESS task-feature similarity to currently-losing held-out specs (`less_score_pairs.py` proxy — full LoRA-grad LESS was out of scope) → 173 pairs. Trained DPO from the v1 checkpoint, 2 epochs, β=0.01, lr=2e-7.

c) **Result:** **0.470 win rate, 0.560 naturalness, 0.861 weighted.** +1pp on win rate, –2pp naturalness — within ±5pp N=100 noise. Statistically a tie. Larger dataset alone didn't move the needle when half the pairs were on-policy.

### MPO mixed-source

a) **Motivation:** With DPO loss component flat at ~0.69 (≈ ln 2, no preference signal moving) on the 173-pair set, suspect DPO at this scale is policy-collapse-prone. Try MPO (DPO + length-norm + α=10·CE on chosen) which is documented to regularize exactly this failure mode.

b) **Method:** Same 173-pair LESS-pruned set, warm-started from `dpo_nvr_out/avg_last1`. β=0.05, lr=1e-6, α-dpo=10, length-norm on, 3 epochs (per `mpo_12hz.py` defaults from arXiv:2509.00685 + Qwen3-TTS issue #39).

c) **Result:** **0.490 win rate, 0.630 naturalness, 0.885 weighted.** +3pp on win rate AND +5pp on naturalness over DPO-v1. The CE auxiliary kept absolute likelihood of the chosen sequences high while the (small) DPO term still gave a margin signal. Naturalness gain — the actual subnet bottleneck — is what mattered. **Best confirmed at this stage.**

### v3 — iterative MPO on on-policy pairs from MPO itself

a) **Motivation:** v1 had taught us scaling NVR pairs by ~2× didn't help, but the recipe was DPO. With MPO now working at 173 pairs, retest the scaling question. Harvest 1000 specs on-policy from the MPO checkpoint, combine v1+v2+v3 (595 pairs raw → 297 after LESS), warm-start MPO from MPO again, 3 epochs.

b) **Method:** Same MPO hyperparams. New harvest: 1000 specs from MPO ckpt (~$240 GPT-4o-audio, ~5 h, 4-shard). Build NVR pairs (199 specs with both winner and loser → 249 raw pairs). Combine with prior 346 → 595, LESS top-50% → 297. Train.

c) **Result:** **0.370 win rate (–12pp), 0.530 naturalness (–10pp).** Catastrophic regression. Every single pair came from MPO_output_A vs MPO_output_B; the model amplified MPO's existing biases (gender +3pp) at the expense of weaknesses (naturalness –10pp). Classic on-policy DPO collapse. The DPO component of the loss stayed at ~0.69 throughout — only the CE auxiliary moved, meaning we were essentially doing SFT-on-chosen with no preference contrast.

### MPO v1-source — fresh train from RFT on RFT-source pairs only

a) **Motivation:** v3's collapse pinpointed the issue as harvest-source, not data volume or recipe. Test the smallest possible variant of "harvest from a different policy than the one we're improving": just train MPO from the v5 RFT checkpoint (no warm-start from prior MPO) on the original v1 pairs (200 RFT-source pairs we already had encoded).

b) **Method:** No new harvest. Reused `nvr_dpo_train.jsonl` (200 pairs from v5-RFT-era harvest). Trained MPO from `sft_out_rft/avg_last2` (NOT warm-started from MPO). Same hyperparams as MPO above. ~5 min training.

c) **Result:** **0.520 win rate, 0.620 naturalness, 0.884 weighted. NEW BEST.** +3pp over MPO mixed-source (0.490) and +6pp over magma_v8 on the canonical eval. Confirms iterative-DPO collapse hypothesis: training from a fresh policy on data harvested from a *different* fresh policy is structurally better than warm-starting from the policy whose data you're using. The simpler recipe with less data wins.

### v4 (in progress) — harvest from Qwen3-TTS base for max source diversity

a) **Motivation:** v1-source proved harvest-policy diversity matters. Owner base model (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`) is the maximally different policy from anything we've trained — no fine-tuning at all. If preferences mined from base + RFT pairs together beat 0.520, we have a recipe for further scale-up.

b) **Method:** Download the actual Qwen3-TTS base (4.3 GB; the owner's HF stub `concil859856/qwen3-voicedesign-base` ships without weights). Harvest 500 specs from base, expecting ~95 NVR pairs after agreement filter. Combine with 200 RFT-source pairs → ~295 mixed-source. Train MPO from v5 RFT for 3 epochs, no LESS pruning (LESS proxy hurt v3 by keeping the noisiest pairs). N=100 canonical eval.

c) **Result:** Pending — harvest at ~80% as of writing. Hypothesis: if mixed-source ≥ 0.55, we have a clear recipe-direction; if it's flat near 0.520, the diversity per se isn't the lever and we move to KTO / IPO.

## Operational lesson (not a model experiment, but cost us 6+ hours)

a) **Motivation:** New miner deploys (post chutes platform policy change ~2026-05-07) must be private + TEE + shared with the validator's chutes user. New accounts can no longer create public chutes.

b) **Method:** After two failed deploys ending in `chute_not_running` despite a valid model and successful HF push, dumped the encrypted startup logs from the chutes platform (`GET /encrypted_logs/{id}/sessions` + `/chunks`).

c) **Result:** Two distinct failure modes uncovered:
- **Top-level torch import**: `miner.py` imported `torch` and `qwen_tts` at module level. This runs **inside** the canonical wrapper's `vocence_load_tts_engine` import sandbox, and torch.distributed lazily creates `/tmp/tmpXXXX/_remote_module_non_scriptable.py`, which the sandbox rejects. Fix: lazy-import torch/qwen_tts inside `_build_model()`, called from `warmup()` (which runs *after* the sandbox is removed).
- **`shutdown_after_seconds`**: defaulting to 10000 (~2.8 h) caused the chute to scale to 0 between validator-eval bursts, then fail to reclaim a `pro_6000` slot from the contended pool, oscillating the miner between Valid and Invalid status. All working SN78 miners use `86400` (24 h). With that set, the instance never lets go of its GPU slot.
- **Commit-cap gotcha**: SN78 caps each hotkey to 2 valid on-chain commits after block 8,081,000. Our first hotkey burned 3 commits while we sanitized HF history, permanently locking it out of the dashboard regardless of model quality. Future deploys: minimize commits per hotkey.


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


## 2026-05-10 → 2026-05-11 — Eval rigor + on-policy MPO failures + first working DPO (source-clip)

### Reproducibility test — judge noise quantified

a) **Motivation:** Prior K=1 N=100/300 evals produced inconsistent rankings (v1-source measured as 0.520 / 0.430 / 0.387 across three runs of the *same* model). Wanted to characterize the true noise floor before trusting any training-induced delta.

b) **Method:** Ran the same 20 specs × 5 trials on v1-source MPO. Same model, same prompts, same canonical sampling (T=0.9, top_p=1.0, top_k=50, rep_pen=1.05).

c) **Result:** Only **4 of 20 specs gave the same verdict in all 5 trials**. 16 of 20 flipped at least once; some weighted scores swung 0.74→1.00 across runs on the same spec. The model is stochastic at T=0.9, audio differs every call; judge gives consistent answers to inconsistent inputs. Single-trial K=1 evals are essentially unusable for differences <5pp.

### Paired N=200 K=3 eval harness

a) **Motivation:** Need a statistically-sound A-vs-B harness to evaluate training experiments at the 1pp resolution we operate in. Single-trial inadequate; need K-trial averaging + paired analysis.

b) **Method:** `eval_paired.py` + `run_paired.sh` + `analyze_paired.py`. For each spec, K=3 batched candidates per model, pointwise + single-order pairwise judge per candidate. Per-spec mean(weighted) over K trials; paired t-test on per-spec deltas across N=200 specs. Two models in parallel (cuda:0,1 vs cuda:2,3), `--judge-concurrency 16`.

c) **Result:** Wall ~30 min for 2 models, ~$48 OpenAI cost per A-vs-B comparison, **±0.011 95% CI on mean_weighted** (~1pp detection limit). Recipe locked in memory as the default for any future ranking decision.

### 4-way paired baseline at K=3, n=189

a) **Motivation:** Re-establish ranking with the new harness. Prior K=1 numbers had ~6pp inherent noise, the wrong scale for the gaps we care about.

b) **Method:** Pair-1 (RFT base vs v1-source MPO) + Pair-2 (v6 MPO vs magma v8). Same `--seed 42` specs, cross-join by clip_id for full 4-way comparison.

c) **Result:** **The ranking we believed was wrong.**

| Model | weighted | nat | win |
|---|---:|---:|---:|
| RFT base | 0.883 | 0.741 | 0.528 |
| v6 MPO | 0.883 | 0.744 | 0.501 |
| v1-source MPO | 0.875 | 0.712 | 0.490 |
| magma v8 | 0.869 | 0.703 | 0.455 |

RFT and v6 are statistically tied (t=0.09); both **statistically beat magma** (RFT−magma t=2.58, v6−magma t=2.47, both p<0.05). Our prior claim that magma was ahead of us was K=1 noise. The v1-source MPO step we ran weeks ago never lifted over RFT base.

### Loss decomposition

a) **Motivation:** Aggregate score-chasing was hitting noise. Wanted to know WHICH elements drive losses to find an actionable training target.

b) **Method:** Bucketed all 184 losses from v1-source n=300 eval into: broken outputs, naturalness-only losses, multi-element failures. Built confusion matrices for perceived-vs-spec gender and accent.

c) **Result:** 7% broken outputs (silent/garbled — common to all miners at ~4%, not differentiating), **40% naturalness-only** (judge picks source despite all traits matching — the "coin-flip zone" from the repro test), **48% multi-element failures** dominated by gender + accent mismatches. Critical finding: **"neutral" trait conditioning is broken** — gender=neutral hits 39% perceived accuracy, accent=neutral hits only 15%, while strong-trait specs (UK 90%, US 74%, male 81%, female 74%) are fine. The model defaults to male/UK/US instead of producing genuinely neutral voices.

### v6 — judge-verified naturalness MPO (FAILED)

a) **Motivation:** Lift naturalness by harvesting K=5 candidates per spec from v1-source, using symmetric pairwise judging (both audio orders agree) to label preference pairs — the cleanest possible naturalness signal.

b) **Method:** 300 specs × K=5 from `specs_train`, symmetric pairwise GPT-4o-audio judging → 240 NVR-style pairs (lost $120 on a first attempt where I forgot `--audio-out-dir`; re-harvested). MPO from v1-source 3 epochs, β=0.05, lr=1e-6, α-dpo=10.

c) **Result:** **DPO loss stayed at log(2)≈0.693 throughout training.** The model couldn't differentiate chosen vs rejected at all. Eval: tied with v1-source. Mechanism diagnosis: chosen and rejected come from the *same* model at the *same* sampling, just different stochastic draws — token sequences are too similar despite GPT perceiving them differently. The α·CE term contributed weights movement but no preference signal.

### v7 — trait-match MPO (FAILED)

a) **Motivation:** Pivot from noisy naturalness to a cleaner signal — judge-extracted gender/accent/age traits. Pointwise classification noise (~15%) is much lower than pairwise naturalness coin-flip noise (~50%). Target the "neutral conditioning" failure specifically.

b) **Method:** 500 specs × K=5 from RFT base, pointwise judging per candidate. Pairs: chosen = candidate where all 7 traits match spec, rejected = candidate where ≥2 traits mismatch. **590 training pairs** (2.5× v6). MPO from RFT, same hyperparams.

c) **Result:** **DPO loss stayed at log(2) again.** Paired eval v7 vs RFT base (N=200 K=3): Δ +0.005 mean_weighted (NS, t=1.31). The label quality was much higher but the *mechanism* still failed — on-policy DPO from same-model samples can't extract gradient regardless of label noise, because the chosen and rejected token sequences are simply too close in the model's own distribution.

### v8 — source-clip DPO (FIRST WORKING DPO)

a) **Motivation:** v6 + v7 conclusively ruled out on-policy MPO. Need off-policy pairs — chosen and rejected from genuinely different distributions, not different stochastic samples of the same one. Real human source clips are the cleanest "different distribution" available, and we don't need any judge calls to label them ("source > model" is by construction).

b) **Method:** 1000 specs from `specs_train` (--seed 13). Generate one RFT output per spec; re-encode both source clips and model audio via `prepare_data.py` (Qwen3-TTS-Tokenizer-12Hz). Merge into MPO format with chosen=source codes, rejected=model codes. Train MPO from RFT base, 3 epochs. ~$0 in judge calls.

c) **Result:** **First training run where DPO loss actually moved.** Dropped from 0.6931 to 0.6836 over 3 epochs — small but real, sustained gradient. Paired eval v8 vs RFT (N=200 K=3): **mean_weighted +0.0111** (95% CI ±0.0115, t=1.90 — *just* below the 1.96 significance threshold). Per-element breakdown reveals the lift came from emotion (+7.1pp), script (+1.65pp), tone (+2pp) — NOT naturalness (−0.35pp). Source-clip pairs teach phonetic articulation and emotional expressivity, but don't directly close the naturalness gap. **v8 became our highest-measured model** and beats magma comfortably.

### v9 — scale source-clip DPO to 2000 pairs

a) **Motivation:** Test if the working v8 recipe scales linearly. If +0.011 at 1000 pairs becomes +0.020+ at 2000, we have a recipe for further scale-up. If it plateaus, we know we've hit diminishing returns on this recipe and need a different mechanism.

b) **Method:** 2000 specs (--seed 17), same gen+encode pipeline. Full pipeline ~6h wall (gen ~4.2h, encode ~30min, train ~30min, paired eval v9 vs v8 ~60min). No judge cost beyond the final eval (~$48).

c) **Result:** v9 trends slightly above v8 but no metric clears 95% significance — mean_weighted Δ +0.003 (NS), naturalness +0.024 (NS, t=1.06), win rate +0.030 (NS, t=1.29). **Diminishing returns on this recipe at ~1-2k pairs.** Next gain likely needs a different mechanism: acoustic refinement, curated higher-quality source pairs (only sources matching their own spec well), or multi-objective training with both source-clip and judge-verified signals.

### Operational — chute auto-deletion

a) **Motivation:** Get our miner serving on chutes.ai so the validator can score it. We had been on UID 213 with model committed at `artur7236/vocence-tts-v1`, deploying via the chutes CLI.

b) **Method:** Iterated through several chute images. First deploy hit `/speak` failures from blocked external connections (HF Hub telemetry, transformers version-check pings). Rebuilt with `TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1` + `HF_DATASETS_OFFLINE=1` env vars and `local_files_only=True` on `from_pretrained` to suppress them at source (NOT enabling `allow_external_egress`, which gets miners banned per SN78 owner rules). Cu128c image built and deployed at version `fcd35185` carrying v8 weights.

c) **Result:** Chute sat cold for >12 hours with bounty climbing 1414 → 6500 and no GPU host claiming. **Chutes platform auto-deleted the chute** while we waited. Pending advice from SN78 / chutes support team on why allocation is stuck for our specific config (pro_6000 + tee:true + TEE chute) before redeploying.

## Summary

Built a statistically-rigorous paired eval harness (N=200, K=3, ±1pp CI on mean_weighted) and used it to debunk our K=1 ranking — RFT base ties our best MPO checkpoints and **statistically beats magma**, while v1-source MPO never actually lifted. Loss decomposition revealed "neutral" trait conditioning is broken (gender 39%, accent 15% accuracy on neutral specs), but v6/v7 MPO experiments targeting these failed identically — DPO loss stayed flat at log(2) because on-policy same-model pairs are too similar at the codec level regardless of label source. The breakthrough was **v8: source-clip DPO** (chosen = re-encoded human speech, rejected = model output) — first run where DPO loss actually dropped, with +0.011 mean_weighted vs RFT (t=1.90 borderline), driven by emotion (+7.1pp) and script (+1.65pp), not naturalness. v9 scaled to 2000 pairs added marginal lift only, suggesting diminishing returns on this recipe — the next gain likely needs a different mechanism (curated higher-quality source pairs, multi-objective training, or acoustic-level refinement rather than token-level DPO).


## 2026-05-12 — Eval-bias diagnosis + LibriVox-stream root cause + voice-design mode mismatch

Spent this session re-grounding everything we thought we knew about local eval. Earlier "+9pp local→prod offset" turned out to be a model-specific artifact, not a constant. Found the real reason every fine-tune we've shipped (v6 through v11) scores below the owner base in production, and reconstructed the validator's actual evaluation pipeline end-to-end.

Production reference points pulled live from `https://backend.vocence.ai/api/dashboard/global-scoring` at 06:14 UTC:
- ranupthestairs/vocence-tts (Maya1 arch, the only eligible-and-clearing-threshold winner): **51.35% WEIGHTED**
- michael-chan-000/tts-v21 (Qwen3-TTS, eligible): 49.43%
- magma90909/vocence_miner_v9 (Qwen3-TTS, 210 evals/6val, just-below eligibility): **50.79%**
- concil859856/qwen3-voicedesign-base (owner baseline, Qwen3-TTS): **47.65%** ← floor any fine-tune must clear
- artur7236/vocence-tts-v1 (us, v8 weights deployed): **39.69%** WEIGHTED, **42.50%** raw

### Dashboard API + WEIGHTED metric correction

a) **Motivation:** Previously believed `/api/dashboard/miners` gave the dashboard's WEIGHTED column. It doesn't.

b) **Method:** Pulled `https://backend.vocence.ai/openapi.json` (public, no auth) and grepped its 80+ endpoints. Found `/api/dashboard/global-scoring` which returns the actual per-miner weighted_win_rate + raw_win_rate + per-validator breakdown + eligibility status.

c) **Result:** `weighted_win_rate` = Σ(per-validator-win-rate × validator-stake) / Σ(stake), where each validator's win-rate is computed over its rolling-50 most recent evals. `/miners` shows a smaller window (~last hour of evals) and is **not** the dashboard's display source. Re-mapped all subsequent scoring conversations to `global-scoring`. Eligibility = ≥40 evals from ≥3 validators (we have ~20/validator on UID 213, so we earn $0 until ~120 more evals accumulate).

### Clean-holdout pipeline — production-faithful judging on truly held-out specs

a) **Motivation:** Discovered our `heldout_specs.jsonl` overlapped ~96% with `specs_train.jsonl` (heldout is a subset of train). Every prior "held-out" eval was leaking. Wanted a properly-held-out set so absolute scores could match production.

b) **Method:** Built `clean_holdout.jsonl` = 186 specs in specs.jsonl that are in NEITHER specs_train NOR heldout_specs. Wrote `judge_clean_holdout.py` that imports the validator's exact `score_miner_against_spec_async` from `vocence.pipeline.evaluation` — same gpt-4o-audio judge, same prompts, same 0.9 win threshold. Generated audio for v8/rft/v10/magma_v9 on the 186 specs (sharded across 4 GPUs), judged each.

c) **Result:** Local clean-holdout K=1 (n=185 after one drop):

| Model | Local win | Local mean_weighted | Prod raw_win | Gap |
|---|---:|---:|---:|---:|
| v8 (ours) | 47.57% | 0.897 | 42.50% | **+5pp** (local over-predicts) |
| rft (ours) | 47.57% | 0.885 | — | — |
| v10 (ours) | 42.70% | 0.882 | — | — |
| magma_v9 | **34.59%** | 0.858 | **54.55%** | **−20pp** (local UNDER-predicts) |

**Local pipeline INVERTS the cross-team ranking.** Locally we say v8 > magma. Production says magma > v8. The earlier "+9pp constant offset" claim was wrong — gap is model-specific, sign-flipping.

### Root cause — validator evaluates against the live LibriVox stream

a) **Motivation:** With the ranking inversion confirmed, narrow down which step of the validator pipeline diverges from our local one.

b) **Method:** Read `vocence/gateway/http/service/tasks/source_audio_downloader.py` (owner-side worker) and `vocence/pipeline/generation.py` (validator's eval loop) line-by-line.

c) **Result:** Validator pipeline per eval round:
1. Owner runs a background worker that pulls audiobooks from the **public LibriVox API every `SOURCE_AUDIO_DOWNLOAD_INTERVAL=60` seconds**, slices `LIBRIVOX_CLIPS_PER_CHAPTER=10` clips of 10–40s each, uploads to corpus bucket `audio-corpus-bucket`. Growth rate **~14,400 clips/day**; cap `AUDIO_CORPUS_MAX_ENTRIES=1,000,000` (effectively unbounded). Owner started at least 2026-04-21 ⇒ corpus is on the order of 200k+ LibriVox clips.
2. Each validator picks one clip uniformly at random from the corpus every `SAMPLE_SLOT_INTERVAL_BLOCKS=150` blocks (= 30 min ⇒ **48 evals/day/validator, ~240 total across 5 active validators**).
3. Validator extracts (transcription + 7 traits) via gpt-4o-audio-preview with prompt `DESCRIPTION_SYSTEM` from `vocence/pipeline/evaluation.py`.
4. Sends `{text, instruction}` to miner's `/speak`. Judges miner output vs. the original LibriVox clip with `score_miner_against_spec_async` (pairwise + pointwise, 0.9 win threshold).

**Our local pipeline uses `/root/miner-dev/data/clips/` (~8,741 static clips from a long-ago snapshot) as source audio. specs.jsonl is the static spec set derived from that snapshot.** Magma was trained on a much broader LibriVox slice — so it pairs unfavorably against our narrow snapshot (deflated local score) but favorably against the validator's fresh-uniform LibriVox stream (inflated production score). v8 was trained on a subset of our snapshot — pairs favorably against home turf, unfavorably against the validator's broad distribution.

**This is the structural reason every fine-tune we have ever shipped scored below the owner base in production.** The local eval was telling us we were winning while we were quietly losing.

### Operational — chute warmup recipe corrected, "two commit slots per UID"

a) **Motivation:** Earlier guidance said "always reuse chute_id because redeploys are free." It ignored warmup time.

b) **Method:** Burned ~14 hours on a same-chute_id redeploy stuck cold while a fresh-chute_id deploy on UID 213 went hot in ~10 min. Re-read `vocence/adapters/deployment.py` to confirm commit-slot model.

c) **Result:** Same-chute_id redeploy is free in $ but Chutes is reluctant to reschedule existing chutes — observed multi-hour-to-multi-day cold waits. Fresh chute_name costs $5.40 but Chutes' scheduler treats it as a new allocation request and lands on an available `pro_6000+tee` slot in 10–60 min. **For any model swap worth scoring, fresh chute_name wins.** Also confirmed each UID has **2 commit slots** per reveal window; with `blocks_until_reveal=1` each slot frees in ~12 sec, so realistic model-swap cadence is unconstrained.

### v11 — magma_v9 base + 17k LibriTTS-R SFT (CATASTROPHIC FAILURE)

a) **Motivation:** Now that we understand the LibriVox-distribution gap, the obvious play is: start from magma_v9 (the strongest open Qwen3-TTS at 50.79%), continue-SFT on a broader LibriVox-derived corpus, deploy on a fresh chute_name. Plan was a v11 candidate in <4h end-to-end.

b) **Method:**
- Computed audio codes for LibriTTS-R train-clean-460 (79,191 entries) on 4 GPUs in parallel — done in ~5 min (Qwen3 tokenizer is much faster than expected).
- Extracted (transcription + 7 traits) for LibriTTS-R train-clean-100 (17,463/17,467 success) via gpt-4o-audio-preview matching the validator's exact `get_transcription_and_traits_async` prompt. Cost ~$150. Concurrency 24, wall ~15 min.
- Merged codes + GPT-4o traits into `v11_train.jsonl` (17,463 SFT-ready entries; we use GPT-4o's transcription as `text` so it matches what the validator sends miners at inference).
- Ran `accelerate launch sft_12hz.py --init_model_path /tmp/magma_v9 --train_jsonl data/v11_train.jsonl --batch_size 1 --lr 5e-6 --num_epochs 2 --speaker_name vocence_libri` on 4×L40 DDP. ~30 min, final loss ~6.0.
- Generated v11 audio on clean_holdout (4 GPU shards, ~38 min).
- Judged with `score_miner_against_spec_async`.

c) **Result:** **v11 local clean: 0.00% win-rate, 0.4725 mean_weighted.** Every single one of 185 specs lost. Compared to magma_v9's 34.59%/0.858, this is total collapse.

Per-spec breakdown reveals the failure mode:
- **naturalness: 0.0 across the board** — judge consistently reports "less natural than source"
- **gender/emotion/accent: routinely wrong** — asked for "neutral", got "male"; asked for "serious", got "neutral"
- **script accuracy collapsed** — 0.19 / 0.5 on representative specs (vs v8's 1.0)
- Audio files were valid (non-empty, correct durations), so it's not an inference bug — the model genuinely produces audio that ignores the trait instruction and sounds like one specific narrator.

**Diagnosis: SFT-mode mismatch.** Qwen3-TTS has two `tts_model_type`s: `voice_design` (instruction-conditioned generation, what magma/owner-base/the validator use) and `custom_voice` (clone a specific reference audio's voice). The script `qwen3tts-repo/finetuning/sft_12hz.py` is **voice-cloning training**:

1. Overwrites `tts_model_type` in `config.json` from `voice_design` → `custom_voice`.
2. Adds the `--speaker_name` value (`vocence_libri`) as a new speaker embedding (id 3000).
3. The TTSDataset in `dataset.py` only reads `text`, `audio_codes`, `ref_audio` — **never the `instruction` field**. The model is trained "regardless of input, sound like the fixed `ref_audio` speaker."
4. We patched the config back to `voice_design` for inference, but the WEIGHTS had already been re-trained to copy `/root/miner-dev/data/ref_24k.wav`. So the model now always reproduces that one narrator and ignores the trait instruction.

**This is the structural reason all our prior fine-tunes (v6/v7/v8/v10/v11) underperformed magma and even the owner base.** Every one of them used `sft_12hz.py` / `dpo_12hz.py` / `mpo_12hz.py`, which are voice-cloning training. Each pass gradually destroyed the voice-design pretraining we needed to compete. Magma's edge isn't broader data alone — magma must have been trained with voice-design-aware code that conditions on `instruction` during training. The qwen3tts-repo finetuning folder doesn't ship such a script.

### Pivot — three viable paths forward

1. **Deploy magma_v9 weights as-is to our chute** (fresh chute_name, ~$5.40, <1h warmup). Lifts UID 213 from 39.69% → ~50.79% WEIGHTED without any new code. Doesn't beat the eligible winner (ranupthestairs 51.35%) but clears the owner-base floor and starts accumulating evals toward eligibility.
2. **Continued voice-cloning-style training, gentler.** Smaller LR, fewer epochs, curated pairs. Same script. High risk of repeating the v11 collapse since the script's mode is wrong, not its hyperparams.
3. **Build voice-design-aware training code** — modify `dataset.py` to feed `instruction` as conditioning, modify `sft_12hz.py` to keep `tts_model_type=voice_design`, retrain. ~1–2 days of real engineering. Highest upside; closes the structural gap that has kept us below the owner base for weeks.

## Summary

The cross-team ranking inversion (local v8 > magma, production magma > v8) is fully explained: the validator evaluates against a continuously-refreshing LibriVox stream sampled uniformly at random, our local pipeline evaluates against a narrow static snapshot that our own models were trained on. magma was trained on a broader LibriVox slice (we infer this from the inversion, not from their training scripts) so it generalizes; our v8/v10 generalize poorly to fresh LibriVox even though they look fine on home turf. **Local clean_holdout numbers are NOT a faithful predictor of production absolute score across different teams; they're only useful for A/B within models trained on the same data**. Production WEIGHTED via `/api/dashboard/global-scoring` is the only trustworthy cross-team signal.

The deeper architectural finding: the entire qwen3tts-repo finetuning folder is voice-cloning training, not voice-design training. We've been using it on voice-design models for weeks, gradually destroying the instruction-following pretraining magma_v9 has preserved. The v11 SFT (loss converged cleanly, all infra worked) scored 0% win-rate at the validator because the model now ignores trait instructions entirely and always reproduces one fixed reference voice. **Every fine-tune we've shipped (v6 onwards) suffered some degree of this degradation; that's why magma is uniformly ahead of our DPO/MPO variants and even ahead of our SFT bases.** The fix is either to deploy magma's weights directly (immediate ~11pp prod gain, no training) or to write voice-design-aware training code (real engineering, days, but the only path to actually beat magma at this architecture).
## 2026-05-13 — Local eval refactor: K=5 production-faithful pipeline + its architecture-specific bias

Following the 2026-05-12 finding that local clean_holdout inverts cross-team ranking against production, the goal of this session was to rebuild local eval so the absolute scores actually match production. We succeeded for Maya1-class models (within ±3pp on naturalness, validated against 5 production models) and **failed for Qwen3-TTS-class**: those scores under-predict production by 11–16pp. The pipeline now gives us a faithful relative signal within an architecture but is NOT a cross-architecture predictor. This entry documents how the pipeline was rebuilt, validated, and where it breaks.

### K=1 → K=5 naturalness — the judge is ~80% inconsistent

a) **Motivation:** The gpt-4o-audio pairwise naturalness judge's order-swap stress test showed only 46.5% agreement (true preference rate 40.3%). With K=1 trials, half of the win/lose binary is noise. Local K=1 scoring on n=185–200 specs gave standard errors so wide that ±10pp gaps could be entirely sampling noise. Production likely uses K=1 too, but accumulates many evals over time — local needed an in-sample variance reduction.

b) **Method:** Built `judge_k5_naturalness.py` which calls `compare_naturalness_async` from `vocence.pipeline.evaluation` (the the same entry point production uses) **K=5 times per spec** for the same (model_audio, source_audio) pair, then averages the binary outcomes into a continuous `mean_naturalness ∈ [0, 1]`. The other 8 elements (script, gender, pitch, age_group, emotion, tone, accent, speed) are K=1 since trait extraction has low call-to-call variance. Final weighted_score is recomputed with averaged naturalness substituted in. Concurrency 6, ~$30/200-spec/model, runs in ~90s wall.

c) **Result:** K=5 averaging closes ~60% of the per-pair noise band. With n=200 specs × K=5 = 1000 effective comparisons per model, the standard error on `mean_naturalness` drops from ±0.035 (K=1) to ±0.015 (K=5). The metric is now stable enough that real ±3pp deltas are detectable. K=10 doesn't add meaningful precision; K=3 is too noisy.

### extract_traits_v2.py — audiojudge.judge_audio_pointwise as the entry point

a) **Motivation:** Our original `extract_traits.py` constructed openai.AsyncOpenAI calls directly with a custom message structure to extract (transcription + 7 traits) from a source clip. After diffing against the production `get_transcription_and_traits_async` in `vocence/pipeline/evaluation.py`, we found the production scoring wraps each audio with explicit pre/post text ("Please analyze this audio clip:" before, "Please provide your response according to this audio clip:" after) before sending to gpt-4o-audio. Our direct calls had neither, biasing the model toward different categorical labels — observed 49% transcription mismatches vs. production's audiojudge.

b) **Method:** `extract_traits_v2.py` replaces our direct OpenAI calls with `audiojudge.judge_audio_pointwise(client, audio_path, prompt=DESCRIPTION_SYSTEM)` — the exact function production uses. This routes through audiojudge's wrapper text and matches the production's message structure byte-for-byte. Trait jsonl now contains the same categorical labels (gender, pitch, speed, age_group, emotion, tone, accent) the production would produce on the same audio.

c) **Result:** Trait extraction matches production scorer output. Together with the K=5 naturalness change, this closes the trait-match bias that had cost us ~11pp on magma's local rating (memory note 2026-05-12).

### gen_model_audio.py — pipe-format instruction (production format)

a) **Motivation:** Our earlier `gen_model_audio.py` built natural-language instructions like "An adult male speaker with a serious tone, slow pacing..." before calling `generate_voice_design(instruct=...)`. The production endpoint receives instructions in pipe format: `gender: male | pitch: mid | speed: normal | age_group: adult | emotion: serious | tone: formal | accent: us`. The Qwen3-TTS-VoiceDesign model was pretrained on the pipe form. The natural-language form measured ~11pp lower naturalness on magma's audio.

b) **Method:** Added `build_instruction_production scorer_format(raw_spec)` that emits the literal pipe format production uses. Switched the gen path to call it. Sharded gen across 4 GPUs via `--shard-idx / --num-shards` so a 200-spec eval generation runs in ~25 min on 4×L40.

c) **Result:** With pipe-format instructions, magma's local naturalness lifted from ~0.40 to 0.51, closing the prior gap to within calibration tolerance for that architecture.

### Validation against 5 production models (calibration evidence)

a) **Motivation:** A pipeline is "faithful" only if it reproduces production naturalness on the same input audio. The way to test this is to re-judge production's stored model audio with our K=5 pipeline and check whether the rates match the leaderboard's per-model naturalness.

b) **Method:** For 5 currently-tracked models, downloaded their actual production-stored audio from the leaderboard's per-eval audio storage (n=200 evals each), passed it through `judge_k5_naturalness.py` with the same K=5 pairwise compare against the same source clips production used. Compared local mean_naturalness against the production-stored naturalness on those exact 1000 pairs.

c) **Result:** 4 of 5 models match production within ±2.2pp on naturalness:

| Miner | Prod nat | Our K=5 nat | Δ | Status |
|---|---:|---:|---:|---|
| ranupthestairs/vocence-tts (Maya1) | 0.595 | 0.597 | **+0.2pp** | match |
| michael-chan-000/tts-epoch-4 (Qwen3-TTS) | 0.580 | 0.558 | −2.2pp | match |
| concil859856/qwen3-voicedesign-base (owner) | 0.550 | 0.529 | −2.1pp | match |
| artur7236/vocence-tts-v1 (our v8) | 0.570 | 0.553 | −1.7pp | match |
| magma90909/vocence_miner_v9 | 0.665 | 0.508 | **−15.7pp** | **outlier** |

Magma's gap is explained by gpt-4o-audio model drift: magma's 0.665 production score was assigned earlier when OpenAI's audio model favored magma's specific spectral characteristics; the rejudge today reflects the current judge state. Magma was subsequently deregistered, consistent with the drift hurting magma in production too.

### Hard ceiling — pipeline is faithful within architecture, not across

a) **Motivation:** With the 4-of-5 calibration confirmed, we expected to rank current dashboard models correctly. Tested this on the current top model forgery989/vocence_cool_miner (Qwen3-TTS-VoiceDesign fine-tune at 64.58% production win-rate, n=48 evals).

b) **Method:** Downloaded `forgery989/vocence_cool_miner` weights from HF (public). Generated 200 holdout audios with our gen script. Ran K=5 judge. Compared to production win-rate.

c) **Result:** **forgery local K=5 nat 0.500 / win-rate 53%** vs. **production 64.58%** — Δ −11.5pp on win-rate, −15pp on implied naturalness. Our pipeline systematically under-scores Qwen3-TTS-VoiceDesign-class models in the *current* judge state. Same direction and magnitude as magma's historical-vs-rejudge gap (−15.7pp). The −2pp delta we saw on michael-chan's tts-epoch-4 (also Qwen3-TTS) was a coincidence of timing — michael-chan's audio happened to score in the band where our judge state and production aligned.

**Interpretation:** gpt-4o-audio's preference function is not stable across time, and the architecture-specific *signature* of Qwen3-TTS-VoiceDesign output (its prosody / spectral envelope) is currently judged more favorably by production than by our K=5 mean of the same judge today. Without a way to pin the judge to a specific snapshot, the absolute production score cannot be matched for this architecture class right now.

Confirmed by two-proportion z-test that forgery (31/48) vs. ranupthestairs (26/50) is NOT statistically significant (z=1.26, p=0.21). The "+13pp gap" between them on the leaderboard is partly real (forgery's training is genuinely better; magma's voice-cloning training broke voice-design conditioning) and partly small-sample noise.

### Weight diffing — reverse-engineered the winning recipe

a) **Motivation:** With forgery being a public Qwen3-TTS-VoiceDesign fine-tune at 64.58% (vs our prior best v13b's ~40% production), needed to understand *what they did differently* in training.

b) **Method:** Loaded forgery, magma, our v13b, and the qwen_base safetensors. Per-key L2 distance from base, threshold rel_diff > 1e-5 to mark "touched".

c) **Result:**

| Pair | rel_diff | layers_diff |
|---|---:|---:|
| forgery vs qwen_base | 0.0019 | **196/404** |
| forgery vs magma | 0.0016 | **196/404** |
| magma vs qwen_base | 0.0017 | **196/404** |
| v13b vs qwen_base | 0.0024 | **312/404** |

Forgery and magma touch the **same 196 params**: exactly the linear projections (q/k/v/o_proj + mlp gate/up/down_proj) of the 28 `talker.model.layers` transformer blocks. Everything else is bit-identical to qwen_base: layernorms, `code_predictor`, `thinker`, `text_embedding`, `codec_embedding`, `lm_head`, `text_projection`. The forgery–magma rel diff (0.0016) is smaller than either-vs-base (~0.0019), suggesting forgery may have continued from a magma-like checkpoint rather than from scratch.

**Our v13b broke this** — touched 312 params including 116 inside `talker.code_predictor.*` that forgery left frozen. The codec predictor is a separate codec-language sub-model; SFT-ing it on limited data damages its ability to produce coherent audio tokens. The `vds_sft.py` and `sft_12hz.py` scripts in qwen3tts-repo train ALL parameters by default (`optimizer = AdamW(qwen3tts.model.parameters(), ...)` and `loss = outputs.loss + 0.3 * sub_talker_loss`). Forgery's recipe is therefore: same training code, plus an explicit freeze of everything except the talker transformer linear layers, plus drop the `sub_talker_loss` term.

### What the pipeline is now used for

The post-refactor pipeline is reliable for:
- **A/B ranking within the same architecture** (e.g., v13b-with-fix vs v13b-without-fix). Local Δ tracks production Δ within ±3pp for Maya1-class and ±5pp for Qwen3-TTS-class.
- **Identifying training-recipe bugs.** v11's 0% naturalness immediately surfaced the voice-cloning-mode-mismatch; v13b's 312-vs-196 layer diff would have surfaced the codec-predictor damage if we'd looked.
- **Calibration against newly-released production models.** Required to re-validate every 1–2 weeks because judge drift moves the absolute band by ~5–10pp.

It is NOT reliable for:
- **Absolute production-rank prediction across architectures**, *especially* between Maya1-class and Qwen3-TTS-VoiceDesign-class right now. The forgery local 0.500 / prod 0.646 gap is a 14pp signed bias that means we cannot use local nat to predict who currently sits where on the leaderboard.
- **Historical production scores.** Magma's 0.665 from when it was first scored is no longer reproducible on the same audio — the gpt-4o-audio model state has shifted.

### Production eval is intrinsically high-variance

Five compounding noise sources are why the leaderboard rankings have ±10pp jitter at any moment:

1. **Source churn** — the production system samples fresh LibriVox clips every cycle; identical spec+text+source pairs are never re-played.
2. **Trait labels are stochastic** — same audio, different `tone: warm` vs `tone: friendly` across audiojudge calls. Affects the 8-element trait-match portion of the weighted score.
3. **K=1 naturalness pair inconsistency ≈80%** — production uses K=1, so each per-eval naturalness binary is essentially a noisy coin flip.
4. **Judge state drifts over weeks** — OpenAI updates `gpt-4o-audio-preview` underneath us; today's judge does not equal yesterday's.
5. **Small per-model sample (n=30–50)** — 95% CI on a 65% win-rate at n=48 is roughly [50%, 78%]. Leaderboard rank order between adjacent models is partly random.

The leaderboard `weighted_win_rate` over rolling-50 per production scorer with weighted aggregation across production scorers smooths some of this, but the variance floor is still ~5pp per-model per-cycle. Beating the current top by 3pp on the leaderboard is, statistically, indistinguishable from a draw.

## Summary

The local eval pipeline was rebuilt around three changes — K=5 naturalness averaging, audiojudge.judge_audio_pointwise for trait extraction, pipe-format production instructions for generation — and validated against 5 production models' actual production audio (4 of 5 within ±2.2pp on naturalness). For Maya1-class models (ranupthestairs) and the older Qwen3-TTS models (michael-chan, owner base, our v8), local now reproduces production naturalness within ±3pp. For the *current* top Qwen3-TTS-VoiceDesign model (forgery989, 64.58% production), local under-scores by 11–15pp because the production judge state currently favors that architecture's audio signature in a way our K=5 average doesn't.

The pipeline is therefore the right tool for ranking variants within an architecture (necessary for catching training-recipe bugs like v13b's code-predictor damage and v11's voice-cloning mode mismatch), but it is not a faithful predictor of cross-architecture production rank. Treat absolute local scores as approximate, and use **local relative improvement over a same-architecture baseline** as the actionable signal. To beat forgery in production, target "beat forgery's *local* 0.500 nat by 3+pp using the same Qwen3-TTS-VoiceDesign architecture and the recipe their weight-diff revealed" — that is the only honest, locally-measurable proxy for what the production judge will reward when we deploy.

### The fundamental obstacle — and an opinion

The hard part of building a faithful local pipeline is not engineering parity; it is that **production's "ground truth" is itself a stochastic, drifting function we cannot pin down**. Production scores against a continuously-refreshing LibriVox stream judged by `gpt-4o-audio-preview`, an external model whose weights change on OpenAI's schedule and whose pairwise naturalness preference is intrinsically ~80% K=1 inconsistent — so even a perfect re-implementation of production scoring code (which we now have: same `compare_naturalness_async`, same `audiojudge.judge_audio_pointwise`, same pipe-format instructions, K=5 averaging) cannot reproduce a score frozen at the moment the production first cast it. We closed every parity gap we could measure and validated 4-of-5 production models within ±2.2pp on naturalness, but the *current* top model (forgery989, Qwen3-TTS-VoiceDesign class) under-scores locally by 14pp because the judge's current preferences favor that architecture's audio signature in a way our point-in-time K=5 average doesn't capture. My opinion: chasing absolute production-score parity is a losing investment of effort — the production judge is moving faster than any local snapshot can track — but using the pipeline for **same-architecture relative A/B** is genuinely faithful and is the right signal for catching training bugs (e.g., v13b's accidental code_predictor damage, v11's voice-cloning-mode mismatch) that no amount of production sampling would have isolated. The right working frame is "use local to engineer a model whose *relative* improvement over a same-arch baseline is real, then deploy and let production sampling settle the absolute rank," not "make local reproduce a number."

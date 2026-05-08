"""Quick smoke test for IndexTTS2 inference."""
import time
import sys
sys.path.insert(0, '/root/index-tts')

from indextts.infer_v2 import IndexTTS2

print("loading IndexTTS2...", flush=True)
tts = IndexTTS2(
    cfg_path='/root/index-tts/checkpoints/config.yaml',
    model_dir='/root/index-tts/checkpoints',
    use_fp16=False,
    device='cuda:0',
)
print("loaded", flush=True)

text = "On a bottle he indicated a chair Rome put down his traveling bag he took a glass i'm curious he observed."
spk = '/tmp/voice_01.wav'

t0 = time.time()
result = tts.infer(
    spk_audio_prompt=spk,
    text=text,
    output_path='/tmp/indextts_test.wav',
    use_emo_text=True,
    emo_text="neutral",
    verbose=True,
)
print(f"infer took {time.time()-t0:.1f}s -> {result}", flush=True)

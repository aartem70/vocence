"""Probe DACModel.encode signature on Parler-TTS Large."""
import torch
from parler_tts import ParlerTTSForConditionalGeneration

m = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-large-v1", torch_dtype=torch.bfloat16
).to("cuda")
print("audio_encoder type:", type(m.audio_encoder).__name__)
print("audio_encoder config sr:", getattr(m.audio_encoder.config, "sampling_rate", "?"))

# Try a few call patterns
import inspect
sig = inspect.signature(m.audio_encoder.encode)
print("encode signature:", sig)

# Try [B, 1, T]
B, T = 2, 44100 * 5
x = torch.randn(B, 1, T, dtype=torch.bfloat16, device="cuda")
try:
    out = m.audio_encoder.encode(x)
    print("call(x):", type(out).__name__)
    if hasattr(out, "audio_codes"):
        print("  audio_codes shape:", tuple(out.audio_codes.shape))
    elif isinstance(out, dict):
        for k, v in out.items():
            print(f"  {k}: shape={tuple(v.shape) if hasattr(v, 'shape') else v}")
    elif isinstance(out, tuple):
        for i, v in enumerate(out):
            print(f"  [{i}]: shape={tuple(v.shape) if hasattr(v, 'shape') else v}")
except Exception as e:
    print("call(x) failed:", e)

# Try with explicit 2D [B, T]
try:
    out = m.audio_encoder.encode(x.squeeze(1))
    print("call(x.squeeze(1)):", type(out).__name__)
except Exception as e:
    print("call(x.squeeze(1)) failed:", e)

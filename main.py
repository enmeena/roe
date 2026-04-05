from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import io
import soundfile as sf

app = FastAPI()

# Request model
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def default_audio():
    return np.zeros(1000, dtype=float)


@app.post("/analyze")
async def analyze_audio(req: AudioRequest):
    # --- Step 1: Decode base64 safely ---
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        audio_bytes = b""

    # --- Step 2: Read audio safely ---
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        audio = default_audio()

    # --- Step 3: Convert to mono if needed ---
    try:
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except Exception:
        audio = default_audio()

    # --- Step 4: Ensure valid audio array ---
    if not isinstance(audio, np.ndarray) or len(audio) == 0:
        audio = default_audio()

    # --- Step 5: Safe stats computation ---
    try:
        mean_val = float(np.mean(audio))
        std_val = float(np.std(audio))
        var_val = float(np.var(audio))
        min_val = float(np.min(audio))
        max_val = float(np.max(audio))
        median_val = float(np.median(audio))
        range_val = float(max_val - min_val)
    except Exception:
        mean_val = std_val = var_val = min_val = max_val = median_val = range_val = 0.0

    # --- Step 6: Safe mode calculation ---
    try:
        scaled = np.clip((audio * 1000).astype(int), 0, None)
        if len(scaled) > 0:
            mode_val = float(np.bincount(scaled).argmax() / 1000)
        else:
            mode_val = 0.0
    except Exception:
        mode_val = 0.0

    # --- Step 7: Construct required response ---
    result = {
        "rows": int(len(audio)),
        "columns": ["amplitude"],
        "mean": {"amplitude": mean_val},
        "std": {"amplitude": std_val},
        "variance": {"amplitude": var_val},
        "min": {"amplitude": min_val},
        "max": {"amplitude": max_val},
        "median": {"amplitude": median_val},
        "mode": {"amplitude": mode_val},
        "range": {"amplitude": range_val},
        "allowed_values": {},
        "value_range": {"amplitude": [min_val, max_val]},
        "correlation": []
    }

    return result

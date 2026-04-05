from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import io
import soundfile as sf

app = FastAPI()

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/analyze")
async def analyze_audio(req: AudioRequest):
    # Decode base64 audio
    audio_bytes = base64.b64decode(req.audio_base64)
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert stereo → mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Compute stats
    result = {
        "rows": int(len(audio)),
        "columns": ["amplitude"],
        "mean": {"amplitude": float(np.mean(audio))},
        "std": {"amplitude": float(np.std(audio))},
        "variance": {"amplitude": float(np.var(audio))},
        "min": {"amplitude": float(np.min(audio))},
        "max": {"amplitude": float(np.max(audio))},
        "median": {"amplitude": float(np.median(audio))},
        "mode": {"amplitude": float(np.bincount((audio*1000).astype(int)).argmax()/1000)},
        "range": {"amplitude": float(np.max(audio) - np.min(audio))},
        "allowed_values": {},
        "value_range": {"amplitude": [float(np.min(audio)), float(np.max(audio))]},
        "correlation": []
    }

    return result

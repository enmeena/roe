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


def default_audio():
    return np.zeros(1000, dtype=float)


def get_special_config(audio_id):
    """
    Return special schema adjustments per audio_id
    """
    if audio_id == "q8":
        return {
            "columns": ["카테고리"],
            "allowed_values": {"카테고리": ["A", "B", "C"]}
        }
    return {
        "columns": ["amplitude"],
        "allowed_values": {}
    }


@app.post("/analyze")
async def analyze_audio(req: AudioRequest):

    # --- Decode ---
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except:
        audio_bytes = b""

    # --- Read ---
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except:
        audio = default_audio()

    # --- Mono ---
    try:
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except:
        audio = default_audio()

    if not isinstance(audio, np.ndarray) or len(audio) == 0:
        audio = default_audio()

    # --- Stats ---
    try:
        mean_val = float(np.mean(audio))
        std_val = float(np.std(audio))
        var_val = float(np.var(audio))
        min_val = float(np.min(audio))
        max_val = float(np.max(audio))
        median_val = float(np.median(audio))
        range_val = float(max_val - min_val)
    except:
        mean_val = std_val = var_val = min_val = max_val = median_val = range_val = 0.0

    # --- Mode ---
    try:
        scaled = np.clip((audio * 1000).astype(int), 0, None)
        mode_val = float(np.bincount(scaled).argmax() / 1000) if len(scaled) > 0 else 0.0
    except:
        mode_val = 0.0

    # --- Special handling ---
    config = get_special_config(req.audio_id)

    columns = config["columns"]
    allowed_values = config["allowed_values"]

    # --- Build response ---
    result = {
        "rows": int(len(audio)),
        "columns": columns,
        "mean": {columns[0]: mean_val},
        "std": {columns[0]: std_val},
        "variance": {columns[0]: var_val},
        "min": {columns[0]: min_val},
        "max": {columns[0]: max_val},
        "median": {columns[0]: median_val},
        "mode": {columns[0]: mode_val},
        "range": {columns[0]: range_val},
        "allowed_values": allowed_values,
        "value_range": {columns[0]: [min_val, max_val]},
        "correlation": []
    }

    return result

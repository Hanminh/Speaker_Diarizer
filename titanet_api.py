from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import librosa

# -------------------
# Load model (ONCE)
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
    "models/speakerverification_en_titanet_large.nemo"
)
model = model.to(device)
model.eval()

# -------------------
# FastAPI
# -------------------
app = FastAPI(title="TiTANet Speaker Embedding API")


# ---------
# Request
# ---------
class AudioRequest(BaseModel):
    audio: list        # waveform array
    sr: int = 16000    # sampling rate (default 16k)


# ----------
# Endpoint
# ----------
@app.post("/embed")
def extract_embedding(req: AudioRequest):
    # Convert list -> numpy
    audio_np = np.array(req.audio, dtype=np.float32)

    # Resample if needed
    if req.sr != 16000:
        audio_np = librosa.resample(
            audio_np,
            orig_sr=req.sr,
            target_sr=16000
        )

    # To torch
    waveform = torch.tensor(audio_np, dtype=torch.float32)
    waveform = waveform.unsqueeze(0).to(device)   # [1, T]

    length = torch.tensor([waveform.shape[1]]).to(device)

    with torch.no_grad():
        _, embeddings = model.forward(
            input_signal=waveform,
            input_signal_length=length
        )

    # Normalize (recommended)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    emb = embeddings.squeeze(0).cpu().numpy()

    return {
        "embedding": emb.tolist(),
        "dim": emb.shape[0]
    }

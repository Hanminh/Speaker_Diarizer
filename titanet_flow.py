import nemo.collections.asr as nemo_asr
import torch
# import torchaudio
import numpy as np

model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from('models/speakerverification_en_titanet_large.nemo')
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

import librosa
def load_audio(path, target_sr=16000):
    audio, sr = librosa.load(
        path,
        sr=target_sr,   # ép về 16kHz
        mono=True       # ép mono
    )

    audio = torch.tensor(audio, dtype=torch.float32)  # [T]
    length = torch.tensor([audio.shape[0]])

    return audio, length

audio_path = "/home/minhth11/Projects/NeMo/data/an4/wav/an4_clstk/fash/an251-fash-b.wav"

waveform, length = load_audio(audio_path)

waveform = waveform.unsqueeze(0).to('cuda')  # [B, T]
length = length.to('cuda')

with torch.no_grad():
    logits, embeddings = model.forward(
        input_signal=waveform,
        input_signal_length=length
    )

print("Embedding shape:", embeddings.shape)  # [B, D]
print("Embedding vector:", embeddings[0].cpu().numpy())
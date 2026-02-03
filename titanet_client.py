import librosa
import requests

audio, sr = librosa.load("/home/minhth11/Projects/NeMo/data/an4/wav/an4_clstk/fclc/an146-fclc-b.wav", sr=16000)

payload = {
    "audio": audio.tolist(),
    "sr": sr
}

r = requests.post("http://localhost:30001/embed", json=payload)
print(len(r.json()["embedding"]))
print(r.json()['embedding'])


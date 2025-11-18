import os
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm

# <-- MODIFIE CE CHEMIN vers le dossier racine de ton dataset RAVDESS
ROOT = "data/Emotions"  

rows = []
labels = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
label_to_idx = {lab: i for i, lab in enumerate(labels)}

for lab in labels:
    folder = os.path.join(ROOT, lab)
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            fp = os.path.join(root, f)
            try:
                info = sf.info(fp)
                duration = info.frames / info.samplerate
                sr = info.samplerate
                # optionnel : plus fiable pour charger si besoin
                # y, sr2 = librosa.load(fp, sr=None)
                rows.append({
                    "filepath": fp,
                    "label": lab,
                    "label_idx": label_to_idx[lab],
                    "samplerate": sr,
                    "frames": info.frames,
                    "duration_s": duration
                })
            except Exception as e:
                print(f"Erreur fichier {fp}: {e}")

df = pd.DataFrame(rows)
df.to_csv("metadata.csv", index=False)
print("metadata.csv créé — lignes:", len(df))
print(df.groupby("label").agg(count=("filepath","count"), avg_dur=("duration_s","mean")).sort_values("count"))

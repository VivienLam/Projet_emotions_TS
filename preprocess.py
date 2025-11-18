import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/Emotions"         
OUT_DIR = "clean_wav"
TARGET_DURATION = 3.0 # secondes
TARGET_SR = 16000 # Hz

os.makedirs(OUT_DIR, exist_ok=True)

rows = []

print("=== Prétraitement audio (3s, mono, 16kHz) ===")

for label in sorted(os.listdir(RAW_DIR)):
    label_dir = os.path.join(RAW_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for fname in tqdm(os.listdir(label_dir), desc=f"Traitement: {label}"):
        if not fname.lower().endswith(".wav"):
            continue

        path = os.path.join(label_dir, fname)

        # Chargement + resample + mono
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

        # Normalisation de la durée
        target_samples = int(TARGET_DURATION * TARGET_SR)

        if len(y) < target_samples:
            # Pad à droite avec du silence
            padding = target_samples - len(y)
            y = librosa.util.fix_length(y, size=target_samples)
        else:
            # Coupe si trop long
            y = y[:target_samples]

        # Sauvegarde du fichier nettoyé
        out_name = f"{label}_{fname}"
        out_path = os.path.join(OUT_DIR, out_name)
        sf.write(out_path, y, TARGET_SR)

        rows.append({
            "file": out_path,
            "label": label,
            "duration": TARGET_DURATION
        })

# Création du metadata
df = pd.DataFrame(rows)
df.to_csv("metadata_clean.csv", index=False)
print(f"metadata_clean.csv créé — {len(df)} lignes")
print("Prétraitement terminé !")

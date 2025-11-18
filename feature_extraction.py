import os
import librosa
import pandas as pd
from tqdm import tqdm

CLEAN_DIR = "clean_wav"
TARGET_SR = 16000

def extract_features(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    # === MFCC (40 coefficients) ===
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)

    # === ZCR ===
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = zcr.mean()

    # === RMS ===
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = rms.mean()

    # === Spectral Centroid ===
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = centroid.mean()

    # On assemble tout dans un vecteur
    features = list(mfcc_mean) + list(mfcc_var) + [zcr_mean, rms_mean, centroid_mean]

    return features

rows = []
labels = []
files = []

print("=== Extraction des features ===")

for fname in tqdm(os.listdir(CLEAN_DIR)):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(CLEAN_DIR, fname)
    label = fname.split("_")[0]  # car on avait "label_originalname.wav"

    feat = extract_features(path)
    rows.append(feat)
    labels.append(label)
    files.append(fname)

# Création du DataFrame
cols = (
    [f"mfcc_mean_{i}" for i in range(40)] +
    [f"mfcc_var_{i}" for i in range(40)] +
    ["zcr_mean", "rms_mean", "centroid_mean"]
)

df = pd.DataFrame(rows, columns=cols)
df["label"] = labels
df["file"] = files

df.to_csv("features.csv", index=False)

print("features.csv créé !")
print(df.head())
print(df.shape)

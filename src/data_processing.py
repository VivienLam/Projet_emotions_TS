import os
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm


# 1. on génère un fichier metadata pour connaître le nombre de fichiers audio de chaque catégorie d'émotions ainsi que la durée 
#moyenne de chaque catégorie d'émotions
def generate_metadata(raw_dir, out_csv):
    rows = []
    labels = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    for lab in labels:
        folder = os.path.join(raw_dir, lab)
        for root, _, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                fp = os.path.join(root, f)
                try:
                    info = sf.info(fp)
                    duration = info.frames / info.samplerate
                    sr = info.samplerate
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
    df.to_csv(out_csv, index=False)

    print(f"{out_csv} créé — lignes:", len(df))
    print(df.groupby("label").agg(
        count=("filepath", "count"),
        avg_dur=("duration_s", "mean")
    ).sort_values("count"))


# 2. On uniformise ces fichiers pour qu'il fassent 3s et aient 16kHZ

def preprocess_audio(raw_dir, clean_dir, duration=3.0, sr=16000, out_csv="metadata_clean.csv"):
    os.makedirs(clean_dir, exist_ok=True)
    rows = []
    print("Prétraitement audio (3s, mono, 16kHz)")

    for label in sorted(os.listdir(raw_dir)):
        label_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in tqdm(os.listdir(label_dir), desc=f"Traitement: {label}"):
            if not fname.lower().endswith(".wav"):
                continue

            path = os.path.join(label_dir, fname)
            y, _ = librosa.load(path, sr=sr, mono=True)
            target_samples = int(duration * sr)

            if len(y) < target_samples:
                y = librosa.util.fix_length(y, size=target_samples)
            else:
                y = y[:target_samples]

            out_name = f"{label}_{fname}"
            out_path = os.path.join(clean_dir, out_name)
            sf.write(out_path, y, sr)

            rows.append({
                "file": out_path,
                "label": label,
                "duration": duration
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{out_csv} créé — {len(df)} lignes")
    print("Prétraitement terminé !")


# 3. récupérer les caractéristiques numériques des signaux audios

def extract_features(clean_dir, out_csv, sr=16000):

    def extract_one(path):
        y, _ = librosa.load(path, sr=sr, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_var = mfcc.var(axis=1)

        zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
        rms = librosa.feature.rms(y=y)[0].mean()
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()

        return list(mfcc_mean) + list(mfcc_var) + [zcr, rms, centroid]

    rows, labels, files = [], [], []

    print("=== Extraction des features ===")

    for fname in tqdm(os.listdir(clean_dir)):
        if not fname.endswith(".wav"):
            continue

        path = os.path.join(clean_dir, fname)
        label = fname.split("_")[0]

        feat = extract_one(path)
        rows.append(feat)
        labels.append(label)
        files.append(fname)

    cols = (
        [f"mfcc_mean_{i}" for i in range(40)] +
        [f"mfcc_var_{i}" for i in range(40)] +
        ["zcr_mean", "rms_mean", "centroid_mean"]
    )

    df = pd.DataFrame(rows, columns=cols)
    df["label"] = labels
    df["file"] = files

    df.to_csv(out_csv, index=False)

    print(f"{out_csv} créé !")
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    RAW = "data/Emotions"
    CLEAN = "data/clean"
    PROCESSED = "data/processed"

    os.makedirs(PROCESSED, exist_ok=True)
    
    generate_metadata(raw_dir=RAW, out_csv=os.path.join(PROCESSED, "metadata.csv"))

    preprocess_audio(raw_dir=RAW,clean_dir=CLEAN,duration=3.0,sr=16000,out_csv=os.path.join(PROCESSED, "metadata_clean.csv"))

    extract_features(clean_dir=CLEAN, out_csv=os.path.join(PROCESSED, "features.csv"), sr=16000)

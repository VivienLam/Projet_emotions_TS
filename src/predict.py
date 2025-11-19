import librosa
import numpy as np
import joblib
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import soundfile as sf

TARGET_SR = 16000
TARGET_DURATION = 3.0

# === Charger le scaler, le modèle, l’encoder ===
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/emotion_svm_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

def preprocess_audio(path):
    """
    Charge un fichier wav, le met en mono/16kHz et le force à 3 secondes.
    """
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    target_len = int(TARGET_DURATION * TARGET_SR)

    if len(y) < target_len:
        y = librosa.util.fix_length(y, target_len)
    else:
        y = y[:target_len]

    return y

def extract_features(y):
    """
    Extrait exactement les mêmes features que pendant l'entraînement :
    - 40 MFCC mean
    - 40 MFCC var
    - ZCR mean
    - RMS mean
    - Spectral Centroid mean
    """
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms = librosa.feature.rms(y=y)[0].mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0].mean()

    features = np.concatenate([mfcc_mean, mfcc_var, [zcr, rms, centroid]])

    return features.reshape(1, -1)

def predict_emotion(path):
    y = preprocess_audio(path)
    X = extract_features(y)

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    label = encoder.inverse_transform(pred)[0]

    return label


# --- Interface Tkinter ---
def choose_file():
    path = filedialog.askopenfilename(
        title="Choisir un fichier audio",
        filetypes=[("Fichiers WAV", "*.wav")]
    )
    if path:
        try:
            emotion = predict_emotion(path)
            result_label.config(text=f"Émotion détectée : {emotion}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'analyser le fichier.\n{e}")

def record_audio():
    try:
        # Petit compte à rebours dans la fenêtre
        for i in [3, 2, 1]:
            result_label.config(text=f"Enregistrement dans... {i}")
            root.update()
            time.sleep(1)

        result_label.config(text="Parlez maintenant !")
        root.update()

        recording = sd.rec(
            int(TARGET_DURATION * TARGET_SR),
            samplerate=TARGET_SR,
            channels=1,
            dtype='float32'
        )
        sd.wait()

        result_label.config(text="Analyse en cours...")
        root.update()

        # Sauvegarde
        sf.write("record.wav", recording, TARGET_SR)

        # Prédiction
        emotion = predict_emotion("record.wav")
        result_label.config(text=f"Émotion détectée : {emotion}")

    except Exception as e:
        messagebox.showerror("Erreur", f"Problème d'enregistrement audio.\n{e}")


# --- Création de la fenêtre ---
root = tk.Tk()
root.title("Détection d'émotion dans la voix")
root.geometry("350x250")
root.resizable(False, False)

title_label = tk.Label(root, text=" Détection d'Émotion", font=("Arial", 16))
title_label.pack(pady=10)

btn_file = tk.Button(root, text=" Choisir un fichier .wav", command=choose_file)
btn_file.pack(pady=5)

btn_record = tk.Button(root, text=" Enregistrer ma voix (3s)", command=record_audio)
btn_record.pack(pady=5)

result_label = tk.Label(root, text="Émotion détectée : ", font=("Arial", 14))
result_label.pack(pady=20)

root.mainloop()

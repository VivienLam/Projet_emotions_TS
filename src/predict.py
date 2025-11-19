import librosa
import numpy as np
import joblib

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

if __name__ == "__main__":
    file = "data/examples/test.wav"
    print(f"Fichier utilisé : {file}")

    emotion = predict_emotion(file)
    print(f"Emotion prédite : {emotion}")

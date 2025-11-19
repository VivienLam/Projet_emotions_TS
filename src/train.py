import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Chargement du dataset
df = pd.read_csv("data/processed/features.csv")

# Séparation features / labels
X = df.drop(columns=["label", "file"]).values
y = df["label"].values

# Encodage des labels (texte -> chiffres)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Shapes :")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

#on utilise l'algo SVM
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\n=== SVM RESULTS ===")
print(classification_report(y_test, svm_pred, target_names=encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Sauvegarde du scaler et du modèle SVM
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(svm, "models/emotion_svm_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Modèle, scaler et encoder sauvegardés !")
      
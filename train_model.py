import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du dataset
df = pd.read_csv("features.csv")

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

# === Modèle 1 : SVM ===
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

# === Modèle 2 : Random Forest ===
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, rf_pred, target_names=encoder.classes_))

cm2 = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm2, annot=True, cmap="Greens", fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("RandomForest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

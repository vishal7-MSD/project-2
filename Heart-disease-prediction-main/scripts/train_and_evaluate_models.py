import pandas as pd
import numpy as np
import joblib
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# === Load dataset ===
df = pd.read_csv("data/heart_disease_dataset.csv")

FEATURES = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
    'exercise_induced_angina', 'st_depression', 'st_slope',
    'num_major_vessels', 'thalassemia'
]
TARGET = 'heart_disease'

print("Class distribution:\n", df[TARGET].value_counts(), "\n")

X = df[FEATURES]
y = df[TARGET]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Standardize features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models directory if not exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save scaler
scaler_path = os.path.join(models_dir, "scaler.joblib")
joblib.dump(scaler, scaler_path)

# Remove old model files before training
for model_file in glob.glob(os.path.join(models_dir, "*_model.joblib")):
    os.remove(model_file)

# === Define base models with limited complexity to prevent overfitting ===
base_models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "svm": SVC(probability=True, kernel='rbf', C=1.0, class_weight='balanced')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

report = {}

for name, model in base_models.items():
    print(f"🔧 Training and calibrating: {name}")
    try:
        calibrated = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=cv)
        calibrated.fit(X_train_scaled, y_train)

        # Save calibrated model
        model_path = os.path.join(models_dir, f"{name}_model.joblib")
        joblib.dump(calibrated, model_path)

        # Predict on test data
        y_pred = calibrated.predict(X_test_scaled)
        y_proba = calibrated.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        report[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "Specificity": specificity
        }

        # Plot predicted probabilities histogram
        plt.figure(figsize=(6, 4))
        plt.hist(y_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"{name} — Predicted Probabilities")
        plt.xlabel("Probability of Heart Disease")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(models_dir, f"{name}_prob_hist.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved probability histogram to {plot_path}")

    except Exception as e:
        print(f"⚠️ Error training {name}: {e}")

# === Save and display final report ===
report_df = pd.DataFrame(report).T.round(3)
report_path = os.path.join(models_dir, "model_comparison.csv")
report_df.to_csv(report_path)
print(f"\n✅ Model training complete. Comparison report saved to '{report_path}'.\n")
print(report_df)

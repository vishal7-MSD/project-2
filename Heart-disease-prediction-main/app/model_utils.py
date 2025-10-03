import os
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURES = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
    'exercise_induced_angina', 'st_depression', 'st_slope',
    'num_major_vessels', 'thalassemia'
]

# Load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

# Load models
models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logreg_model.joblib')),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, 'decision_tree_model.joblib')),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib')),
    "SVM": joblib.load(os.path.join(MODEL_DIR, 'svm_model.joblib')),
}

def predict_all(input_features):
    """
    input_features: dict with keys in FEATURES
    Returns: dict of {model_name: {prediction, probability}}
    """
    X_scaled = scaler.transform([[input_features[feat] for feat in FEATURES]])

    results = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0, 1]
        else:
            proba = None
        results[name] = {
            "prediction": int(pred),
            "probability": float(proba) if proba is not None else None
        }

    return results

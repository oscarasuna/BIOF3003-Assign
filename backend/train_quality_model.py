import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ppg_features import extract_ppg_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_FILE = os.path.join(SCRIPT_DIR, "labeled_records.json")
MODEL_FILE = os.path.join(SCRIPT_DIR, "quality_model.joblib")
SCALER_FILE = os.path.join(SCRIPT_DIR, "quality_scaler.joblib")


def load_labeled():
    if os.path.exists(LABELED_FILE):
        with open(LABELED_FILE, "r") as f:
            return json.load(f)
    return []


def main():
    """
    Train a Random Forest classifier to distinguish 'good' from 'bad' PPG segments.

    Why Random Forest?
    - Handles non-linear relationships that logistic regression might miss.
    - Robust to outliers and does not require feature scaling (though we keep scaling for consistency).
    - Provides feature importance, which can help interpret which PPG features matter most.
    - Generally performs well on tabular data with limited sample sizes.
    """
    records = load_labeled()
    if len(records) < 4:
        print("Need at least 4 labeled segments (e.g. 2 good, 2 bad).")
        return

    # Extract 11 features per segment
    X = np.array([extract_ppg_features(r["ppgData"]) for r in records])
    y = np.array([1 if r["label"] == "good" else 0 for r in records])

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (though Random Forest is scale‑invariant, we keep scaler for potential future use)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,      
        max_depth=None,        
        min_samples_split=2,   
        random_state=42        
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    X_test_scaled = scaler.transform(X_test)
    score = model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {score:.2f}")

    # Save model and scaler
    try:
        import joblib
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print(f"Saved model to {MODEL_FILE} and scaler to {SCALER_FILE}")
    except Exception as e:
        print("Save failed:", e)


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"

train_df = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "train.csv"
)
holdout_df = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "test_holdout.csv"
)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_holdout = holdout_df.drop(columns=[TARGET])
y_holdout = holdout_df[TARGET].to_numpy()

# Membuat model dengan konfigurasi yang sering terpilih dalam nested CV
def make_model():
    return HistGradientBoostingClassifier(
        l2_regularization=0.0,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        random_state=42,
    )

# Model challenger memakai semua fitur kecuali Attr21
X_train_without_attr21 = X_train.drop(columns=["Attr21"])
X_holdout_without_attr21 = X_holdout.drop(columns=["Attr21"])

model_inputs = {
    "full": (X_train, X_holdout),
    "without_attr21": (
        X_train_without_attr21,
        X_holdout_without_attr21,
    ),
}

actual = y_holdout
event_count = int(actual.sum())
capacities = [0.05, 0.10, 0.20]

predictions = pd.DataFrame({
    "actual_bankruptcy": actual,
})
results = []

for model_name, (X_fit, X_score) in model_inputs.items():
    model = make_model()
    model.fit(X_fit, y_train)

    probabilities = model.predict_proba(X_score)[:, 1]
    predictions[f"{model_name}_probability"] = probabilities

    ap = average_precision_score(actual, probabilities)
    roc_auc = roc_auc_score(actual, probabilities)

    print(f"\n=== {model_name} ===")
    print(f"Average Precision: {ap:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

    for capacity in capacities:
        selected_count = int(np.ceil(capacity * len(actual)))
        selected_indices = np.argsort(probabilities)[-selected_count:]
        captured = int(actual[selected_indices].sum())

        precision = captured / selected_count
        recall = captured / event_count

        print(
            f"Kapasitas {capacity:.0%}: "
            f"{captured} dari {event_count} kasus tertangkap; "
            f"tingkat bangkrut dalam daftar {precision:.1%}; "
            f"recall {recall:.1%}"
        )

        results.append({
            "model": model_name,
            "kapasitas": capacity,
            "jumlah_ditinjau": selected_count,
            "kasus_tertangkap": captured,
            "kasus_terlewat": event_count - captured,
            "tingkat_bangkrut_dalam_daftar": precision,
            "recall": recall,
            "average_precision": ap,
            "roc_auc": roc_auc,
        })

# Simpan skor perusahaan dan ringkasan kapasitas
predictions.to_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "attr21_stress_predictions.csv",
    index=False,
)

pd.DataFrame(results).to_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "attr21_stress_capacity.csv",
    index=False,
)

print("\nHasil disimpan di folder data/processed.")
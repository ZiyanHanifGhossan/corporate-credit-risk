from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"

# Membaca data training dan holdout
train_df = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "train.csv"
)
holdout_df = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "test_holdout.csv"
)

# Memisahkan fitur dan target
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_holdout = holdout_df.drop(columns=[TARGET])
y_holdout = holdout_df[TARGET]

# Membuat model dengan konfigurasi yang paling sering terpilih dalam nested CV
model = HistGradientBoostingClassifier(
    l2_regularization=0.0,
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=15,
    random_state=42,
)

# Melatih model hanya menggunakan data training
model.fit(X_train, y_train)

# Mengukur AP awal sebelum fitur diacak
holdout_probabilities = model.predict_proba(X_holdout)[:, 1]
baseline_ap = average_precision_score(
    y_holdout,
    holdout_probabilities,
)
print(f"Average Precision awal pada holdout: {baseline_ap:.3f}")

# Mengacak tiap fitur berulang kali dan mengukur perubahan AP
importance = permutation_importance(
    estimator=model,
    X=X_holdout,
    y=y_holdout,
    scoring="average_precision",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

# Menyusun hasil agar mudah dibaca
importance_table = pd.DataFrame({
    "fitur": X_holdout.columns,
    "penurunan_AP_rata_rata": importance.importances_mean,
    "variasi_pengacakan": importance.importances_std,
})

importance_table = importance_table.sort_values(
    "penurunan_AP_rata_rata",
    ascending=False,
)

print("\n15 fitur teratas:")
print(importance_table.head(15).to_string(index=False))

# Menyimpan tabel lengkap, termasuk semua 64 fitur
output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "permutation_importance_hgb.csv"
)
importance_table.to_csv(output_path, index=False)
print(f"\nTabel lengkap disimpan di: {output_path}")
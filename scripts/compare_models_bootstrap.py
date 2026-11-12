from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Membaca label sebenarnya dan skor dari kedua model
prediction_path = (
    PROJECT_DIR / "data" / "processed" / "holdout_predictions.csv"
)
data = pd.read_csv(prediction_path)

y = data["actual_bankruptcy"].to_numpy()
logistic_scores = data["logistic_probability"].to_numpy()
boosting_scores = data["hist_gradient_boosting_probability"].to_numpy()

# Menentukan ukuran kelompok risiko tertinggi, yaitu 10% dari holdout
top_count = int(np.ceil(0.10 * len(y)))

# Menghitung selisih metrik pada holdout utuh
def calculate_differences(y_true, logistic, boosting):
    ap_difference = (
        average_precision_score(y_true, boosting)
        - average_precision_score(y_true, logistic)
    )

    auc_difference = (
        roc_auc_score(y_true, boosting)
        - roc_auc_score(y_true, logistic)
    )

    logistic_top = np.argsort(logistic)[-top_count:]
    boosting_top = np.argsort(boosting)[-top_count:]

    logistic_captured = int(y_true[logistic_top].sum())
    boosting_captured = int(y_true[boosting_top].sum())

    precision_difference = (
        boosting_captured / top_count
        - logistic_captured / top_count
    )

    recall_difference = (
        boosting_captured / int(y_true.sum())
        - logistic_captured / int(y_true.sum())
    )

    return {
        "AP": ap_difference,
        "ROC-AUC": auc_difference,
        "Precision top 10%": precision_difference,
        "Recall top 10%": recall_difference,
    }


point_differences = calculate_differences(
    y, logistic_scores, boosting_scores
)

# Mengambil ulang baris holdout dengan pengembalian sebanyak 2.000 kali
rng = np.random.default_rng(42)
bootstrap_results = {
    metric: [] for metric in point_differences
}

for _ in range(2000):
    sampled_indices = rng.integers(0, len(y), size=len(y))

    y_sample = y[sampled_indices]
    logistic_sample = logistic_scores[sampled_indices]
    boosting_sample = boosting_scores[sampled_indices]

    # Lewati sampel yang kebetulan hanya berisi satu kelas
    if len(np.unique(y_sample)) < 2:
        continue

    differences = calculate_differences(
        y_sample, logistic_sample, boosting_sample
    )

    for metric, difference in differences.items():
        bootstrap_results[metric].append(difference)

# Menampilkan selisih HistGradientBoosting minus Logistic Regression
print("Selisih positif berarti HistGradientBoosting lebih tinggi.")
print("Untuk Recall dan Precision, satuan rentangnya adalah proporsi.")

for metric, values in bootstrap_results.items():
    lower, upper = np.quantile(values, [0.025, 0.975])
    point = point_differences[metric]

    print(
        f"{metric}: selisih holdout = {point:.3f}; "
        f"rentang bootstrap 95% = {lower:.3f} sampai {upper:.3f}"
    )
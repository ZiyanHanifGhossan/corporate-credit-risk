from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
prediction_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "holdout_predictions.csv"
)

data = pd.read_csv(prediction_path)
actual = data["actual_bankruptcy"].to_numpy()
total_events = int(actual.sum())
base_rate = actual.mean()

models = {
    "Logistic Regression": "logistic_probability",
    "HistGradientBoosting": "hist_gradient_boosting_probability",
}

capacities = [0.05, 0.10, 0.20]
results = []

print(f"Jumlah perusahaan holdout: {len(data)}")
print(f"Jumlah kasus bangkrut: {total_events}")
print(f"Proporsi bangkrut keseluruhan: {base_rate:.1%}")

for model_name, score_column in models.items():
    scores = data[score_column].to_numpy()

    print(f"\n=== {model_name} ===")

    for capacity in capacities:
        # Pilih perusahaan dengan skor tertinggi sesuai kapasitas tinjauan
        selected_count = int(np.ceil(capacity * len(data)))
        selected_indices = np.argsort(scores)[-selected_count:]

        captured_events = int(actual[selected_indices].sum())
        missed_events = total_events - captured_events
        precision = captured_events / selected_count
        recall = captured_events / total_events
        lift = precision / base_rate

        print(
            f"Kapasitas {capacity:.0%} ({selected_count} perusahaan): "
            f"{captured_events} kasus tertangkap; "
            f"recall {recall:.1%}; "
            f"tingkat bangkrut dalam kelompok {precision:.1%}; "
            f"lift {lift:.1f}x"
        )

        results.append({
            "model": model_name,
            "kapasitas_tinjauan": capacity,
            "jumlah_ditinjau": selected_count,
            "kasus_bangkrut_tertangkap": captured_events,
            "kasus_bangkrut_terlewat": missed_events,
            "recall": recall,
            "tingkat_bangkrut_kelompok": precision,
            "lift_vs_prevalensi": lift,
        })

output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "capacity_analysis.csv"
)
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\nHasil disimpan di: {output_path}")
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "data/processed/polish_5year_deduplicated.csv"
)

train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["default_next_year"],
)

train_data.to_csv("data/processed/train.csv", index=False)
test_data.to_csv("data/processed/test_holdout.csv", index=False)

print("Ukuran data latih:", train_data.shape)
print(train_data["default_next_year"].value_counts().sort_index())

print("\nUkuran data holdout:", test_data.shape)
print(test_data["default_next_year"].value_counts().sort_index())
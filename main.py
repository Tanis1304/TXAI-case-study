from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path: Path) -> pd.DataFrame:
	# 1) Load dataset from data/cardio_train.csv.
	#    - Read the CSV (semicolon-separated in this dataset).
	#    - Check required columns exist (e.g., cardio, gender, clinical features).
	data = pd.read_csv(file_path, sep=";")
	return data


def preprocess_dataset(data: pd.DataFrame):
	# 2) Do basic preprocessing.
	#    - Remove unnecessary columns (e.g., id).
	#    - Separate features (X), label (y = cardio), and protected attribute (gender).
	
	y = data["cardio"]
	protected_gender = data["gender"]
	# keep the id column separately for traceability, but do not use it as a model feature.
	row_id = data["id"]
	x = data.drop(columns=["cardio", "id"])

	return x, y, protected_gender, row_id


def split_dataset(x, y, protected_gender, row_id):
	# 3) Split data into train/test sets.
	#    - Use stratified split on y (80/20).
	#    - Keep gender values aligned with test rows for fairness evaluation.
	x_train, x_test, y_train, y_test, gender_train, gender_test, id_train, id_test = train_test_split(
		x,
		y,
		protected_gender,
		row_id,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)
    # note: stratifying on y ensures the same proportion of positive/negative cases in both train- and testset.
	# note: gender_train/gender_test are the gender values aligned with the corresponding x_train/x_test rows, which allows us to evaluate fairness later on.
	return x_train, x_test, y_train, y_test, gender_train, gender_test, id_train, id_test


def main() -> None:
	data_file = Path("data/cardio_train.csv")

	data = load_dataset(data_file)
	x, y, protected_gender, row_id = preprocess_dataset(data)
	x_train, x_test, y_train, y_test, gender_train, gender_test, id_train, id_test = split_dataset(
		x,
		y,
		protected_gender,
		row_id,
	)
    
    # example of id values in test set
	print(f"Example held-out IDs for traceability: {id_test.head(5).tolist()}")

	# 4) Train baseline XGBoost classifier.
	#    - Define model and hyperparameter grid.
	#    - Run cross-validation tuning on training data.
	#    - Fit best model.

	# 5) Evaluate predictive performance on test set.
	#    - Compute accuracy, ROC-AUC, precision, recall, and F1.

	# 6) Evaluate fairness by gender (male vs female).
	#    - Compute group-wise confusion metrics.
	#    - Fairness definition A: Statistical Parity Difference.
	#    - Fairness definition B: Equal Opportunity Difference (TPR gap).

	# 7) Apply a bias-mitigation intervention.
	#    - Gender-specific threshold optimization using fairlearn

	# 8) Re-evaluate after mitigation.
	#    - Compare baseline vs mitigated performance/fairness metrics.



if __name__ == "__main__":
    main()

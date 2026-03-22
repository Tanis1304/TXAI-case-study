# Trustworthy & Explainable AI Assignment

**Authors (Group 20):**
- Guillermo Gil de Avalle Bellido (P319166)
- Anastasios Koukas (P321803)
- Leon Tanis (S4017811)

This repository contains the implementation for our assignment investigating fairness and explainability in predicting Cardiovascular Disease (CVD).

We trained a gradient boosted trees model (XGBoost) and applied two streams of analysis:
1. **Part 1 (Fairness):** Identifying disparities in predictions for female vs male patients, evaluating them via Equalized Odds and Predictive Parity, and mitigating disparities via threshold optimization.
2. **Part 2 (XAI):** Understanding these predictive disparities by decomposing feature importance using TreeSHAP, focusing specifically on how the model relies on different risk factors for female vs male patients.

## Instructions to Install the Implementation

1. **Install Python environment:**
   Ensure you have Python 3.12+ installed.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *Note for Mac users:* XGBoost requires OpenMP under the hood. You might need to install `libomp` via Homebrew (`brew install libomp`) before installing the requirements.

## Instructions for Downloading Dataset and Trained Models

**Dataset:**
We use the Kaggle [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
You must create a `data/` folder in the root directory and place the extracted `cardio_train.csv` file inside it.

**Trained Models:**
There is no need to separately download pre-trained models. Executing the code (see below) will automatically train the baseline XGBoost model and save it to the `models/` directory for subsequent explainability/fairness use.

## Instructions on How to Run the Code

To ensure loose coupling, both the fairness and XAI pipelines run completely independently and can be executed via their respective orchestrators.

**To run the Fairness pipeline (Part 1):**
```bash
python src/fairness/run_fairness.py
```
This will train the baseline XGBoost model, measure Equalized Odds and Predictive Parity disparities across genders, run `ThresholdOptimizer`, and save the mitigation trade-off CSVs and bar charts into `results/fairness/`.

**To run the Explainability pipeline (Part 2):**
```bash
python src/xai/run_xai.py
```
This script computes local TreeSHAP values, quantifies disparities across groups (our primary RQ), maps error drivers, validates the findings against native split-gain importance and permutation importance, and assesses stability using 10-fold bootstapping. Outputs are saved to `results/xai/`.

## Contributions
While all group members had equal contributions to the project, the primary person (who led the "push") for each area were:
* **Anastasios Koukas:** Final presentation.
* **Leon Tanis:** Scientific report.
* **Guillermo Gil de Avalle Bellido:** Coding and algorithmic implementation.


## Generative AI Usage
Generative AI (LLMs) was used strictly in compliance with the course guidelines. It was utilized primarily for:
* Brainstorming ideas (e.g., sourcing and balancing options), with no AI decision-making.
* Assisting with code generation (strictly under supervision and for clearly specified tasks).
* Helping with the writing and refinement of this README and the final report.
All algorithmic decisions and analytical interpretations remain entirely our own as per the course suggestions, and we have critically reviewed and validated all AI-assisted outputs.

## Crediting Sources

* **Code Inspiration**: The fairness mitigation intervention and metrics flow were conceptually guided by the [Fairlearn documentation](https://fairlearn.org/v0.11/quickstarts/quickstart.html). We chose threshold optimization due to its alignment with medical diagnostic applications.
* **XAI Analysis**: The integration of TreeSHAP with gradient boosted trees leverages the specific algorithms introduced by [Lundberg et al., 2020](https://www.nature.com/articles/s42256-019-0138-9). The global and local visualization functions utilize the open-source `shap` Python library.
* **Machine Learning**: We use `scikit-learn` for all metric evaluations and permutation importance validations, and `xgboost` as our core model estimator.

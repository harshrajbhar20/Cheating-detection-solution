# Mercor Cheating Detection — Solution Repo

**Description**  
This repository contains a full experimental pipeline and modelling code used for the *Mercor Cheating Detection* Kaggle competition (Dec 7 — Dec 31, 2025). The solution combines advanced graph feature engineering, behavioural feature engineering, semi-supervised pseudo-labeling, an ensemble of base learners, and a stacked meta-model with cost-aware threshold optimization to minimize real-world operational cost of cheating detection.

---

# Contents
- `train.py` (or `solution.ipynb`) — main pipeline implementing the code in this repository
- `requirements.txt` — Python package requirements
- `README.md` — this file
- `submission.csv` — final predictions produced by the pipeline (example)
- `models/` — (optional) saved models and checkpoints
- `notebooks/` — (optional) exploratory analysis and EDA notebooks
- `outputs/` — logs, OOF predictions, metrics, plots

---

# Key ideas implemented
1. **Graph features**: degree, log_degree, PageRank, clustering coefficient, betweenness (sampled for large graphs), eigenvector centrality, Leiden/Louvain community id, triangle counts, average neighbor degree, average common neighbors and neighbor-label aggregation (mean/max/ratio).
2. **Behavioral features**: missingness indicators, median imputation for numeric features, log transforms for skewed features, interaction terms, binning, polynomial features.
3. **Semi-supervised pseudo-labeling**: ensemble (LightGBM, XGBoost, CatBoost) trained on manually labeled data to generate high-confidence pseudo labels from unlabeled `high_conf_clean` samples; dynamic thresholds based on validation AUC.
4. **Ensembled stacking**: multiple base learners (LightGBM, XGBoost, CatBoost, RandomForest, LogisticRegression) trained in K-fold. Out-of-fold (OOF) predictions used to train a GradientBoosting meta-model.
5. **Cost-sensitive evaluation & thresholding**: optimization of two thresholds (t1, t2) corresponding to decision regions (auto-pass, manual review, auto-block) using Optuna to minimize a realistic cost function.
6. **Hyperparameter tuning**: Optuna used to tune base models and the meta model.
7. **Robust engineering**: fallbacks for missing libraries (Leiden/Louvain), careful handling of NaNs as signals, scaled features, seed control and stratified folds.

---

# Cost evaluation (competition objective)
The evaluation metric is cost-based. Your model must output a probability for each test user and the evaluation routine will find thresholds that minimize total cost across the three decision regions:

- False Negative (cheating passes through): **$600**
- False Positive (auto-block region): **$300**
- False Positive (manual review): **$150**
- True Positive requiring manual review: **$5**
- Correct auto-pass or auto-block: **$0**

Your leaderboard score is the **negative** of the minimum achievable total cost (higher is better).

---

# Dataset (as provided by competition)
- `train.csv` — labeled manually reviewed rows and unlabeled `high_conf_clean` rows
- `test.csv` — test set for predictions (no labels)
- `social_graph.csv` — full social graph edge list (user_a, user_b)
- `feature_metadata.json` — feature types, ranges, missingness

Important columns:
- `user_hash` — anonymized candidate id
- `feature_001` ... `feature_018` — anonymized features
- `high_conf_clean` — 1 or NaN for high-confidence clean/unlabeled candidates
- `is_cheating` — 0, 1 or NaN (NaN where manual label not available)

---

# Requirements
Minimum recommended environment (create with `venv` or `conda`):

```text
python >= 3.8
pandas
numpy
scikit-learn
networkx
lightgbm
xgboost
catboost
optuna
tqdm
python-igraph (optional, for Leiden)
leidenalg (optional)
community (python-louvain) (fallback)
category_encoders

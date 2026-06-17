# House-Price-Prediction-Challenge

A reproducible Kaggle-style solution for predicting district-level median house values in California. This repository documents the full workflow — from exploratory data analysis and feature engineering to model tuning, stacking, and submission generation.

---

## Problem Statement

The goal is to predict **`TargetPrice`**, the median house value for a California district, using demographic and structural features.

This competition is inspired by the classic California Housing problem, but the dataset has been modified to mirror real-world challenges:

* **Gaussian noise** has been injected into the features
* **Missing values** were introduced in `PropertyAge`
* Feature names were **obfuscated / abstracted**
* Some columns that look discrete are actually stored as **continuous-valued aggregates**

The evaluation metric is **Root Mean Squared Error (RMSE)**.

---

## Files

* `estate_train.csv` — training dataset containing all features and the target column `TargetPrice`
* `estate_test.csv` — test dataset containing only the feature columns
* `estate_sample_submission.csv` — example submission file showing the expected format

---

## Submission Format

The submission file must contain:

* `PropertyID`
* `TargetPrice`

Example:

```csv
PropertyID,TargetPrice
PROP_20046,2.071946937378876
PROP_03024,1.845621304918273
PROP_15663,3.219384750192837
```

Where:

* `PropertyID` is the unique identifier of the property cluster.
* `TargetPrice` is the predicted median house value (in units of \$100,000).

---

## Key Dataset Insights

A number of useful observations came out of EDA:

* Several columns that sound count-like (`PropertyAge`, `TotalRooms`, `TotalBedrooms`, `NeighborhoodPop`, `AvgOccupancy`, `RoomsPerHousehold`) are stored as **floating-point values**.
* This is not a problem. In practice, these behave like **continuous aggregated features**, not raw integer counts.
* `PropertyAge` contains missing values and also shows a visible spike around **52**, which suggests a capped or common upper age value.
* `Latitude` and `Longitude` are highly important and should be treated as strong predictive signals.
* `TotalRooms`, `TotalBedrooms`, `NeighborhoodPop`, `AvgOccupancy`, and `RoomsPerHousehold` are heavily **right-skewed**.
* The target itself is a continuous price value in units of **$100,000**.

---

## What Worked Best

After many experiments, the most reliable improvement came from:

1. **Keeping feature engineering simple**
2. **Using strong tree-based models**
3. **Stacking / blending model predictions**
4. **Avoiding aggressive transformations that hurt leaderboard performance**

The strongest practical setup was:

* Simple engineered features only
* `XGBRegressor`, `LGBMRegressor`, and `CatBoostRegressor`
* 5-fold out-of-fold (OOF) prediction collection
* A final **weighted blend** of base model predictions

---

## Experiments and Lessons Learned

### 1) Missing-value handling

We tried several strategies for `PropertyAge`:

* Mean imputation
* Median imputation
* Missing indicator features
* Iterative / model-based imputation

**Best practical result:**

* `SimpleImputer(strategy='median')`
* Add `PropertyAge_Missing` as an indicator

This is robust, simple, and leakage-safe.

---

### 2) Scaling

Scaling was tested early in the process.

**Observation:**

* Helpful for linear models
* Unnecessary for tree-based models like XGBoost, LightGBM, and CatBoost

For the final stacked solution, scaling was **not** used.

---

### 3) Log transforms and skew handling

We tested log transforms on heavily skewed columns such as:

* `TotalRooms`
* `TotalBedrooms`
* `NeighborhoodPop`
* `AvgOccupancy`
* `RoomsPerHousehold`

These sometimes improved validation, but not consistently on the Kaggle leaderboard.

**Takeaway:**

* Mild feature engineering helped
* Over-processing hurt more often than it helped

---

### 4) Geo features

We tried several location-based ideas:

* `dist_center`
* `Lat_Long`
* `Lat2`, `Long2`
* KMeans geo clusters
* Geo binning
* Target encoding of geo bins

**Outcome:**

* Simple geo features sometimes helped
* More complex geo encoding often degraded leaderboard performance

The safest geo additions were:

* `dist_center`
* `Lat_Long`

---

### 5) Target transformation

We tried training on:

* `log1p(TargetPrice)`

**Result:**

* Validation looked stable
* Kaggle leaderboard performance did not improve

Final decision: **predict the original target scale directly**.

---

### 6) Model blending and stacking

We compared several approaches:

* Single `RandomForestRegressor`
* Single `XGBRegressor`
* XGB + LightGBM + CatBoost blending
* Ridge stacking on OOF predictions
* Weighted average blending of OOF predictions

**Best overall direction:**

* Use a small stack of diverse tree models
* Blend final predictions using OOF-guided weights

---

## Final Modeling Strategy

The final notebook uses:

* Simple feature engineering
* Median imputation
* 5-fold CV
* Base models:

  * XGBoost
  * LightGBM
  * CatBoost
* OOF prediction collection
* Weighted blending of model outputs

This approach was chosen because it consistently performed better than a single heavily tuned model.

---

## Data Access

The dataset is already included in this repository for convenience.

Alternatively, you can download it directly from Kaggle using `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('predict-housing-price')

print("Path to competition files:", path)
```

---

## Repository Structure

```text
.
├── main.ipynb
├── data/
│   ├── estate_sample_submission.csv
│   ├── estate_test.csv
│   └── estate_train.csv
└── README.md
```

---

## Reproducibility

To reproduce the results presented in this repository:

1. Install the required dependencies.
2. Place the dataset files in the appropriate directory.
3. Run the notebook sequentially from the first cell to the last.

The final output generated will be:

```text
submission.csv
```

---

## Suggested Environment

The notebook was built to run on a Kaggle notebook environment with GPU enabled.

Recommended packages:

* `numpy`
* `pandas`
* `scikit-learn`
* `xgboost`
* `lightgbm`
* `catboost`
* `tqdm`

---

## Practical Tips from the Experiments

A few important lessons from the trial-and-error phase:

* For tabular housing data, simple tree models are very strong.
* Cross-validation stability is more important than chasing one lucky split.

---

## Final Output

The notebook generates a submission file in the correct Kaggle format:

```text
submission.csv
```

---

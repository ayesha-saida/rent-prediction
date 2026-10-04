# Flat Rent Price Prediction in Dhaka City

A machine learning and explainable AI notebook that predicts monthly flat rents in Dhaka, Bangladesh, from property and location features. It compares **Linear Regression**, **Random Forest**, and **XGBoost** on a small, manually collected dataset, and uses **SHAP** to explain the best model.


The notebook was written for **Google Colab**.

----

## Highlights of this project

- Small, incomplete real-world data (103 listings, many missing values), so the evaluation is built to avoid optimistic results:
  - preprocessing is fitted **inside** the cross-validation folds (no leakage);
  - hyperparameters are tuned with **randomized search**, and models are compared with **nested repeated cross-validation**;
  - the **held-out test set** is used once, for the final evaluation only.
- Metrics: MAE, RMSE, and R² (errors are in Bangladeshi Taka, BDT), plus a four-band "rent band" accuracy and confusion matrix.
- SHAP analysis of the selected model, with one-hot columns aggregated back to the original features.

----

## Dataset

The dataset is **not included** in this repository. It was collected manually from public rental listings on Bproperty and Facebook property groups.

The notebook expects a CSV with the below columns:

| Column | Type | Description |
|---|---|---|
| `Listing_ID` | Identifier | Unique listing ID (not used as a feature) |
| `Location` | Categorical | Neighborhood name |
| `Property_Type` | Categorical | Apartment, house, bachelor, or sublet |
| `Area_sqft` | Numerical | Floor area in square feet |
| `Bedrooms` | Numerical | Number of bedrooms |
| `Bathrooms` | Categorical | Number of bathrooms, or `C` (common) |
| `Floor` | Numerical | Floor level |
| `Balcony` | Numerical | Number of balconies (`N` = none, `C` = common, treated as missing) |
| `Gas` | Binary | Gas availability (`Y`/`N`) |
| `Parking` | Categorical | Parking availability (`Y`/`N`) |
| `Rent_BDT` | **Target** | Advertised monthly rent (BDT) |
| `Service_Charge_BDT` | Numerical/Binary | Service charge (excluded from the final feature set) |
| `Listing Date` | Date | Listing date (not used) |

Of the 103 listings, 102 have a rent value and are used (81 train / 21 test). Floor area is missing for 46 of the 102 usable listings. There are no latitude/longitude columns, so location is modeled as a categorical feature.

----

## Notebook structure

1. **Setup and configuration**: imports, plotting style, constants.
2. **Data load and inspection**: shape, dtypes, missing values, placeholder codes, duplicates, descriptive statistics, target distribution.
3. **Data cleaning**: whitespace stripping, numeric conversion of rent, service-charge `N/Y` mapping, balcony/gas encoding.
4. **Exploratory analysis**: correlation matrix, rent by property type and parking.
5. **Feature set and train/test split**: 80/20 split.
6. **Preprocessing pipeline**: median imputation with missing indicators and scaling for numerical features, mode imputation for the binary feature, one-hot encoding for categorical features (categories with fewer than 3 listings are pooled).
7. **Service-charge ablation**: repeated cross-validated Random Forest with and without the feature.
8. **Models and hyperparameter tuning**: `RandomizedSearchCV` for Random Forest and XGBoost.
9. **Nested cross-validation** comparison and **holdout test** evaluation.
10. **Best-model selection**, predicted vs. actual plot, and rent-band confusion matrix.
11. **SHAP analysis**: summary plot, aggregated importance, dependence plots.

----

## Configuration

Set in the configuration cell near the top:

|      Constant        |     Value     |             Meaning                          |
|----------------------|---------------|----------------------------------------------|
| `RANDOM_STATE`       |     `42`      |    Seed for the split, CV, and models        |
| `TARGET_COL`         |  `"Rent_BDT"` |    Target column                             |
| `USE_SERVICE_CHARGE` |   `False`     |    Include `Service_Charge_BDT` as a feature |
| `N_ITER_SEARCH`      |    `20`       |    Random-search configurations per model    |

----

## Method summary

- **Split:** 80/20 train/test, seed 42.
- **Tuning:** 20 random configurations per model, scored by negative RMSE with 5-fold CV repeated 2 times on the training set.
- **Model comparison:** nested CV, with an outer loop of 5-fold repeated 3 times (15 validation runs per model).
- **Selection:** the model with the lowest mean cross-validated RMSE.
- **Final evaluation:** each tuned model is refitted on the full training set and scored once on the test set.

----

## Results

**Nested cross-validation (mean ± std)**

|     Model         | MAE (BDT)  | RMSE (BDT)  |       R²      |
|-------------------|------------|-------------|---------------|
| Linear Regression | 2824 ± 886 | 3764 ± 1163 | 0.511 ± 0.256 |
| Random Forest     | 2059 ± 463 | 2899 ± 652  | 0.718 ± 0.120 |
| XGBoost           | 2025 ± 467 | 3014 ± 685  | 0.697 ± 0.116 |


**Held-out test set (n = 21)**

|      Model        | MAE (BDT) | RMSE (BDT) | R²    |
|-------------------|-----------|------------|-------|
| Linear Regression | 1871      |       2394 | 0.773 |
| Random Forest     | 1450      |       1883 | 0.859 |
| XGBoost           | 1623      |       2437 | 0.764 |

- **Best model:** Random Forest (lowest cross-validated RMSE). Its rent-band accuracy on the test set is 71.4% (15 of 21).
- **Service-charge ablation:** including it only lowers the RMSE from 2895 to 2864 BDT, within one standard deviation, so it is excluded.
- **SHAP importance (mean |SHAP|, BDT):** Bedrooms 2557, Bathrooms 856, Area_sqft 770, Location 763, Property_Type 614, Balcony 531, Parking 508, Floor 311, Gas 97.

With only 21 test listings, test scores are noisy, so model comparison relies mainly on cross-validation.

----

## Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` 1.4 or newer (the notebook imports `root_mean_squared_error`)
- `xgboost`
- `shap`

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
```

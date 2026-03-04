# Solar-energy-output-calculator-
Predict the daily kilowatt hours generation of a solar farm based on cloud cover percentage and direct sunlight hours.Use Min Max scaling to normalize weather variables between 0 and 1.

## Preprocessing

### Objective

Prepare the cleaned solar dataset for model training by:
* Removing statistical outliers
* Splitting data into training and testing sets
* Scaling feature values for consistent model performance
* Saving processed datasets for downstream training

### 1) Dataset Loading

The dataset is loaded from:
`cleaned_synthetic_solar_dataset.csv`

Key features used:
* `cloud_cover`
* `direct_sunlight_hours`
* `solar_radiation`
* `panel_efficiency`

Target variable:
* `generated_power_kw`

### 2) Outlier Removal (IQR Method)

Outliers in `generated_power_kw` are removed using the Interquartile Range (IQR) method:

* Q1 = 25th percentile
* Q3 = 75th percentile
* IQR = Q3 − Q1
* Lower bound = Q1 − 1.5 × IQR
* Upper bound = Q3 + 1.5 × IQR

Only values within this range are retained.
This ensures extreme power values do not bias the model training.

### 3) Train–Test Split

The dataset is split using:
* 70% Training Data
* 30% Testing Data
* `random_state = 42` (for reproducibility)

This ensures consistent results across runs.

### 4) Feature Scaling

Feature scaling is performed using:
* `MinMaxScaler`

All feature values are transformed into the range:
`[0,1]`

### Final Processed Outputs

Two CSV files are generated:

`train_dataset.csv`
`test_dataset.csv`

Each file contains:

* Scaled feature columns
* Corresponding `generated_power_kw` target values

These files are saved inside the `preprocessing/` directory.

## Model Training & Evaluation:

### 1) Model Training

The solar power prediction model was developed using a supervised machine learning regression approach.

A Random Forest Regressor was selected because it:

* Captures non-linear relationships
* Handles feature interactions effectively
* Reduces overfitting through ensemble learning

The dataset was split into 70% training and 30% testing data using `train_test_split`.
The model was trained with:

* n_estimators = 500
* max_depth = 15
* min_samples_split = 5
* min_samples_leaf = 2
* random_state = 42

After training, the model was saved as `model.pkl` for deployment.

### 2) Feature Engineering

The following input features were used:

* Cloud Cover
* Direct Sunlight Hours
* Solar Radiation
* Panel Efficiency

These features represent environmental and system-level factors that directly influence solar power generation.

The target variable used for prediction:

* Generated Power (kW)

### 3) Model Evaluation

To assess performance, multiple regression metrics were used:

a)R² Score:Measures how much variance in solar power output is explained by the model.
The final model achieved R² ≈ 0.87**, meaning it explains approximately **87% of the variation** in energy generation.

b)Mean Absolute Error (MAE):Represents the average absolute difference between predicted and actual power values.

c)Mean Squared Error (MSE):Penalizes larger prediction errors more heavily.

d)Root Mean Squared Error (RMSE):Provides prediction error in the same unit as output (kW), making interpretation practical.RMSE was also compared with average power output to evaluate relative prediction error.

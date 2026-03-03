# Solar-energy-output-calculator-
Predict the daily kilowatt hours generation of a solar farm based on cloud cover percentage and direct sunlight hours.Use Min Max scaling to normalize weather variables between 0 and 1.
## Model Training & Evaluation:

### 1) Model Training

The solar power prediction model was developed using a supervised machine learning regression approach.

A Random Forest Regressorwa s selected because it:

* Captures non-linear relationships
* Handles feature interactions effectively
* Reduces overfitting through ensemble learning

The dataset was split into 80% training and 20% testing data using `train_test_split`.
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

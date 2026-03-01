import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import os


# Get project root directory
current_file = os.path.abspath(__file__)
model_folder = os.path.dirname(current_file)
project_root = os.path.dirname(model_folder)

train_path = os.path.join(project_root, "preprocessing", "train_dataset.csv")
test_path = os.path.join(project_root, "preprocessing", "test_dataset.csv")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


features = ["cloud_cover","direct_sunlight_hours","solar_radiation","panel_efficiency"]
X_train = train_data[features]

y_train = train_data["generated_power_kw"]

X_test = test_data[["cloud_cover","direct_sunlight_hours","solar_radiation","panel_efficiency"]]
y_test = test_data["generated_power_kw"]



scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, (y_pred))
r2 = r2_score(y_test, (y_pred))

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)
model_path = os.path.join(project_root, "model.pkl")
scaler_path = os.path.join(project_root, "scaler.pkl")
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("Model and Scaler saved successfully!")

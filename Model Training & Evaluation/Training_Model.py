import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

train_data = pd.read_csv("train_dataset_v2.csv")
test_data = pd.read_csv("test_dataset_v2.csv")


features = ["total_cloud_cover_sfc","shortwave_radiation_backwards_sfc","Angle_of_incidence","Zenith"]
X_train = train_data[features]
y_train = train_data["generated_power_kw"]

X_test = test_data[["total_cloud_cover_sfc","shortwave_radiation_backwards_sfc","Angle_of_incidence", "Zenith"]]
y_test = test_data["generated_power_kw"]


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved successfully!")
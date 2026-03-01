import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv("cleaned_synthetic_solar_dataset.csv")

features = [ "cloud_cover","direct_sunlight_hours", "solar_radiation","panel_efficiency"]
X = data[features]
y = data["generated_power_kw"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
print("R^2 Score:", r2)

mean_power = y.mean()
error_percent = (rmse / mean_power) * 100

print("Average Power:", mean_power)
print("RMSE as % of Average Power:", error_percent, "%")

joblib.dump(model, "model.pkl")
print("Model saved successfully!")

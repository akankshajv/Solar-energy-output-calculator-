import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned dataset
data = pd.read_csv("preprocessing/cleaned_solar_dataset.csv")

X = data[['total_cloud_cover_sfc',
          'shortwave_radiation_backwards_sfc']]

y = data['generated_power_kw']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and Scaler Saved Successfully!")
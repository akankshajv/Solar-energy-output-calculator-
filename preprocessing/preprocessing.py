import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os


# Get project root directory
current_file = os.path.abspath(__file__)
model_folder = os.path.dirname(current_file)
project_root = os.path.dirname(model_folder)


# 0 in the dataset means either the radiation was below the sensor's detection limit as in night radiation is very low
# 0 in cloud cover means clear sky, 100 means completely overcast sky.
dataset = pd.read_csv(os.path.join(project_root, "cleaned_synthetic_solar_dataset.csv"))


Q1 = dataset["generated_power_kw"].quantile(0.25)
Q3 = dataset["generated_power_kw"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dataset = dataset[
    (dataset["generated_power_kw"] >= lower_bound) &
    (dataset["generated_power_kw"] <= upper_bound)
]
X = dataset[["cloud_cover", "direct_sunlight_hours", "solar_radiation", "panel_efficiency"]]
y = dataset["generated_power_kw"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train,columns=["cloud_cover", "direct_sunlight_hours", "solar_radiation", "panel_efficiency"])

X_test = pd.DataFrame(X_test,columns=["cloud_cover", "direct_sunlight_hours", "solar_radiation", "panel_efficiency"])
train_data = X_train.copy()
train_data["generated_power_kw"] = y_train.values

test_data = X_test.copy()
test_data["generated_power_kw"] = y_test.values

train_data.to_csv(os.path.join(project_root, "preprocessing", "train_dataset.csv"), index=False)
test_data.to_csv(os.path.join(project_root, "preprocessing", "test_dataset.csv"), index=False)
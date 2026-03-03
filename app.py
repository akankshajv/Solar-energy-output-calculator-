from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        cloud = float(request.form['cloud_cover'])
        radiation = float(request.form['solar_radiation'])
        hours = float(request.form['direct_sunlight_hours'])
        efficiency = float(request.form['panel_efficiency'])/100

        input_data = np.array([[cloud,hours,radiation,efficiency]])

        # Scale input
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]
        prediction = round(prediction, 4)

        return render_template("predict.html", result=prediction)

    except:
        return render_template("predict.html", result="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True,port=8000)
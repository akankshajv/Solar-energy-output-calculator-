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
        cloud = float(request.form['total_cloud_cover_sfc'])
        radiation = float(request.form['shortwave_radiation_backwards_sfc'])

        input_data = np.array([[cloud, radiation]])

        # Scale input
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)[0]
        prediction = round(prediction, 2)

        return render_template("predict.html", result=prediction)

    except:
        return render_template("predict.html", result="Invalid Input")

if __name__ == "__main__":
    app.run(debug=True)
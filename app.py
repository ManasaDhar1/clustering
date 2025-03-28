from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load dataset and model
df = pd.read_csv("World_Development_Dataset.csv")
kmeans = joblib.load("model.pkl")  # Ensure model.pkl exists

# Define feature columns (excluding 'Country')
columns = [
    "Birth Rate", "Business Tax Rate", "CO2 Emissions", "Days to Start Business",
    "Ease of Business", "Energy Usage", "GDP", "Health Exp % GDP", "Health Exp/Capita",
    "Life Expectancy Female", "Life Expectancy Male", "Mobile Phone Usage", "Number of Records",
    "Population 0-14", "Population 15-64", "Population 65+", "Population Total",
    "Population Urban", "Tourism Inbound", "Tourism Outbound"
]

@app.route('/')
def index():
    return render_template('index.html', columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_values = [float(request.form[col]) for col in columns]
        input_data = np.array(feature_values).reshape(1, -1)

        predicted_cluster = kmeans.predict(input_data)[0]

        return render_template('result.html', prediction=predicted_cluster, image=None)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

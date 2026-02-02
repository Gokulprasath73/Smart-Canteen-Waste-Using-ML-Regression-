from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------
# Create Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Create synthetic dataset
# -----------------------------
np.random.seed(42)
n_samples = 500

data = pd.DataFrame({
    "total_customers": np.random.randint(80, 300, n_samples),
    "food_prepared_kg": np.random.uniform(40, 120, n_samples),
    "menu_variety": np.random.randint(5, 20, n_samples),
    "avg_item_price": np.random.uniform(40, 120, n_samples),
    "previous_day_waste_kg": np.random.uniform(2, 25, n_samples),
    "temperature_c": np.random.uniform(20, 40, n_samples),
    "special_event": np.random.choice([0, 1], n_samples),
    "day_of_week": np.random.randint(0, 7, n_samples)
})

data["waste_cost_inr"] = (
    15 * data["food_prepared_kg"]
    + 10 * data["menu_variety"]
    + 8 * data["previous_day_waste_kg"]
    - 5 * data["total_customers"]
    + 12 * data["special_event"]
    + np.random.normal(0, 50, n_samples)
)

# -----------------------------
# Train model
# -----------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return {"message": "Smart Canteen Waste Prediction API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json

        features = [
            input_data["total_customers"],
            input_data["food_prepared_kg"],
            input_data["menu_variety"],
            input_data["avg_item_price"],
            input_data["previous_day_waste_kg"],
            input_data["temperature_c"],
            input_data["special_event"],
            input_data["day_of_week"]
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "predicted_waste_cost_inr": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

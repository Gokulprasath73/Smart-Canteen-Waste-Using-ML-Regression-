from flask import Flask, request, jsonify
import numpy as np
import joblib

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load saved model & scaler
# -----------------------------
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Home route
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Smart Canteen Waste Prediction API is running"
    })

# -----------------------------
# Prediction route (with validation)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Required fields
        required_fields = [
            "total_customers",
            "food_prepared_kg",
            "menu_variety",
            "avg_item_price",
            "previous_day_waste_kg",
            "temperature_c",
            "special_event",
            "day_of_week"
        ]

        # Check missing fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert data types
        total_customers = int(data["total_customers"])
        food_prepared_kg = float(data["food_prepared_kg"])
        menu_variety = int(data["menu_variety"])
        avg_item_price = float(data["avg_item_price"])
        previous_day_waste_kg = float(data["previous_day_waste_kg"])
        temperature_c = float(data["temperature_c"])
        special_event = int(data["special_event"])
        day_of_week = int(data["day_of_week"])

        # Validate ranges
        if total_customers < 0:
            return jsonify({"error": "total_customers must be >= 0"}), 400

        if food_prepared_kg <= 0:
            return jsonify({"error": "food_prepared_kg must be > 0"}), 400

        if not (0 <= special_event <= 1):
            return jsonify({"error": "special_event must be 0 or 1"}), 400

        if not (0 <= day_of_week <= 6):
            return jsonify({"error": "day_of_week must be between 0 and 6"}), 400

        # Prepare input
        features = np.array([[
            total_customers,
            food_prepared_kg,
            menu_variety,
            avg_item_price,
            previous_day_waste_kg,
            temperature_c,
            special_event,
            day_of_week
        ]])

        # Scale & predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            "predicted_waste_cost_inr": round(float(prediction), 2)
        })

    except ValueError:
        return jsonify({"error": "Invalid data type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

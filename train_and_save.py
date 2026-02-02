import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import joblib

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

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Save model & scaler
# -----------------------------
joblib.dump(model, "ridge_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")

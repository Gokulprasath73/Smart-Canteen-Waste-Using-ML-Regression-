import streamlit as st
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Smart Canteen Waste Predictor", layout="centered")

st.title("🍽 Smart Canteen Food Waste Predictor")
st.write("Streamlit UI → Flask API → ML Model")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Canteen Details")

total_customers = st.slider("Total Customers", 80, 300, 180)
food_prepared_kg = st.slider("Food Prepared (kg)", 40, 120, 85)
menu_variety = st.slider("Menu Variety Count", 5, 20, 12)
avg_item_price = st.slider("Average Item Price (₹)", 40, 120, 75)
previous_day_waste_kg = st.slider("Previous Day Waste (kg)", 2, 25, 10)
temperature_c = st.slider("Temperature (°C)", 20, 40, 32)
special_event = st.selectbox("Special Event?", [0, 1])
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", [0,1,2,3,4,5,6])

# -----------------------------
# API Call
# -----------------------------
if st.button("Predict Waste Cost"):
    payload = {
        "total_customers": total_customers,
        "food_prepared_kg": food_prepared_kg,
        "menu_variety": menu_variety,
        "avg_item_price": avg_item_price,
        "previous_day_waste_kg": previous_day_waste_kg,
        "temperature_c": temperature_c,
        "special_event": special_event,
        "day_of_week": day_of_week
    }

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload
        )

        result = response.json()
        st.success(f"💰 Predicted Waste Cost: ₹ {result['predicted_waste_cost_inr']}")

    except Exception as e:
        st.error(f"API Error: {e}")

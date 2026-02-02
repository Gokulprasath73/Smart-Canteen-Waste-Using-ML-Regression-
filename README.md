# Smart-Canteen-Waste-Using-ML-Regression-


🍽 Smart Canteen Food Waste Cost Prediction

A machine learning project that predicts daily food waste cost (₹) in a canteen using Linear, Ridge, and Lasso Regression, deployed with a Streamlit web interface.

🔍 Problem Statement

Food wastage in institutional canteens leads to significant financial loss.
This project predicts daily food waste cost using operational and environmental factors to support data-driven food preparation planning.

🚀 Features

Synthetic but realistic dataset

Linear, Ridge & Lasso Regression comparison

Hyperparameter tuning

Model persistence using joblib

Flask REST API (backend)

Streamlit graphical interface (frontend)

Input validation & error handling

Ready for cloud deployment

📊 Input Features

| Feature               | Description                |
| --------------------- | -------------------------- |
| total_customers       | Number of customers served |
| food_prepared_kg      | Total food cooked (kg)     |
| menu_variety          | Number of menu items       |
| avg_item_price        | Average item price (₹)     |
| previous_day_waste_kg | Waste from previous day    |
| temperature_c         | Daily temperature          |
| special_event         | 0 = No, 1 = Yes            |
| day_of_week           | 0 = Monday, 6 = Sunday     |



🎯 Target

waste_cost_inr — Estimated food waste cost (₹)


🧠 Machine Learning Models

Linear Regression (baseline)

Ridge Regression (final model)

Lasso Regression (feature selection)


┌──────────────┐
│     User     │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│   Streamlit UI      │
│  (Frontend Layer)   │
└──────┬──────────────┘
       │ HTTP / JSON
       ▼
┌─────────────────────┐
│    Flask API        │
│  (Backend Layer)    │
│ - Input Validation  │
│ - Preprocessing     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   ML Model Layer    │
│ Ridge Regression    │
│ + StandardScaler    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Prediction (₹)    │
└─────────────────────┘


🛠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

Flask

Streamlit

Joblib

uv (package manager)


▶️ How to Run Locally
1️⃣ Clone repository
git clone https://github.com/your-username/smart-canteen-waste
cd smart-canteen-waste

2️⃣ Install dependencies
uv add -r requirements.txt

3️⃣ Run Flask API
python app.py

4️⃣ Run Streamlit UI
streamlit run streamlit_app.py

📈 Sample Output
Predicted Food Waste Cost: ₹ 1325.47

🎓 Learning Outcomes

Applied regression techniques to real-world problems

Understood regularization (Ridge & Lasso)

Built REST APIs with Flask

Designed interactive ML dashboards with Streamlit

Learned production practices (model saving, validation)

📌 Future Improvements

Add real canteen data

Role-based authentication

Daily waste optimization recommendations

Cloud deployment with CI/CD

👤 Author

Gokul Prasath
Aspiring Data Scientist | ML & Full-Stack Enthusiast


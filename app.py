# ============================================
# Streamlit App: Delivery Time Prediction (Improved UI + Random Data)
# ============================================

import streamlit as st
import pandas as pd
import joblib
import random

# ============================================
# 1. Load Preprocessing and Models
# ============================================

preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 2. Prediction Function
# ============================================

def predict_delivery_time(input_data):
    df_input = pd.DataFrame([input_data])
    numeric_features = [
        "Delivery_person_Age","Delivery_person_Ratings",
        "Restaurant_latitude","Restaurant_longitude",
        "Delivery_location_latitude","Delivery_location_longitude",
        "multiple_deliveries","order_day_of_week","order_month",
        "order_hour","pickup_hour","pickup_delay_min"
    ]
    df_input[numeric_features] = df_input[numeric_features].astype(float)
    return {
        "Linear Regression": round(lr_model.predict(df_input)[0],2),
        "Decision Tree": round(dt_model.predict(df_input)[0],2),
        "Random Forest": round(rf_model.predict(df_input)[0],2)
    }

# ============================================
# 3. Random Data Generator
# ============================================

def generate_random_delivery_data():
    return {
        "Delivery_person_Age": random.randint(18, 60),
        "Delivery_person_Ratings": round(random.uniform(2.5, 5.0), 1),
        "Restaurant_latitude": round(random.uniform(12.90, 13.00), 6),
        "Restaurant_longitude": round(random.uniform(77.55, 77.65), 6),
        "Delivery_location_latitude": round(random.uniform(12.90, 13.00), 6),
        "Delivery_location_longitude": round(random.uniform(77.55, 77.65), 6),
        "multiple_deliveries": random.randint(1, 5),
        "order_day_of_week": random.randint(0, 6),
        "order_month": random.randint(1, 12),
        "order_hour": random.randint(8, 22),
        "pickup_hour": random.randint(8, 23),
        "pickup_delay_min": random.randint(0, 30),
        "Weatherconditions": random.choice(["Sunny","Cloudy","Rainy","Stormy","Fog"]),
        "Road_traffic_density": random.choice(["Low","Medium","High","Jam"]),
        "Type_of_order": random.choice(["Vegetables","Meat","Meat and Vegetables"]),
        "Type_of_vehicle": random.choice(["Bike","Car","Scooter"]),
        "Festival": random.choice(["Yes","No"])
    }

# ============================================
# 4. Streamlit UI
# ============================================

st.set_page_config(
    page_title="Delivery Time Predictor 🚀",
    page_icon="🛵",
    layout="wide"
)

st.title("🛵 Delivery Time Prediction System")
st.markdown(
    "Enter delivery details or generate random data to predict delivery times using three models."
)

# ---- Generate Random Data Button ----
if st.button("🎲 Generate Random Delivery Data"):
    random_data = generate_random_delivery_data()
    for key, value in random_data.items():
        st.session_state[key] = value
    st.success("✅ Random delivery data generated!")

# ---- Input Sections ----
with st.expander("📍 Restaurant & Delivery Location"):
    col1, col2 = st.columns(2)
    with col1:
        restaurant_lat = st.number_input("Restaurant Latitude", format="%.6f", key="Restaurant_latitude")
        restaurant_long = st.number_input("Restaurant Longitude", format="%.6f", key="Restaurant_longitude")
    with col2:
        delivery_lat = st.number_input("Delivery Latitude", format="%.6f", key="Delivery_location_latitude")
        delivery_long = st.number_input("Delivery Longitude", format="%.6f", key="Delivery_location_longitude")

with st.expander("🧑 Delivery Person Info"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Delivery Person Age", 18, 60, key="Delivery_person_Age")
        rating = st.number_input("Delivery Person Rating", 0.0, 5.0, 0.1, key="Delivery_person_Ratings")
    with col2:
        multiple_deliveries = st.number_input("Multiple Deliveries", 1, 10, key="multiple_deliveries")

with st.expander("⏰ Order & Pickup Info"):
    col1, col2 = st.columns(2)
    with col1:
        order_day = st.number_input("Order Day of Week (0=Mon,6=Sun)", 0, 6, key="order_day_of_week")
        order_month = st.number_input("Order Month", 1, 12, key="order_month")
        order_hour = st.number_input("Order Hour", 0, 23, key="order_hour")
    with col2:
        pickup_hour = st.number_input("Pickup Hour", 0, 23, key="pickup_hour")
        pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120, key="pickup_delay_min")

with st.expander("🌦️ Traffic & Conditions"):
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Fog"], key="Weatherconditions")
        traffic = st.selectbox("Traffic Density", ["Low","Medium","High","Jam"], key="Road_traffic_density")
    with col2:
        order_type = st.selectbox("Type of Order", ["Snack","Meal","Drinks","Other"], key="Type_of_order")
        vehicle = st.selectbox("Vehicle Type", ["Bike","Car","Scooter"], key="Type_of_vehicle")
        festival = st.selectbox("Festival", ["Yes","No"], key="Festival")

# ---- Predict Button ----
if st.button("🚀 Predict Delivery Time"):
    input_data = {key: st.session_state[key] for key in [
        "Delivery_person_Age","Delivery_person_Ratings",
        "Restaurant_latitude","Restaurant_longitude",
        "Delivery_location_latitude","Delivery_location_longitude",
        "multiple_deliveries","order_day_of_week","order_month",
        "order_hour","pickup_hour","pickup_delay_min",
        "Weatherconditions","Road_traffic_density","Type_of_order",
        "Type_of_vehicle","Festival"
    ]}
    
    predictions = predict_delivery_time(input_data)
    
    st.markdown("### 📊 Predictions by Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{predictions['Linear Regression']} min")
    col2.metric("Decision Tree", f"{predictions['Decision Tree']} min")
    col3.metric("Random Forest", f"{predictions['Random Forest']} min")
    
    # Bar chart
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"]).set_index("Model")
    st.bar_chart(df_pred)
    
    # Highlight fastest
    fastest_model = df_pred["Predicted Time"].idxmin()
    fastest_time = df_pred["Predicted Time"].min()
    st.success(f"✅ Fastest Predicted Delivery: {fastest_model} ({fastest_time} min)")

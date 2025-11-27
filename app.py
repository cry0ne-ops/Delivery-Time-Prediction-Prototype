# ============================================
# Streamlit App: Delivery Time Prediction (Improved UI)
# ============================================

import streamlit as st
import pandas as pd
import joblib

# ============================================
# 1. Load Models
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
# 3. UI Layout
# ============================================

st.set_page_config(
    page_title="Delivery Time Predictor 🚀",
    page_icon="🛵",
    layout="wide"
)

st.title("🛵 Delivery Time Prediction System")
st.markdown(
    """
    Enter the delivery details below to predict estimated delivery times using three models.
    """
)

# Group Inputs in Expanders
with st.expander("📍 Restaurant & Delivery Location"):
    col1, col2 = st.columns(2)
    with col1:
        restaurant_lat = st.number_input("Restaurant Latitude", format="%.6f")
        restaurant_long = st.number_input("Restaurant Longitude", format="%.6f")
    with col2:
        delivery_lat = st.number_input("Delivery Latitude", format="%.6f")
        delivery_long = st.number_input("Delivery Longitude", format="%.6f")

with st.expander("🧑 Delivery Person Info"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Delivery Person Age", 18, 60)
        rating = st.number_input("Delivery Person Rating", 0.0, 5.0, 0.1)
    with col2:
        multiple_deliveries = st.number_input("Multiple Deliveries", 1, 10)

with st.expander("⏰ Order & Pickup Info"):
    col1, col2 = st.columns(2)
    with col1:
        order_day = st.number_input("Order Day of Week (0=Mon,6=Sun)", 0, 6)
        order_month = st.number_input("Order Month", 1, 12)
        order_hour = st.number_input("Order Hour", 0, 23)
    with col2:
        pickup_hour = st.number_input("Pickup Hour", 0, 23)
        pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120)

with st.expander("🌦️ Traffic & Conditions"):
    col1, col2 = st.columns(2)
    with col1:
        weather = st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Fog"])
        traffic = st.selectbox("Traffic Density", ["Low","Medium","High","Jam"])
    with col2:
        order_type = st.selectbox("Type of Order", ["Meat","Vegetables","Meat and Vegetables"])
        vehicle = st.selectbox("Vehicle Type", ["Bike","Car","Scooter"])
        festival = st.selectbox("Festival", ["Yes","No"])

# ============================================
# 4. Prediction
# ============================================

if st.button("🚀 Predict Delivery Time"):
    input_data = {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Restaurant_latitude": restaurant_lat,
        "Restaurant_longitude": restaurant_long,
        "Delivery_location_latitude": delivery_lat,
        "Delivery_location_longitude": delivery_long,
        "multiple_deliveries": multiple_deliveries,
        "order_day_of_week": order_day,
        "order_month": order_month,
        "order_hour": order_hour,
        "pickup_hour": pickup_hour,
        "pickup_delay_min": pickup_delay,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle,
        "Festival": festival
    }

    predictions = predict_delivery_time(input_data)

    st.markdown("### 📊 Predictions by Model")
    
    # Show metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{predictions['Linear Regression']} min")
    col2.metric("Decision Tree", f"{predictions['Decision Tree']} min")
    col3.metric("Random Forest", f"{predictions['Random Forest']} min")
    
    # Bar chart comparison
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"])
    df_pred = df_pred.set_index("Model")
    st.bar_chart(df_pred)
    
    # Highlight fastest model
    fastest_model = df_pred["Predicted Time"].idxmin()
    fastest_time = df_pred["Predicted Time"].min()
    st.success(f"✅ Fastest Predicted Delivery: {fastest_model} ({fastest_time} min)")

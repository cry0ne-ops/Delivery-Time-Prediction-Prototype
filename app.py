import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import requests

# ------------------------------
# LOAD MODELS & SCALER
# ------------------------------
def load_model(model_name):
    try:
        return joblib.load(model_name)
    except Exception as e:
        st.error(f"Could not load model {model_name}: {e}")
        return None

scaler = joblib.load("scaler.pkl")  # global scaler for ALL models

# ------------------------------
# STREAMLIT UI SETUP
# ------------------------------
st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="wide",
    page_icon="⏱️"
)

st.markdown("<h1 style='text-align:center;'>📦 Delivery Time Prediction App</h1>", unsafe_allow_html=True)
st.write("Use the sidebar to configure prediction settings.")

# ------------------------------
# SIDEBAR CONFIGURATION PANEL
# ------------------------------
st.sidebar.header("⚙️ Model & Input Settings")

model_choice = st.sidebar.selectbox(
    "Select Regression Model",
    [
        "delivery_time_model.pkl",
        "linear_regression_model.pkl",
        "decision_tree_model.pkl"
    ]
)

# Example inputs – replace with real UI components
delivery_person_age = st.sidebar.number_input("Delivery Person Age", 18, 70, 30)
delivery_person_ratings = st.sidebar.number_input("Delivery Rating", 1.0, 5.0, 4.5)
restaurant_lat = st.sidebar.number_input("Restaurant Latitude", value=16.4023)
restaurant_long = st.sidebar.number_input("Restaurant Longitude", value=120.5960)
delivery_lat = st.sidebar.number_input("Delivery Latitude", value=16.4152)
delivery_long = st.sidebar.number_input("Delivery Longitude", value=120.5900)

weather = st.sidebar.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
type_of_order = st.sidebar.selectbox("Order Type", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Bike", "Motorcycle", "Car"])
multiple_deliveries = st.sidebar.number_input("Multiple Deliveries", 0, 5, 0)
festival = st.sidebar.selectbox("Festival", ["Yes", "No"])

# ------------------------------
# MAP (Left Column)
# ------------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🗺️ Delivery Route Map")

    map_center = [(restaurant_lat + delivery_lat) / 2,
                  (restaurant_long + delivery_long) / 2]

    route_map = folium.Map(location=map_center, zoom_start=13)
    folium.Marker([restaurant_lat, restaurant_long],
                  tooltip="Restaurant", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker([delivery_lat, delivery_long],
                  tooltip="Delivery Location", icon=folium.Icon(color="red")).add_to(route_map)
    
    st_folium(route_map, width=500, height=400)

# ------------------------------
# PREDICTION PANEL (Right Column)
# ------------------------------
with col2:
    st.subheader("⏱️ Predicted Delivery Time")

    if st.button("Predict Delivery Time", use_container_width=True):

        model = load_model(model_choice)
        if model is None:
            st.error("Model failed to load.")
            st.stop()

        # ------------------------------
        # CREATE FEATURE ROW
        # ------------------------------
        df = pd.DataFrame([{
            "Delivery_person_Age": delivery_person_age,
            "Delivery_person_Ratings": delivery_person_ratings,
            "Restaurant_latitude": restaurant_lat,
            "Restaurant_longitude": restaurant_long,
            "Delivery_location_latitude": delivery_lat,
            "Delivery_location_longitude": delivery_long,
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Type_of_order": type_of_order,
            "Type_of_vehicle": vehicle_type,
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival
        }])

        # One-hot encode categorical columns (replace with the same encoding used in training)
        df = pd.get_dummies(df)

        # Load saved feature names from training
        feature_names = joblib.load("feature_names.pkl")  # <--- saved after training

        # Add missing columns and reorder
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

        # Scale inputs
        scaled_features = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled_features)[0]
        prediction = round(float(prediction), 2)

        st.success(f"Estimated Delivery Time: **{prediction} minutes**")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.write("© 2025 Delivery Time Prediction System | Powered by Streamlit")

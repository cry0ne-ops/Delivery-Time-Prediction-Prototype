# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# =============================================
# 1. LOAD MODELS
# =============================================
pipeline_lr = joblib.load("linear_regression_model.pkl")
pipeline_dt = joblib.load("decision_tree_model.pkl")
pipeline_rf = joblib.load("random_forest_model.pkl")

models = {
    "Linear Regression": pipeline_lr,
    "Decision Tree": pipeline_dt,
    "Random Forest": pipeline_rf
}

# =============================================
# 2. STREAMLIT APP
# =============================================
st.title("Delivery Time Prediction")

# Select model
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# =============================================
# 3. INPUT FEATURES
# =============================================
st.sidebar.header("Enter Order Details:")

def user_input_features():
    Delivery_person_Age = st.sidebar.number_input("Delivery Person Age", 18, 70, 25)
    Delivery_person_Ratings = st.sidebar.number_input("Delivery Person Ratings", 0.0, 5.0, 4.5)
    Restaurant_latitude = st.sidebar.number_input("Restaurant Latitude", -90.0, 90.0, 14.6)
    Restaurant_longitude = st.sidebar.number_input("Restaurant Longitude", -180.0, 180.0, 120.9)
    Delivery_location_latitude = st.sidebar.number_input("Delivery Latitude", -90.0, 90.0, 14.65)
    Delivery_location_longitude = st.sidebar.number_input("Delivery Longitude", -180.0, 180.0, 120.95)
    
    Weatherconditions = st.sidebar.selectbox("Weather Conditions", ["Sunny", "Cloudy", "Rainy", "Stormy", "Fog"])
    Road_traffic_density = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
    Type_of_order = st.sidebar.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"])
    Type_of_vehicle = st.sidebar.selectbox("Type of Vehicle", ["Bike", "Scooter", "Car"])
    multiple_deliveries = st.sidebar.number_input("Multiple Deliveries", 0, 5, 1)
    Festival = st.sidebar.selectbox("Festival", ["Yes", "No"])
    order_day_of_week = st.sidebar.number_input("Order Day of Week (0=Monday)", 0, 6, 0)
    order_month = st.sidebar.number_input("Order Month", 1, 12, 1)
    order_hour = st.sidebar.number_input("Order Hour", 0, 23, 12)
    pickup_hour = st.sidebar.number_input("Pickup Hour", 0, 23, 12)
    pickup_delay_min = st.sidebar.number_input("Pickup Delay (minutes)", 0, 180, 5)

    data = {
        "Delivery_person_Age": Delivery_person_Age,
        "Delivery_person_Ratings": Delivery_person_Ratings,
        "Restaurant_latitude": Restaurant_latitude,
        "Restaurant_longitude": Restaurant_longitude,
        "Delivery_location_latitude": Delivery_location_latitude,
        "Delivery_location_longitude": Delivery_location_longitude,
        "Weatherconditions": Weatherconditions,
        "Road_traffic_density": Road_traffic_density,
        "Type_of_order": Type_of_order,
        "Type_of_vehicle": Type_of_vehicle,
        "multiple_deliveries": multiple_deliveries,
        "Festival": Festival,
        "order_day_of_week": order_day_of_week,
        "order_month": order_month,
        "order_hour": order_hour,
        "pickup_hour": pickup_hour,
        "pickup_delay_min": pickup_delay_min
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# =============================================
# 4. PREDICTION
# =============================================
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")

    # =============================================
    # 5. DISPLAY MAP
    # =============================================
    map_center = [
        (input_df["Restaurant_latitude"][0] + input_df["Delivery_location_latitude"][0]) / 2,
        (input_df["Restaurant_longitude"][0] + input_df["Delivery_location_longitude"][0]) / 2
    ]
    m = folium.Map(location=map_center, zoom_start=13)

    # Restaurant marker
    folium.Marker(
        location=[input_df["Restaurant_latitude"][0], input_df["Restaurant_longitude"][0]],
        popup="Restaurant",
        icon=folium.Icon(color="green", icon="cutlery", prefix='fa')
    ).add_to(m)

    # Delivery location marker
    folium.Marker(
        location=[input_df["Delivery_location_latitude"][0], input_df["Delivery_location_longitude"][0]],
        popup="Delivery Location",
        icon=folium.Icon(color="red", icon="home", prefix='fa')
    ).add_to(m)

    # Draw line between restaurant and delivery
    folium.PolyLine(
        locations=[
            [input_df["Restaurant_latitude"][0], input_df["Restaurant_longitude"][0]],
            [input_df["Delivery_location_latitude"][0], input_df["Delivery_location_longitude"][0]]
        ],
        color="blue",
        weight=3,
        opacity=0.7
    ).add_to(m)

    st.subheader("Delivery Route Map")
    st_folium(m, width=700, height=500)

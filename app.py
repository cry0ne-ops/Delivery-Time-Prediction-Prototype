# app.py
import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# =============================================
# 1. LOAD MODELS
# =============================================
def load_model(filename):
    return joblib.load(filename)

pipeline_lr = load_model("linear_regression_model.pkl")
pipeline_dt = load_model("decision_tree_model.pkl")
pipeline_rf = load_model("random_forest_model.pkl")

models = {
    "Linear Regression": pipeline_lr,
    "Decision Tree": pipeline_dt,
    "Random Forest": pipeline_rf
}

# =============================================
# 2. APP LAYOUT
# =============================================
st.set_page_config(layout="wide")
st.title("Delivery Time Prediction App")
st.write("Predict delivery time and visualize locations on the map.")

# Select model
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# =============================================
# 3. SIDEBAR INPUTS
# =============================================
st.sidebar.header("Enter Order Details:")

def user_input_features():
    data = {
        "Delivery_person_Age": st.sidebar.number_input("Delivery Person Age", 18, 70, 25),
        "Delivery_person_Ratings": st.sidebar.number_input("Delivery Person Ratings", 0.0, 5.0, 4.5),
        "Restaurant_latitude": st.sidebar.number_input("Restaurant Latitude", -90.0, 90.0, 14.6),
        "Restaurant_longitude": st.sidebar.number_input("Restaurant Longitude", -180.0, 180.0, 120.9),
        "Delivery_location_latitude": st.sidebar.number_input("Delivery Latitude", -90.0, 90.0, 14.65),
        "Delivery_location_longitude": st.sidebar.number_input("Delivery Longitude", -180.0, 180.0, 120.95),
        "Weatherconditions": st.sidebar.selectbox("Weather Conditions", ["Sunny", "Cloudy", "Rainy", "Stormy", "Fog"]),
        "Road_traffic_density": st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"]),
        "Type_of_order": st.sidebar.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"]),
        "Type_of_vehicle": st.sidebar.selectbox("Type of Vehicle", ["Bike", "Scooter", "Car"]),
        "multiple_deliveries": st.sidebar.number_input("Multiple Deliveries", 0, 5, 1),
        "Festival": st.sidebar.selectbox("Festival", ["Yes", "No"]),
        "order_day_of_week": st.sidebar.number_input("Order Day of Week (0=Monday)", 0, 6, 0),
        "order_month": st.sidebar.number_input("Order Month", 1, 12, 1),
        "order_hour": st.sidebar.number_input("Order Hour", 0, 23, 12),
        "pickup_hour": st.sidebar.number_input("Pickup Hour", 0, 23, 12),
        "pickup_delay_min": st.sidebar.number_input("Pickup Delay (minutes)", 0, 180, 5)
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
    # 5. MAP VISUALIZATION
    # =============================================
    m = folium.Map(location=[input_df["Restaurant_latitude"][0], input_df["Restaurant_longitude"][0]], zoom_start=12)
    
    # Restaurant marker
    folium.Marker(
        location=[input_df["Restaurant_latitude"][0], input_df["Restaurant_longitude"][0]],
        popup="Restaurant",
        icon=folium.Icon(color="green", icon="cutlery", prefix="fa")
    ).add_to(m)
    
    # Delivery location marker
    folium.Marker(
        location=[input_df["Delivery_location_latitude"][0], input_df["Delivery_location_longitude"][0]],
        popup="Delivery Location",
        icon=folium.Icon(color="red", icon="truck", prefix="fa")
    ).add_to(m)
    
    # Draw line between points
    folium.PolyLine(
        locations=[
            [input_df["Restaurant_latitude"][0], input_df["Restaurant_longitude"][0]],
            [input_df["Delivery_location_latitude"][0], input_df["Delivery_location_longitude"][0]]
        ],
        color="blue",
        weight=3,
        opacity=0.7
    ).add_to(m)
    
    st.subheader("Route Map")
    st_folium(m, width=700, height=500)

# ============================================
# Streamlit App: Delivery Time Prediction with Visualization
# ============================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# 3. Streamlit User Interface
# ============================================

st.title("🚀 Delivery Time Prediction")

st.header("Enter Delivery Details:")

input_data = {
    "Delivery_person_Age": st.number_input("Delivery Person Age", 18, 60),
    "Delivery_person_Ratings": st.number_input("Delivery Person Ratings", 0.0, 5.0, 0.1),
    "Restaurant_latitude": st.number_input("Restaurant Latitude", format="%.6f"),
    "Restaurant_longitude": st.number_input("Restaurant Longitude", format="%.6f"),
    "Delivery_location_latitude": st.number_input("Delivery Location Latitude", format="%.6f"),
    "Delivery_location_longitude": st.number_input("Delivery Location Longitude", format="%.6f"),
    "multiple_deliveries": st.number_input("Multiple Deliveries", 1, 10),
    "order_day_of_week": st.number_input("Order Day of Week (0=Mon, 6=Sun)", 0, 6),
    "order_month": st.number_input("Order Month", 1, 12),
    "order_hour": st.number_input("Order Hour", 0, 23),
    "pickup_hour": st.number_input("Pickup Hour", 0, 23),
    "pickup_delay_min": st.number_input("Pickup Delay (minutes)", 0, 120),
    "Weatherconditions": st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Fog"]),
    "Road_traffic_density": st.selectbox("Traffic", ["Low","Medium","High","Jam"]),
    "Type_of_order": st.selectbox("Order Type", ["Snack","Meal","Drinks","Other"]),
    "Type_of_vehicle": st.selectbox("Vehicle Type", ["Bike","Car","Scooter"]),
    "Festival": st.selectbox("Festival", ["Yes","No"])
}

# ============================================
# 4. Make Predictions
# ============================================

if st.button("Predict Delivery Time"):
    predictions = predict_delivery_time(input_data)
    
    # Display predictions
    st.subheader("Predicted Delivery Times (minutes):")
    st.write(predictions)
    
    # ============================================
    # 5. Visualization
    # ============================================
    st.subheader("Prediction Comparison Chart")
    
    # Convert predictions to DataFrame for plotting
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"])
    
    # Highlight fastest model
    fastest_time = df_pred["Predicted Time"].min()
    
    colors = ['green' if t==fastest_time else 'blue' for t in df_pred["Predicted Time"]]
    
    # Plot bar chart
    fig, ax = plt.subplots()
    df_pred.plot(kind='bar', x='Model', y='Predicted Time', ax=ax, color=colors, legend=False)
    ax.set_ylabel("Predicted Delivery Time (min)")
    ax.set_title("Delivery Time Prediction by Model")
    ax.set_xticklabels(df_pred["Model"], rotation=0)
    
    st.pyplot(fig)
    
    # Show which model predicts fastest
    fastest_model = df_pred.loc[df_pred["Predicted Time"]==fastest_time, "Model"].values[0]
    st.success(f"✅ Fastest Predicted Delivery: {fastest_model} ({fastest_time} min)")

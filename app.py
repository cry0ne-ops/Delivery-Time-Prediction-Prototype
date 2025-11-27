# ============================================
# Streamlit App: Delivery Time Prediction with Model Accuracy
# ============================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================
# 1. Load Dataset & Models
# ============================================

df = pd.read_csv("update dataset (1).csv")  # Update path if needed

# Load preprocessing and models
preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 2. Preprocess Dataset for Evaluation
# ============================================

TARGET = "Time_taken(min)"
FEATURES = [
    "Delivery_person_Age","Delivery_person_Ratings",
    "Restaurant_latitude","Restaurant_longitude",
    "Delivery_location_latitude","Delivery_location_longitude",
    "multiple_deliveries","order_day_of_week","order_month",
    "order_hour","pickup_hour","pickup_delay_min",
    "Weatherconditions","Road_traffic_density",
    "Type_of_order","Type_of_vehicle","Festival"
]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 3. Prediction Function
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
# 4. Streamlit UI
# ============================================

st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")

st.title("🛵 Delivery Time Prediction System")
st.markdown("Enter delivery details to predict time and see model accuracies.")

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)

with col1:
    delivery_person_age = st.number_input("Delivery Person Age", 18, 60)
    delivery_person_rating = st.number_input("Delivery Person Rating", 0.0, 5.0, 0.1)
    pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120)
    order_type = st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"])

with col2:
    # Dummy example lat/lon; ideally, use actual inputs or map widget
    restaurant_lat = st.number_input("Restaurant Latitude", 12.9716)
    restaurant_long = st.number_input("Restaurant Longitude", 77.5946)
    delivery_lat = st.number_input("Delivery Latitude", 12.9352)
    delivery_long = st.number_input("Delivery Longitude", 77.6245)

# ---------------- Predict Button ----------------
if st.button("🚀 Predict Delivery Time"):
    
    input_data = {
        "Delivery_person_Age": delivery_person_age,
        "Delivery_person_Ratings": delivery_person_rating,
        "Restaurant_latitude": restaurant_lat,
        "Restaurant_longitude": restaurant_long,
        "Delivery_location_latitude": delivery_lat,
        "Delivery_location_longitude": delivery_long,
        "multiple_deliveries": 1,
        "order_day_of_week": 3,
        "order_month": 11,
        "order_hour": 18,
        "pickup_hour": 18,
        "pickup_delay_min": pickup_delay,
        "Weatherconditions": "Sunny",
        "Road_traffic_density": "High",
        "Type_of_order": order_type,
        "Type_of_vehicle": "Bike",
        "Festival": "No"
    }

    # ---------- Predictions ----------
    predictions = predict_delivery_time(input_data)
    st.subheader("📊 Predicted Delivery Times (minutes)")
    st.write(predictions)
    
    # Bar chart comparison
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"]).set_index("Model")
    st.bar_chart(df_pred)
    
    # ---------- Model Accuracy ----------
    st.subheader("📈 Model Accuracy on Test Set")
    
    models = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }
    
    metrics_list = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics_list.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
    
    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.2f}"))
    
    # Highlight most accurate
    best_model = metrics_df["RMSE"].idxmin()
    st.success(f"✅ Most Accurate Model Based on RMSE: {best_model}")

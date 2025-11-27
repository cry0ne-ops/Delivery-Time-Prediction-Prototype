# ============================================
# Streamlit App: Delivery Time Prediction with Model Accuracy
# ============================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ============================================
# 1. Load Dataset & Models
# ============================================

# Load dataset (needed for evaluation)
df = pd.read_csv("update dataset (1).csv")  # replace path if needed

# Load preprocessing and models
preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 2. Preprocess Data (same as training)
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

# Train/test split for evaluation
from sklearn.model_selection import train_test_split
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

st.title("🛵 Delivery Time Prediction with Model Accuracy")

st.header("Enter Delivery Details:")

# For simplicity, a few main inputs (expandable sections can be added)
order_type = st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"])
delivery_person_age = st.number_input("Delivery Person Age", 18, 60)
delivery_person_rating = st.number_input("Delivery Person Rating", 0.0, 5.0, 0.1)
pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120)

# Dummy values for other numeric features (you can expand to full form)
input_data = {
    "Delivery_person_Age": delivery_person_age,
    "Delivery_person_Ratings": delivery_person_rating,
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
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

if st.button("🚀 Predict Delivery Time"):
    predictions = predict_delivery_time(input_data)
    st.subheader("Predicted Delivery Times (minutes)")
    st.write(predictions)
    
    # ============================================
    # 5. Evaluate Models on Test Set
    # ============================================
    st.subheader("Model Accuracy on Test Set")
    
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
    
    # Highlight most accurate (lowest RMSE)
    best_model = metrics_df["RMSE"].idxmin()
    st.success(f"✅ Most Accurate Model Based on RMSE: {best_model}")

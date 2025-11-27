# ============================================
# Streamlit App: Robust Delivery Time Prediction
# ============================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("update dataset (1).csv")  # Update path if needed

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ============================================
# 2. Feature Engineering
# ============================================

# Convert Order_Date to datetime
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"].astype(str), errors='coerce')
    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["order_month"] = df["Order_Date"].dt.month

# Function to convert time to HHMM integer
def clean_time_to_hhmm_int(time_str):
    time_str = str(time_str).strip()
    if ':' in time_str:
        try:
            dt_obj = pd.to_datetime(time_str, format='%H:%M:%S').time()
            return dt_obj.hour*100 + dt_obj.minute
        except:
            return np.nan
    else:
        try:
            return int(time_str.zfill(4))
        except:
            return np.nan

# Create Time features
for col in ["Time_Orderd", "Time_Order_picked"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_time_to_hhmm_int)

if "Time_Orderd" in df.columns and "Time_Order_picked" in df.columns:
    df.dropna(subset=["Time_Orderd", "Time_Order_picked"], inplace=True)
    df["Time_Orderd"] = df["Time_Orderd"].astype(int)
    df["Time_Order_picked"] = df["Time_Order_picked"].astype(int)
    df["order_hour"] = df["Time_Orderd"] // 100
    df["pickup_hour"] = df["Time_Order_picked"] // 100
    df["pickup_delay_min"] = ((df["pickup_hour"] - df["order_hour"])*60).clip(lower=0)

# Clean target
if "Time_taken(min)" in df.columns:
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.replace('(min) ', '', regex=False).astype(float)

# ============================================
# 3. Features and Target
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

# Keep only existing features
FEATURES = [f for f in FEATURES if f in df.columns]
X = df[FEATURES]
y = df[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 4. Load Models
# ============================================

preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 5. Prediction Function
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
    # Keep only numeric features present in input
    numeric_features = [f for f in numeric_features if f in df_input.columns]
    df_input[numeric_features] = df_input[numeric_features].astype(float)
    
    return {
        "Linear Regression": round(lr_model.predict(df_input)[0],2),
        "Decision Tree": round(dt_model.predict(df_input)[0],2),
        "Random Forest": round(rf_model.predict(df_input)[0],2)
    }

# ============================================
# 6. Streamlit UI
# ============================================

st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")
st.title("🛵 Delivery Time Prediction System")
st.markdown("Enter delivery details to predict time and see model accuracies.")

# Input columns
col1, col2 = st.columns(2)
with col1:
    delivery_person_age = st.number_input("Delivery Person Age", 18, 60)
    delivery_person_rating = st.number_input("Delivery Person Rating", 0.0, 5.0, 0.1)
    pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120)
    order_type = st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"])
with col2:
    restaurant_lat = st.number_input("Restaurant Latitude", 12.9716)
    restaurant_long = st.number_input("Restaurant Longitude", 77.5946)
    delivery_lat = st.number_input("Delivery Latitude", 12.9352)
    delivery_long = st.number_input("Delivery Longitude", 77.6245)

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

    # Predictions
    predictions = predict_delivery_time(input_data)
    st.subheader("📊 Predicted Delivery Times (minutes)")
    st.write(predictions)
    
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"]).set_index("Model")
    st.bar_chart(df_pred)
    
    # Model accuracy
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
    
    best_model = metrics_df["RMSE"].idxmin()
    st.success(f"✅ Most Accurate Model Based on RMSE: {best_model}")

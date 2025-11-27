# ============================================
# Streamlit App: Delivery Time Prediction + Distance + Map
# ============================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import random
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_csv("update dataset (1).csv")  # Update path if needed
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ============================================
# 2. Feature Engineering
# ============================================

if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"].astype(str), errors='coerce')
    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["order_month"] = df["Order_Date"].dt.month

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

if "Time_taken(min)" in df.columns:
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.replace('(min) ', '', regex=False).astype(float)

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
FEATURES = [f for f in FEATURES if f in df.columns]
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 3. Load Models
# ============================================

preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 4. Distance Calculation
# ============================================

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

# ============================================
# 5. Random Data Generator
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
        "Type_of_order": random.choice(["Meat","Vegetables","Meat or Vegetables"]),
        "Type_of_vehicle": random.choice(["Bike","Car","Scooter"]),
        "Festival": random.choice(["Yes","No"])
    }

# ============================================
# 6. Prediction Function
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
    numeric_features = [f for f in numeric_features if f in df_input.columns]
    df_input[numeric_features] = df_input[numeric_features].astype(float)
    return {
        "Linear Regression": round(lr_model.predict(df_input)[0],2),
        "Decision Tree": round(dt_model.predict(df_input)[0],2),
        "Random Forest": round(rf_model.predict(df_input)[0],2)
    }

# ============================================
# 7. Map Function
# ============================================

def create_delivery_map(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon, predictions, distance_km):
    mid_lat = (restaurant_lat + delivery_lat) / 2
    mid_lon = (restaurant_lon + delivery_lon) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13)
    
    folium.Marker(
        [restaurant_lat, restaurant_lon],
        popup=f"Restaurant",
        tooltip="Restaurant",
        icon=folium.Icon(color="green", icon="cutlery", prefix="fa")
    ).add_to(m)

    folium.Marker(
        [delivery_lat, delivery_lon],
        popup=f"Delivery Location",
        tooltip="Delivery",
        icon=folium.Icon(color="red", icon="motorcycle", prefix="fa")
    ).add_to(m)

    folium.PolyLine(
        locations=[[restaurant_lat, restaurant_lon], [delivery_lat, delivery_lon]],
        color="blue", weight=3, opacity=0.7
    ).add_to(m)

    folium.Marker(
        [(restaurant_lat + delivery_lat)/2, (restaurant_lon + delivery_lon)/2],
        icon=folium.DivIcon(html=f"""<div style="font-size: 14pt; color: black">
                                      Distance: {distance_km:.2f} km<br>
                                      LR: {predictions['Linear Regression']} min, 
                                      DT: {predictions['Decision Tree']} min, 
                                      RF: {predictions['Random Forest']} min
                                      </div>""")
    ).add_to(m)
    return m

# ============================================
# 8. Streamlit UI
# ============================================

st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")
st.title("🛵 Delivery Time Prediction with Map")
st.markdown("Generate random delivery details or enter your own to predict delivery times and see the delivery map.")

# ---- Initialize session_state keys ----
for key, default_value in {
    "multiple_deliveries": 1,
    "order_day_of_week": 3,
    "order_month": 11,
    "order_hour": 18,
    "pickup_hour": 18,
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "High",
    "Type_of_vehicle": "Bike",
    "Festival": "No"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---- Random Data Button ----
if st.button("🎲 Generate Random Delivery Details"):
    random_data = generate_random_delivery_data()
    for key, value in random_data.items():
        st.session_state[key] = value
    st.success("✅ Random delivery details generated!")

# ---- Input Fields ----
col1, col2 = st.columns(2)
with col1:
    delivery_person_age = st.number_input("Delivery Person Age", 18, 60, value=st.session_state["Delivery_person_Age"], key="Delivery_person_Age")
    delivery_person_rating = st.number_input("Delivery Person Rating", 0.0, 5.0, value=st.session_state["Delivery_person_Ratings"], step=0.1, key="Delivery_person_Ratings")
    pickup_delay = st.number_input("Pickup Delay (minutes)", 0, 120, value=st.session_state["pickup_delay_min"], key="pickup_delay_min")
    order_type = st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"], index=["Meat","Vegetables","Meat or Vegetables"].index(st.session_state["Type_of_order"]), key="Type_of_order")
with col2:
    restaurant_lat = st.number_input("Restaurant Latitude", 12.90, 13.00, value=st.session_state["Restaurant_latitude"], format="%.6f", key="Restaurant_latitude")
    restaurant_long = st.number_input("Restaurant Longitude", 77.55, 77.65, value=st.session_state["Restaurant_longitude"], format="%.6f", key="Restaurant_longitude")
    delivery_lat = st.number_input("Delivery Latitude", 12.90, 13.00, value=st.session_state["Delivery_location_latitude"], format="%.6f", key="Delivery_location_latitude")
    delivery_long = st.number_input("Delivery Longitude", 77.55, 77.65, value=st.session_state["Delivery_location_longitude"], format="%.6f", key="Delivery_location_longitude")

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

    # Distance calculation
    distance_km = haversine_distance(
        input_data["Restaurant_latitude"],
        input_data["Restaurant_longitude"],
        input_data["Delivery_location_latitude"],
        input_data["Delivery_location_longitude"]
    )
    st.subheader(f"🛣️ Distance: {distance_km:.2f} km")

    # Predictions
    predictions = predict_delivery_time(input_data)
    st.subheader("📊 Predicted Delivery Times (minutes)")
    st.write(predictions)
    df_pred = pd.DataFrame(list(predictions.items()), columns=["Model","Predicted Time"]).set_index("Model")
    st.bar_chart(df_pred)

    # Accuracy metrics
    st.subheader("📈 Model Accuracy on Test Set")
    models = {"Linear Regression": lr_model, "Decision Tree": dt_model, "Random Forest": rf_model}
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

    # Map
    st.subheader("🗺️ Delivery Map")
    delivery_map = create_delivery_map(
        input_data["Restaurant_latitude"],
        input_data["Restaurant_longitude"],
        input_data["Delivery_location_latitude"],
        input_data["Delivery_location_longitude"],
        predictions,
        distance_km
    )
    st_data = st_folium(delivery_map, width=700, height=500)

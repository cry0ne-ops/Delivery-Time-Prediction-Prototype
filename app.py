# ============================================
# Streamlit Delivery Time Dashboard
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import folium
from streamlit_folium import st_folium
from openrouteservice import Client
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt

# ============================================
# 1. Load Preprocessor + Models
# ============================================
preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

models = {
    "Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

# ============================================
# 2. Load Dataset (for charts / test set)
# ============================================
df = pd.read_csv("update dataset (1).csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert times and dates
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d/%m/%Y", errors="coerce")
    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["order_month"] = df["Order_Date"].dt.month

if "Time_Orderd" in df.columns and "Time_Order_picked" in df.columns:
    def clean_time(time_str):
        try:
            h, m, *_ = map(int, str(time_str).split(":"))
            return h*100 + m
        except: 
            try: return int(time_str)
            except: return np.nan
    df["Time_Orderd"] = df["Time_Orderd"].apply(clean_time)
    df["Time_Order_picked"] = df["Time_Order_picked"].apply(clean_time)
    df.dropna(subset=["Time_Orderd","Time_Order_picked"], inplace=True)
    df["order_hour"] = df["Time_Orderd"] // 100
    df["pickup_hour"] = df["Time_Order_picked"] // 100
    df["pickup_delay_min"] = ((df["pickup_hour"] - df["order_hour"])*60).clip(lower=0)

if "Time_taken(min)" in df.columns:
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.replace('(min) ','',regex=False).astype(float)

# Features & target
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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# 3. ORS API Key
# ============================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="

# ============================================
# 4. Streamlit Session Defaults
# ============================================
default_values = {
    "Delivery_person_Age": 25,
    "Delivery_person_Ratings": 4.0,
    "pickup_delay_min": 5,
    "Type_of_order": "Meat",
    "Type_of_vehicle": "Bike",
    "Festival": "No",
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "multiple_deliveries": 1,
    "order_day_of_week": 0,
    "order_month": 1,
    "order_hour": 12,
    "pickup_hour": 12,
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Low"
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# 5. Random Data Generator
# ============================================
def generate_random_delivery_data():
    return {
        "Delivery_person_Age": random.randint(18, 60),
        "Delivery_person_Ratings": round(random.uniform(2.5,5.0),1),
        "Restaurant_latitude": round(random.uniform(12.90,13.00),6),
        "Restaurant_longitude": round(random.uniform(77.55,77.65),6),
        "Delivery_location_latitude": round(random.uniform(12.90,13.00),6),
        "Delivery_location_longitude": round(random.uniform(77.55,77.65),6),
        "multiple_deliveries": random.randint(1,5),
        "order_day_of_week": random.randint(0,6),
        "order_month": random.randint(1,12),
        "order_hour": random.randint(8,22),
        "pickup_hour": random.randint(8,23),
        "pickup_delay_min": random.randint(0,30),
        "Weatherconditions": random.choice(["Sunny","Cloudy","Rainy","Stormy","Fog"]),
        "Road_traffic_density": random.choice(["Low","Medium","High","Jam"]),
        "Type_of_order": random.choice(["Meat","Vegetables","Meat or Vegetables"]),
        "Type_of_vehicle": random.choice(["Bike","Car","Scooter"]),
        "Festival": random.choice(["Yes","No"])
    }

# ============================================
# 6. Prediction Function (Preprocessor Wired)
# ============================================
def predict_delivery_time(input_data, preprocessor, models):
    df_input = pd.DataFrame([input_data])
    try:
        X_processed = preprocessor.transform(df_input)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return None

    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X_processed)[0]
            predictions[name] = round(float(pred),2)
        except Exception as e:
            predictions[name] = None
            st.error(f"Prediction failed for {name}: {e}")
    return predictions

# ============================================
# 7. ORS Route + Distance
# ============================================
@st.cache_data(ttl=600)
def get_ors_route(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    client = Client(key=ORS_API_KEY)
    coords = [[restaurant_long, restaurant_lat],[delivery_long, delivery_lat]]
    try:
        return client.directions(coords, profile='driving-car', format='geojson')
    except:
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def visualize_route_simple(lat1, lon1, lat2, lon2):
    m = folium.Map(location=[(lat1+lat2)/2,(lon1+lon2)/2], zoom_start=13)
    folium.Marker([lat1, lon1], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([lat2, lon2], tooltip="Delivery Location", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([[lat1, lon1],[lat2, lon2]], color="blue", weight=3, opacity=0.8).add_to(m)
    return m

# ============================================
# 8. Streamlit UI (Dashboard Style)
# ============================================
st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")
st.title("🛵 Delivery Time Predictor Dashboard")
st.markdown("Glassmorphic cards, ORS route, distances, predictions, and charts.")

# Columns layout: left=inputs, right=outputs
left_col, right_col = st.columns([1,1.4], gap="large")

with left_col:
    st.subheader("📋 Delivery Details")
    if st.button("🎲 Generate Random Delivery Details"):
        random_data = generate_random_delivery_data()
        for key, val in random_data.items():
            st.session_state[key] = val
        st.success("Random details generated!")

    st.number_input("Delivery Person Age", min_value=18, max_value=60, key="Delivery_person_Age")
    st.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, step=0.1, key="Delivery_person_Ratings")
    st.number_input("Pickup Delay (minutes)", min_value=0, max_value=120, key="pickup_delay_min")
    st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"], key="Type_of_order")
    st.selectbox("Type of Vehicle", ["Bike","Car","Scooter"], key="Type_of_vehicle")
    st.selectbox("Festival", ["Yes","No"], key="Festival")

    st.markdown("**Supplier & Delivery Location**")
    st.number_input("Supplier Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="Restaurant_latitude")
    st.number_input("Supplier Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="Restaurant_longitude")
    st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="Delivery_location_latitude")
    st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="Delivery_location_longitude")

    if st.button("🚀 Predict Delivery Time"):
        input_data = {key: st.session_state[key] for key in default_values.keys()}
        st.session_state["predictions"] = predict_delivery_time(input_data, preprocessor, models)

        # Model metrics
        metrics_list = []
        for name, model in models.items():
            try:
                y_pred = model.predict(preprocessor.transform(X_test))
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            except:
                rmse = mae = r2 = float("nan")
            metrics_list.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
        st.session_state["metrics_df"] = pd.DataFrame(metrics_list).set_index("Model")

with right_col:
    st.subheader("⏱ Predictions & KPIs")
    if "predictions" in st.session_state:
        preds = st.session_state["predictions"]
        for name, val in preds.items():
            st.metric(name, f"{val} min")
    else:
        st.info("Run prediction to see results.")

    st.subheader("📏 Distances")
    lat_r = st.session_state["Restaurant_latitude"]
    lon_r = st.session_state["Restaurant_longitude"]
    lat_c = st.session_state["Delivery_location_latitude"]
    lon_c = st.session_state["Delivery_location_longitude"]

    straight_km = haversine_km(lat_r, lon_r, lat_c, lon_c)
    st.metric("Straight-line Distance", f"{straight_km:.2f} km")

    route_data = get_ors_route(lat_r, lon_r, lat_c, lon_c)
    if route_data:
        try:
            feat = route_data["features"][0]
            seg0 = feat["properties"]["segments"][0]
            driving_km = seg0["distance"]/1000
            driving_min = seg0["duration"]/60
            st.metric("ORS Driving Distance", f"{driving_km:.2f} km")
            st.metric("ORS Drive Time", f"{driving_min:.1f} min")
        except:
            st.info("ORS route available but parsing failed.")
    else:
        st.info("ORS route not available, using straight-line only.")

    st.subheader("🗺️ Delivery Route Map")
    if route_data:
        map_center = [(lat_r+lat_c)/2,(lon_r+lon_c)/2]
        m = folium.Map(location=map_center, zoom_start=13)
        folium.GeoJson(route_data).add_to(m)
        folium.Marker([lat_r, lon_r], tooltip="Supplier", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([lat_c, lon_c], tooltip="Customer", icon=folium.Icon(color='red')).add_to(m)
    else:
        m = visualize_route_simple(lat_r, lon_r, lat_c, lon_c)
    st_folium(m, width=700, height=380)

    st.subheader("📈 Charts")
    # Chart 1: Actual vs Predicted
    try:
        chosen_model = "Random Forest"
        preds_test = rf_model.predict(preprocessor.transform(X_test))
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds_test, alpha=0.5)
        minv, maxv = min(y_test.min(), preds_test.min()), max(y_test.max(), preds_test.max())
        ax.plot([minv, maxv],[minv, maxv], linestyle='--', color='red')
        ax.set_xlabel("Actual Time (min)")
        ax.set_ylabel("Predicted Time (min)")
        ax.set_title("Actual vs Predicted — Random Forest")
        st.pyplot(fig)
    except:
        st.info("Could not generate Actual vs Predicted chart.")

    # Chart 3: Avg Delivery Time by Order Type
    try:
        agg = df.groupby("Type_of_order")[TARGET].mean()
        fig2, ax2 = plt.subplots()
        ax2.bar(agg.index.astype(str), agg.values)
        ax2.set_ylabel("Avg Delivery Time (min)")
        ax2.set_title("Avg Delivery Time by Order Type")
        st.pyplot(fig2)
    except:
        st.info("Could not generate Avg Delivery Time chart.")

    # Chart 5: Model Accuracy
    if "metrics_df" in st.session_state:
        metrics_df = st.session_state["metrics_df"]
        fig3, axes = plt.subplots(2,1,figsize=(6,6))
        metrics_df["RMSE"].plot(kind="bar", ax=axes[0], color="skyblue", title="Model RMSE")
        metrics_df["MAE"].plot(kind="bar", ax=axes[1], color="lightgreen", title="Model MAE")
        st.pyplot(fig3)
        st.markdown("**R² Scores:**")
        for idx, row in metrics_df.iterrows():
            st.markdown(f"- {idx}: {row['R²']:.3f}")

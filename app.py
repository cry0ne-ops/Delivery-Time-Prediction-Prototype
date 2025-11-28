# ============================================
# Streamlit App: Persistent Delivery Time Prediction with ORS Map
# ============================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import random
import folium
from streamlit_folium import st_folium
from openrouteservice import Client
import plotly.express as px

# ============================================
# 1. ORS API Key
# ============================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="

# ============================================
# 2. Load Dataset
# ============================================
df = pd.read_csv("update dataset (1).csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ============================================
# 3. Feature Engineering
# ============================================
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"].astype(str), format="%d/%m/%Y", errors="coerce"
    )
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
# 4. Load Models
# ============================================
preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# 5. Session State Initialization
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
# 6. Random Data Generator
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
# 7. Prediction Function
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
# 8. ORS Route with Caching
# ============================================
@st.cache_data(ttl=600)
def get_ors_route(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    client = Client(key=ORS_API_KEY)
    coords = [[restaurant_long, restaurant_lat], [delivery_long, delivery_lat]]
    try:
        return client.directions(coords, profile='driving-car', format='geojson')
    except:
        return None

def visualize_route_simple(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    map_center = [(restaurant_lat + delivery_lat)/2, (restaurant_long + delivery_long)/2]
    m = folium.Map(location=map_center, zoom_start=13)
    folium.Marker([restaurant_lat, restaurant_long], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([delivery_lat, delivery_long], tooltip="Delivery Location", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([(restaurant_lat, restaurant_long), (delivery_lat, delivery_long)],
                    color="blue", weight=3, opacity=0.8).add_to(m)
    return m

# ============================================
# 9. Streamlit UI
# ============================================
st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")
st.title("🛵 Persistent Delivery Time Prediction with ORS Map")
st.markdown("Generate random delivery details or enter your own to predict delivery times and visualize the delivery route.")

# ---- Random Data Button ----
if st.button("🎲 Generate Random Delivery Details"):
    random_data = generate_random_delivery_data()
    for key, value in random_data.items():
        st.session_state[key] = value
    st.success("✅ Random delivery details generated!")

# ---- Input Fields ----
col1, col2 = st.columns(2)
with col1:
    st.number_input("Delivery Person Age", min_value=18, max_value=60, key="Delivery_person_Age")
    st.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, step=0.1, key="Delivery_person_Ratings")
    st.number_input("Pickup Delay (minutes)", min_value=0, max_value=120, key="pickup_delay_min")
    st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"], key="Type_of_order")
    st.selectbox("Type of Vehicle", ["Bike","Car","Scooter"], key="Type_of_vehicle")
    st.selectbox("Festival", ["Yes","No"], key="Festival")
with col2:
    st.number_input("Restaurant Latitude", min_value=12.90, max_value=13.00, format="%.6f", key="Restaurant_latitude")
    st.number_input("Restaurant Longitude", min_value=77.55, max_value=77.65, format="%.6f", key="Restaurant_longitude")
    st.number_input("Delivery Latitude", min_value=12.90, max_value=13.00, format="%.6f", key="Delivery_location_latitude")
    st.number_input("Delivery Longitude", min_value=77.55, max_value=77.65, format="%.6f", key="Delivery_location_longitude")

# ---- Predict Button ----
if st.button("🚀 Predict Delivery Time"):
    input_data = {key: st.session_state[key] for key in default_values.keys()}
    st.session_state["predictions"] = predict_delivery_time(input_data)

    # compute accuracy metrics
    models = {"Linear Regression": lr_model, "Decision Tree": dt_model, "Random Forest": rf_model}
    metrics_list = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics_list.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
    st.session_state["metrics_df"] = pd.DataFrame(metrics_list).set_index("Model")

# ---- Display Predictions (Enhanced UI) ----
if "predictions" in st.session_state:
    st.subheader("📊 Predicted Delivery Times (minutes)")
    preds = st.session_state["predictions"]
    preds_df = pd.DataFrame(list(preds.items()), columns=["Model", "Predicted_Time"])

    # Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{preds['Linear Regression']} min")
    col2.metric("Decision Tree", f"{preds['Decision Tree']} min")
    col3.metric("Random Forest", f"{preds['Random Forest']} min")

    # Fastest prediction
    fastest = preds_df.loc[preds_df["Predicted_Time"].idxmin()]
    st.success(f"🚀 Fastest Model: {fastest['Model']} ({fastest['Predicted_Time']} min)")

    # Plotly bar chart
    fig = px.bar(
        preds_df,
        x="Model",
        y="Predicted_Time",
        text="Predicted_Time",
        color="Predicted_Time",
        color_continuous_scale="Viridis",
        title="Predicted Delivery Times"
    )
    fig.update_traces(texttemplate='%{text:.2f} min', textposition='outside')
    fig.update_layout(yaxis_title="Minutes", xaxis_title="Model", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- Model Accuracy Section ----
if "metrics_df" in st.session_state:
    st.subheader("📈 Model Accuracy on Test Set")
    st.dataframe(st.session_state["metrics_df"].style.format("{:.2f}"))

    best_model = st.session_state["metrics_df"]["RMSE"].idxmin()
    st.info(f"🏆 Most Accurate Model Based on RMSE: **{best_model}**")

    # Plotly accuracy chart
    fig2 = px.bar(
        st.session_state["metrics_df"].reset_index(),
        x="Model",
        y="RMSE",
        text="RMSE",
        title="RMSE Comparison"
    )
    fig2.update_traces(texttemplate='%{text:.2f}', textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

# ---- Map Visualization ----
st.subheader("🗺️ Delivery Route Visualization")
route_data = get_ors_route(
    st.session_state["Restaurant_latitude"],
    st.session_state["Restaurant_longitude"],
    st.session_state["Delivery_location_latitude"],
    st.session_state["Delivery_location_longitude"]
)

if route_data:
    map_center = [
        (st.session_state["Restaurant_latitude"] + st.session_state["Delivery_location_latitude"]) / 2,
        (st.session_state["Restaurant_longitude"] + st.session_state["Delivery_location_longitude"]) / 2
    ]
    m = folium.Map(location=map_center, zoom_start=13)
    folium.GeoJson(route_data, name="Route").add_to(m)
    folium.Marker(
        [st.session_state["Restaurant_latitude"], st.session_state["Restaurant_longitude"]],
        tooltip="Restaurant", icon=folium.Icon(color='green')
    ).add_to(m)
    folium.Marker(
        [st.session_state["Delivery_location_latitude"], st.session_state["Delivery_location_longitude"]],
        tooltip="Delivery Location", icon=folium.Icon(color='red')
    ).add_to(m)
else:
    m = visualize_route_simple(
        st.session_state["Restaurant_latitude"],
        st.session_state["Restaurant_longitude"],
        st.session_state["Delivery_location_latitude"],
        st.session_state["Delivery_location_longitude"]
    )

st_folium(m, width=900, height=550)

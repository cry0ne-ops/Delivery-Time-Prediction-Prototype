# ============================================
# 9. Streamlit UI (Glassmorphic Cards + Distances)
# ============================================
st.set_page_config(page_title="Delivery Time Predictor 🚀", layout="wide")
st.markdown(
    """
    <style>
    /* Glassmorphic card */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        color: #eaeaea;
        margin-bottom: 16px;
    }
    .glass-header {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .kpi {
        font-size: 28px;
        font-weight: 800;
        margin-top: 4px;
    }
    .muted {
        color: #bfc6cc;
        font-size: 13px;
    }
    /* Make Streamlit background slightly dark for contrast */
    .stApp {
        background: linear-gradient(135deg, rgba(10,20,30,1) 0%, rgba(20,28,40,1) 100%);
        color: #eaeaea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛵 Delivery Time Predictor — Glass Dashboard")
st.markdown("Generate delivery details or enter your own. Predictions, model metrics, distances and route map are shown in glass cards.")

# ---------------------------
# Utility: haversine distance
# ---------------------------
import math
def haversine_km(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ---------------------------
# Layout: Inputs (left) | Results (right)
# ---------------------------
left_col, right_col = st.columns([1, 1.4], gap="large")

# ---------------------------
# LEFT: Inputs + Controls
# ---------------------------
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📋 Delivery Details</div>', unsafe_allow_html=True)

    # Random generator (keeps existing behavior)
    if st.button("🎲 Generate Random Delivery Details"):
        random_data = generate_random_delivery_data()
        for key, value in random_data.items():
            st.session_state[key] = value
        st.success("✅ Random delivery details generated!")

    st.number_input("Delivery Person Age", min_value=18, max_value=60, key="Delivery_person_Age")
    st.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, step=0.1, key="Delivery_person_Ratings")
    st.number_input("Pickup Delay (minutes)", min_value=0, max_value=120, key="pickup_delay_min")
    st.selectbox("Type of Order", ["Meat","Vegetables","Meat or Vegetables"], key="Type_of_order")
    st.selectbox("Type of Vehicle", ["Bike","Car","Scooter"], key="Type_of_vehicle")
    st.selectbox("Festival", ["Yes","No"], key="Festival")

    st.markdown("---")
    st.markdown('<div class="glass-header">📍 Locations</div>', unsafe_allow_html=True)
    st.number_input("Supplier Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="Restaurant_latitude")
    st.number_input("Supplier Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="Restaurant_longitude")
    st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="Delivery_location_latitude")
    st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="Delivery_location_longitude")

    st.markdown("---")
    # Predict button
    if st.button("🚀 Predict Delivery Time", use_container_width=True):
        input_data = {key: st.session_state[key] for key in default_values.keys()}
        st.session_state["predictions"] = predict_delivery_time(input_data)

        # model metrics -- unchanged logic
        models = {"Linear Regression": lr_model, "Decision Tree": dt_model, "Random Forest": rf_model}
        metrics_list = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics_list.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2})
        st.session_state["metrics_df"] = pd.DataFrame(metrics_list).set_index("Model")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# RIGHT: Cards (Predictions, Metrics, Distances, Map, Summary)
# ---------------------------
with right_col:
    # Predictions Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">⏱ Predictions</div>', unsafe_allow_html=True)
    if "predictions" not in st.session_state:
        st.markdown('<div class="muted">No predictions yet — click Predict.</div>', unsafe_allow_html=True)
    else:
        preds = st.session_state["predictions"]
        # Show each model as mini KPI inside the card
        kpi_cols = st.columns(len(preds))
        for (model_name, value), k in zip(preds.items(), kpi_cols):
            k.markdown(f"<div class='muted'>{model_name}</div>", unsafe_allow_html=True)
            k.markdown(f"<div class='kpi'>{value:.1f} min</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Metrics Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📈 Model Metrics (Test Set)</div>', unsafe_allow_html=True)
    if "metrics_df" in st.session_state:
        # show dataframe compactly
        st.dataframe(st.session_state["metrics_df"].style.format("{:.2f}"))
        best_model = st.session_state["metrics_df"]["RMSE"].idxmin()
        st.markdown(f"<div class='muted'>Best model (RMSE): <strong>{best_model}</strong></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">No metrics available yet. Run a prediction to compute metrics.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Distances Card: Haversine + ORS (if available)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">📏 Distances</div>', unsafe_allow_html=True)

    lat_r = float(st.session_state["Restaurant_latitude"])
    lon_r = float(st.session_state["Restaurant_longitude"])
    lat_c = float(st.session_state["Delivery_location_latitude"])
    lon_c = float(st.session_state["Delivery_location_longitude"])

    # Straight-line
    try:
        straight_km = haversine_km(lat_r, lon_r, lat_c, lon_c)
        st.markdown(f"<div class='muted'>Straight-line distance</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi'>{straight_km:.2f} km</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div class='muted'>Straight-line distance could not be calculated: {e}</div>", unsafe_allow_html=True)

    # ORS driving distance & duration (if route_data available)
    route_data = get_ors_route(lat_r, lon_r, lat_c, lon_c)
    if route_data:
        # openrouteservice geojson structure: features[0].properties.segments[0].distance (meters), duration (seconds)
        try:
            feat = route_data.get("features", [None])[0]
            props = feat.get("properties", {}) if feat else {}
            segments = props.get("segments", [{}])
            summary = props.get("summary", {})
            seg0 = segments[0] if segments and len(segments) > 0 else {}
            driving_m = seg0.get("distance") or summary.get("distance")
            driving_s = seg0.get("duration") or summary.get("duration")
            if driving_m is not None:
                driving_km = driving_m / 1000.0
                st.markdown(f"<div class='muted'>ORS driving distance</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi'>{driving_km:.2f} km</div>", unsafe_allow_html=True)
            if driving_s is not None:
                driving_min = driving_s / 60.0
                st.markdown(f"<div class='muted'>ORS estimated drive time</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi'>{driving_min:.1f} min</div>", unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div class='muted'>ORS route available but parsing failed: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="muted">ORS route not available — showing straight-line distance only.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Map + Summary row
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">🗺️ Route Map & Summary</div>', unsafe_allow_html=True)
    map_col, sum_col = st.columns([1.6, 0.9])

    with map_col:
        # Use route_data (cached) if possible, otherwise fallback
        if route_data:
            try:
                map_center = [(lat_r + lat_c)/2.0, (lon_r + lon_c)/2.0]
                m = folium.Map(location=map_center, zoom_start=13)
                folium.GeoJson(route_data, name="Route").add_to(m)
                folium.Marker([lat_r, lon_r], tooltip="Supplier", icon=folium.Icon(color='green')).add_to(m)
                folium.Marker([lat_c, lon_c], tooltip="Customer", icon=folium.Icon(color='red')).add_to(m)
            except Exception:
                m = visualize_route_simple(lat_r, lon_r, lat_c, lon_c)
        else:
            m = visualize_route_simple(lat_r, lon_r, lat_c, lon_c)

        st_folium(m, width=700, height=430)

    with sum_col:
        st.markdown('<div class="muted">Delivery Summary</div>', unsafe_allow_html=True)
        st.markdown(f"**Driver Age:** {st.session_state['Delivery_person_Age']}")
        st.markdown(f"**Rating:** {st.session_state['Delivery_person_Ratings']}")
        st.markdown(f"**Order Type:** {st.session_state['Type_of_order']}")
        st.markdown(f"**Vehicle:** {st.session_state['Type_of_vehicle']}")
        st.markdown(f"**Pickup Delay:** {st.session_state['pickup_delay_min']} min")
        # Traffic & Weather could be missing if not in FEATURES
        if "Road_traffic_density" in st.session_state:
            st.markdown(f"**Traffic:** {st.session_state['Road_traffic_density']}")
        if "Weatherconditions" in st.session_state:
            st.markdown(f"**Weather:** {st.session_state['Weatherconditions']}")

    st.markdown('</div>', unsafe_allow_html=True)

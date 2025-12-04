# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt
import numpy as np

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Predictive Vehicle Maintenance",
    page_icon="🚛",
    layout="wide",
)

# ----------------- CONSTANTS & FEATURES -----------------
DATA_PATH = Path("predictive_truck_maintenance_2000.csv")
MODEL_PATH = Path("truck_maintenance_regressor.pkl")

# This must match the features used when training your model (exclude technician_id).
# If you retrain, ensure the pipeline uses exactly this list (or update here to match model).
FEATURES = [
    "vehicle_model","year_bought","total_km_run","km_after_last_service",
    "avg_daily_km_est","engine_temperature_c","vibrations_level","oil_life_percent","battery_health_percent",
    "route_type","load_profile","ambient_temp_c","service_type","parts_in_stock_status",
    "technician_experience_years","current_queue_length","shift_hours_remaining",
    "brake_pad_thickness_mm","tyre_health_percent","fuel_efficiency_kmpl","approx_past_services"
]

# Allowed categories as per your request:
ROUTE_TYPES = ["Highway", "City", "Mixed", "Off-road"]
LOAD_PROFILES = ["Light", "Medium", "Heavy"]
PARTS_STATUS = ["Available", "Out of Stock"]
SERVICE_TYPES = []  # populated from CSV if possible (keeps flexibility)

# ----------------- STYLES -----------------
def inject_css():
    st.markdown(
        """
        <style>
        /* SAFE DARK TRUCK BACKGROUND WITH OVERLAY */
        .main {
            background:
                linear-gradient(rgba(3,7,18,0.82), rgba(3,7,18,0.82)),
                url('https://images.pexels.com/photos/2199293/pexels-photo-2199293.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #e6eef8;
        }
        .glass-card {
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: rgba(255,255,255,0.04);
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
            padding: 16px;
            margin-bottom: 18px;
        }
        .section-title { font-size:1.25rem; font-weight:700; color:#e6eef8; margin-bottom:8px; }
        .badge { padding:6px 10px; border-radius:10px; font-weight:700; }
        .good { background: rgba(34,197,94,0.18); color:#4ade80; }
        .warn { background: rgba(234,179,8,0.18); color:#facc15; }
        .bad  { background: rgba(239,68,68,0.18); color:#f87171; }
        label { color: #e6eef8 !important; font-weight:600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# ----------------- LOAD DATA & MODEL -----------------
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Normalize column names (just in case)
        df.columns = [c.strip() for c in df.columns]
        return df
    return pd.DataFrame(columns=FEATURES + ["days_until_next_service", "truck_number_plate", "service_date"])

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

# populate service types if present
if "service_type" in df.columns:
    SERVICE_TYPES = sorted(df["service_type"].dropna().unique().tolist())
if not SERVICE_TYPES:
    SERVICE_TYPES = ["Full", "Minor", "Major"]

# ----------------- HELPERS -----------------
def get_latest_record_for_plate(plate: str):
    plate = plate.strip().upper()
    if plate == "":
        return None, None
    if "truck_number_plate" not in df.columns:
        return None, None
    match = df[df["truck_number_plate"].astype(str).str.upper() == plate]
    if match.empty:
        return None, None
    match_sorted = match.sort_values("service_date", ascending=False)
    return match_sorted.iloc[0], match_sorted

def health_level(value, good, warn):
    if value >= good:
        return "🟢 Healthy", "good"
    elif value >= warn:
        return "🟡 Moderate", "warn"
    else:
        return "🔴 Critical", "bad"

def safe_get(latest, col, default=None):
    """Return typed default from latest record or None so fields can be blank on load."""
    if latest is not None and col in latest and pd.notna(latest[col]):
        return latest[col]
    return default

def prepare_input_row(raw_row: dict):
    """Ensure all FEATURES exist in the row and are in expected order/format for model."""
    row = {}
    for f in FEATURES:
        if f in raw_row and raw_row[f] is not None:
            row[f] = raw_row[f]
        else:
            # sensible defaults
            if f in ["vehicle_model","route_type","load_profile","service_type","parts_in_stock_status"]:
                row[f] = "Unknown"
            elif f in ["year_bought","technician_experience_years","approx_past_services"]:
                row[f] = 0
            else:
                row[f] = 0.0
    return row


# ----------------- UI PAGES -----------------
st.sidebar.title("🚛 Predictive Vehicle Maintenance")
page = st.sidebar.radio("Navigate", ["Dashboard", "Predict Next Service", "Service History", "Data Explorer"])
st.sidebar.markdown("---")
st.sidebar.caption("Built for truck service centers • Demo prototype")

# ----------------- DASHBOARD -----------------
if page == "Dashboard":
    st.markdown("""
        <div class="glass-card">
          <h1 style="margin:0;">🚛 Fleet Maintenance Intelligence</h1>
          <p style="margin-top:6px;color:#cbd5e1;">Insights, health monitoring, and predictive servicing for your fleet.</p>
        </div>
    """, unsafe_allow_html=True)

    # Metrics row
    colA, colB, colC, colD = st.columns(4)
    def metric_card(title, value, helper=""):
        st.markdown(f"""
            <div class="glass-card">
              <div style="font-weight:700">{title}</div>
              <div style="font-size:1.6rem;color:#38bdf8;font-weight:800">{value}</div>
              <div style="color:#aab4c3">{helper}</div>
            </div>
        """, unsafe_allow_html=True)

    avg_next = df["days_until_next_service"].mean() if "days_until_next_service" in df.columns and not df.empty else float("nan")
    metric_card("Avg. Days Until Next Service", f"{avg_next:.1f} days" if not np.isnan(avg_next) else "N/A", "Fleet average")
    critical_pct = (df["days_until_next_service"] <= 15).mean()*100 if "days_until_next_service" in df.columns and not df.empty else 0
    metric_card("Trucks in Critical Window", f"{critical_pct:.1f} %", "≤ 15 days")
    metric_card("Avg KM Since Last Service", f"{df['km_after_last_service'].mean():,.0f} km" if "km_after_last_service" in df.columns and not df.empty else "N/A")
    metric_card("Unique Trucks in System", df["truck_number_plate"].nunique() if "truck_number_plate" in df.columns else 0)

    st.markdown("---")
    st.markdown("### Quick glance at sample records")
    st.dataframe(df.head(20), use_container_width=True)

# ----------------- PREDICT NEXT SERVICE -----------------
elif page == "Predict Next Service":
    st.markdown("<h1 style='color:white;'>🔮 Predict Next Service Interval</h1>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>Choose mode</div>", unsafe_allow_html=True)
    mode = st.selectbox("Select mode", ["-- Select --", "Existing Truck (auto-fill)", "New Truck (manual entry)"], index=0)
    st.markdown("</div>", unsafe_allow_html=True)

    latest = None
    history_df = None

    if mode == "Existing Truck (auto-fill)":
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        plate = st.text_input("Enter Truck Number Plate (exact)", key="fetch_plate")
        fetch_btn = st.button("Fetch History", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if fetch_btn and plate.strip():
            latest, history_df = get_latest_record_for_plate(plate)
            if latest is None:
                st.warning("No history found for this plate. Switch to 'New Truck' to enter details manually.")
            else:
                st.success("History found; form will be pre-filled from latest record.")
    elif mode == "New Truck (manual entry)":
        plate = None
    else:
        # nothing selected -> don't show form
        st.info("Choose a mode above to continue.")
        st.stop()

    # ---- Show form (either autofilled or blank) ----
    with st.form("predict_form"):

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🚚 Vehicle Information</div>", unsafe_allow_html=True)
        v1, v2, v3, v4 = st.columns(4)

        # vehicle_model: include a top placeholder option for new truck
        model_options = ["-- Select model --"] + sorted(df["vehicle_model"].dropna().unique().tolist()) if "vehicle_model" in df.columns else ["-- Select model --"]
        with v1:
            sel_vehicle_model = safe_get(latest, "vehicle_model", None)
            vehicle_model = st.selectbox("Vehicle Model", model_options, index=model_options.index(sel_vehicle_model) if sel_vehicle_model in model_options else 0)

        with v2:
            year_bought = st.number_input("Year Bought", min_value=1990, max_value=2026, value=int(safe_get(latest, "year_bought", 2020)))

        with v3:
            total_km_run = st.number_input("Total KM Run", min_value=0, max_value=5_000_000, value=int(safe_get(latest, "total_km_run", 0)), step=500)

        with v4:
            km_after_last_service = st.number_input("KM After Last Service", min_value=0, max_value=500_000, value=int(safe_get(latest, "km_after_last_service", 0)), step=100)

        st.markdown("</div>", unsafe_allow_html=True)

        # Engine & Sensors
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧪 Engine & Sensor Health</div>", unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)

        with e1:
            engine_temperature_c = st.slider("Engine Temp (°C)", 60, 140, int(safe_get(latest, "engine_temperature_c", 90)), step=1)
            txt, badge = health_level(140 - engine_temperature_c, 80, 40)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        with e2:
            vibrations_level = st.slider("Vibration Level", 0.0, 10.0, float(safe_get(latest, "vibrations_level", 5.0)), step=0.1)
            txt, badge = health_level(10 - vibrations_level, 7, 4)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        with e3:
            oil_life_percent = st.slider("Oil Life (%)", 0, 100, int(safe_get(latest, "oil_life_percent", 60)), step=1)
            txt, badge = health_level(oil_life_percent, 60, 30)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        with e4:
            battery_health_percent = st.slider("Battery Health (%)", 0, 100, int(safe_get(latest, "battery_health_percent", 75)), step=1)
            txt, badge = health_level(battery_health_percent, 70, 40)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Additional performance indicators
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚙️ Additional Performance Indicators</div>", unsafe_allow_html=True)
        x1, x2, x3, x4, x5, x6 = st.columns(6)

        with x1:
            avg_daily_km_est = st.number_input("Avg Daily KM Estimate", min_value=0, max_value=5000, value=int(safe_get(latest, "avg_daily_km_est", 0)))

        with x2:
            ambient_temp_c = st.slider("Ambient Temp (°C)", -10, 50, int(safe_get(latest, "ambient_temp_c", 25)), step=1)

        with x3:
            brake_pad_thickness_mm = st.slider("Brake Pad Thickness (mm)", 1, 30, int(safe_get(latest, "brake_pad_thickness_mm", 12)), step=1)

        with x4:
            tyre_health_percent = st.slider("Tyre Health (%)", 0, 100, int(safe_get(latest, "tyre_health_percent", 70)), step=1)

        with x5:
            fuel_efficiency_kmpl = st.slider("Fuel Efficiency (kmpl)", 1.0, 12.0, float(safe_get(latest, "fuel_efficiency_kmpl", 3.5)), step=0.1)

        with x6:
            approx_past_services = st.number_input("Past Service Count", min_value=0, max_value=200, value=int(safe_get(latest, "approx_past_services", 0)))

        st.markdown("</div>", unsafe_allow_html=True)

        # Workshop / categorical context
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧰 Workshop Context & Categories</div>", unsafe_allow_html=True)
        w1, w2, w3, w4 = st.columns(4)

        with w1:
            route_type = st.selectbox("Route Type", ["-- Select --"] + ROUTE_TYPES, index=(ROUTE_TYPES.index(safe_get(latest, "route_type"))+1) if safe_get(latest, "route_type") in ROUTE_TYPES else 0)

        with w2:
            load_profile = st.selectbox("Load Profile", ["-- Select --"] + LOAD_PROFILES, index=(LOAD_PROFILES.index(safe_get(latest, "load_profile"))+1) if safe_get(latest, "load_profile") in LOAD_PROFILES else 0)

        with w3:
            service_type = st.selectbox("Service Type", ["-- Select --"] + SERVICE_TYPES, index=(SERVICE_TYPES.index(safe_get(latest, "service_type"))+1) if safe_get(latest, "service_type") in SERVICE_TYPES else 0)

        with w4:
            parts_in_stock_status = st.selectbox("Parts in Stock?", ["-- Select --"] + PARTS_STATUS, index=(PARTS_STATUS.index(safe_get(latest, "parts_in_stock_status"))+1) if safe_get(latest, "parts_in_stock_status") in PARTS_STATUS else 0)

        st.markdown("</div>", unsafe_allow_html=True)

        # Workshop numeric context
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>👷 Workshop Numeric Context</div>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)

        with q1:
            technician_experience_years = st.number_input("Technician Experience (years)", min_value=0, max_value=80, value=int(safe_get(latest, "technician_experience_years", 0)))

        with q2:
            current_queue_length = st.number_input("Current Queue Length", min_value=0, max_value=200, value=int(safe_get(latest, "current_queue_length", 0)))

        with q3:
            shift_hours_remaining = st.number_input("Shift Hours Remaining", min_value=0.0, max_value=24.0, value=float(safe_get(latest, "shift_hours_remaining", 0.0)), step=0.5)

        with q4:
            # placeholder to visually balance layout
            st.write("")

        st.markdown("</div>", unsafe_allow_html=True)

        # submit
        submitted = st.form_submit_button("🚀 Predict Next Service")

    # ---------- handle submission ----------
    if submitted:
        if model is None:
            st.error("Model not found. Place your trained `truck_maintenance_regressor.pkl` in the app folder.")
        else:
            # compose raw row from inputs
            raw_row = {
                "vehicle_model": vehicle_model if vehicle_model and vehicle_model != "-- Select --" else "Unknown",
                "year_bought": year_bought,
                "total_km_run": total_km_run,
                "km_after_last_service": km_after_last_service,
                "avg_daily_km_est": avg_daily_km_est,
                "engine_temperature_c": engine_temperature_c,
                "vibrations_level": vibrations_level,
                "oil_life_percent": oil_life_percent,
                "battery_health_percent": battery_health_percent,
                "route_type": route_type if route_type and route_type != "-- Select --" else "Unknown",
                "load_profile": load_profile if load_profile and load_profile != "-- Select --" else "Unknown",
                "ambient_temp_c": ambient_temp_c,
                "service_type": service_type if service_type and service_type != "-- Select --" else "Unknown",
                "parts_in_stock_status": parts_in_stock_status if parts_in_stock_status and parts_in_stock_status != "-- Select --" else "Unknown",
                "technician_experience_years": technician_experience_years,
                "current_queue_length": current_queue_length,
                "shift_hours_remaining": shift_hours_remaining,
                "brake_pad_thickness_mm": brake_pad_thickness_mm,
                "tyre_health_percent": tyre_health_percent,
                "fuel_efficiency_kmpl": fuel_efficiency_kmpl,
                "approx_past_services": approx_past_services,
            }

            clean_row = prepare_input_row(raw_row)
            X = pd.DataFrame([clean_row])

            try:
                pred_days = float(model.predict(X)[0])
                pred_hours = pred_days * 24.0
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown(f"""
                    <h2 style='color:#38bdf8'>📅 Predicted Next Service: <strong>{pred_days:.1f} days</strong> — (~{pred_hours:.0f} hours)</h2>
                    <p style='color:#cbd5e1'>This estimate uses the saved model pipeline and the inputs you provided.</p>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.exception(f"Prediction failed: {e}")

# ----------------- SERVICE HISTORY -----------------
elif page == "Service History":
    st.title("📜 Service History Lookup")
    plate = st.text_input("Enter Truck Number Plate (exact)", key="history_plate")
    if st.button("Get History") and plate:
        latest, hist = get_latest_record_for_plate(plate)
        if hist is None or hist.empty:
            st.error("No history found for this plate.")
        else:
            st.success(f"Found {len(hist)} record(s).")
            st.dataframe(hist.sort_values("service_date", ascending=False), use_container_width=True)

# ----------------- DATA EXPLORER -----------------
elif page == "Data Explorer":
    st.title("📂 Data Explorer")
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="predictive_truck_maintenance_2000.csv", mime="text/csv")

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

# PAGE CONFIG
st.set_page_config(page_title="Predictive Truck Maintenance", page_icon="🚛", layout="wide")


# GLASS UI CSS

def inject_css():
    st.markdown("""
        <style>
        /* CLEAN DARK BACKGROUND – NO TRUCK IMAGE */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0f172a 100%) !important;
            background-size: cover;
            background-attachment: fixed;
        }

        .glass-card {
            backdrop-filter: blur(14px);
            background: rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.4rem;
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        }

        .section-title {
            font-size: 1.3rem;
            color: #e2e8f0;
            font-weight: 600;
        }

        label {
            color: #e5e7eb !important;
            font-weight: 600;
        }

        .badge {
            padding: 6px 12px;
            border-radius: 12px;
            font-weight: 600;
        }
        .good { background: rgba(34,197,94,0.2); color: #4ade80; }
        .warn { background: rgba(234,179,8,0.2); color: #facc15; }
        .bad  { background: rgba(239,68,68,0.2); color: #f87171; }
        </style>
    """, unsafe_allow_html=True)



inject_css()


# LOAD DATA & MODEL

DATA_PATH = Path("truck_dataset.csv")
MODEL_PATH = Path("truck_maintenance_regressor.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()


# Model feature set (must match training)
model_features = [
    "vehicle_model",
    "year_bought",
    "total_km_run",
    "km_after_last_service",
    "avg_daily_km_est",
    "engine_temperature_c",
    "vibrations_level",
    "oil_life_percent",
    "battery_health_percent",
    "brake_pad_thickness_mm",
    "tyre_health_percent",
    "fuel_efficiency_kmpl",
    "ambient_temp_c",
    "approx_past_services",
    "parts_changed_last_service",
    "route_type",
    "load_profile",
]


# UTILITY FUNCTIONS

def get_latest_record_for_plate(plate: str):
    plate = plate.strip().upper()
    match = df[df["truck_number_plate"].str.upper() == plate]
    if len(match) == 0:
        return None
    return match.sort_values("service_date", ascending=False).iloc[0]


def safe_value(value, default=None):
    if pd.isna(value) or value == "None" or value == "":
        return default
    return value


def health_level(value, good, warn):
    if value >= good:
        return ("🟢 Healthy", "good")
    elif value >= warn:
        return ("🟡 Moderate", "warn")
    else:
        return ("🔴 Critical", "bad")

# SIDEBAR

st.sidebar.title("🚛 Predictive Vehicle Maintenance")
page = st.sidebar.radio("Navigate", ["Dashboard", "Predict Next Service", "Service History", "Data Explorer"])
st.sidebar.markdown("---")


# DASHBOARD

if page == "Dashboard":

    st.markdown("<h1 style='color:white;'>📊 Fleet Maintenance Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Days Until Service", f"{df['days_until_next_service'].mean():.1f}")

    with col2:
        critical = (df["days_until_next_service"] <= 15).mean() * 100
        st.metric("Critical Trucks (%)", f"{critical:.1f}%")

    with col3:
        st.metric("Avg KM After Last Service", f"{df['km_after_last_service'].mean():,.0f}")

    with col4:
        st.metric("Unique Trucks", df["truck_number_plate"].nunique())

    st.markdown("### Service Window Distribution")
    chart = alt.Chart(df).mark_area(opacity=0.6).encode(
        x=alt.X("days_until_next_service", bin=True),
        y="count()",
        color=alt.value("#38bdf8"),
    )
    st.altair_chart(chart, use_container_width=True)



# PREDICT NEXT SERVICE

elif page == "Predict Next Service":

    st.markdown("<h1 style='color:white;'>🔮 Predict Next Service</h1>", unsafe_allow_html=True)

    plate = st.text_input("Enter Truck Number Plate (optional)")
    fetched = st.button("Fetch History")

    latest = None
    if fetched and plate.strip():
        latest = get_latest_record_for_plate(plate)
        if latest is None:
            st.info("No history found, entering new truck.")
        else:
            st.success("History found! Autofilling values.")

    with st.form("predict"):

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🚚 Vehicle Information</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            vehicle_model = st.selectbox(
                "Vehicle Model",
                df["vehicle_model"].unique(),
                index=df["vehicle_model"].unique().tolist().index(safe_value(latest["vehicle_model"], df["vehicle_model"].unique()[0])) if latest is not None else 0
            )
        with c2:
            year_bought = st.number_input(
                "Year Bought", 2010, 2025,
                value=int(safe_value(latest["year_bought"], 2018)) if latest is not None else 2018
            )
        with c3:
            total_km_run = st.number_input(
                "Total KM Run", 1000, 1_500_000,
                value=int(safe_value(latest["total_km_run"], 500000)) if latest is not None else 500000
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ENGINE HEALTH
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧪 Engine & Sensor Health</div>", unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            engine_temp = st.slider(
                "Engine Temp (°C)", 60, 140,
                int(safe_value(latest["engine_temperature_c"], 90)) if latest is not None else 90
            )

        with e2:
            vibrations = st.slider(
                "Vibration Level", 0.0, 10.0,
                float(safe_value(latest["vibrations_level"], 5.0)) if latest is not None else 5.0
            )

        with e3:
            oil_life = st.slider(
                "Oil Life (%)", 0, 100,
                int(safe_value(latest["oil_life_percent"], 60)) if latest is not None else 60
            )

        with e4:
            battery = st.slider(
                "Battery Health (%)", 0, 100,
                int(safe_value(latest["battery_health_percent"], 75)) if latest is not None else 75
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # EXTRA FEATURES
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚙️ Additional Indicators</div>", unsafe_allow_html=True)

        x1, x2, x3, x4 = st.columns(4)

        with x1:
            km_after_last_service = st.number_input(
                "KM After Last Service", 0, 200000,
                int(safe_value(latest["km_after_last_service"], 10000)) if latest is not None else 10000
            )

        with x2:
            avg_daily_km = st.number_input(
                "Avg Daily KM", 0, 2000,
                int(safe_value(latest["avg_daily_km_est"], 200)) if latest is not None else 200
            )

        with x3:
            brake_pad = st.number_input(
                "Brake Pad Thickness (mm)", 1, 30,
                int(safe_value(latest["brake_pad_thickness_mm"], 10)) if latest is not None else 10
            )

        with x4:
            tyre = st.number_input(
                "Tyre Health (%)", 1, 100,
                int(safe_value(latest["tyre_health_percent"], 70)) if latest is not None else 70
            )

        y1, y2, y3 = st.columns(3)

        with y1:
            fuel = st.number_input(
                "Fuel Efficiency (kmpl)", 1.0, 8.0,
                float(safe_value(latest["fuel_efficiency_kmpl"], 4.0)) if latest is not None else 4.0
            )

        with y2:
            ambient = st.number_input(
                "Ambient Temp (°C)", -5, 50,
                int(safe_value(latest["ambient_temp_c"], 30)) if latest is not None else 30
            )

        with y3:
            past_services = st.number_input(
                "Past Service Count", 0, 50,
                int(safe_value(latest["approx_past_services"], 5)) if latest is not None else 5
            )

        # Categorical
        p1, p2, p3 = st.columns(3)

        with p1:
            route_type = st.selectbox(
                "Route Type",
                df["route_type"].unique(),
                index=df["route_type"].unique().tolist().index(safe_value(latest["route_type"], df["route_type"].unique()[0])) if latest is not None else 0
            )

        with p2:
            load_profile = st.selectbox(
                "Load Profile",
                df["load_profile"].unique(),
                index=df["load_profile"].unique().tolist().index(safe_value(latest["load_profile"], df["load_profile"].unique()[0])) if latest is not None else 0
            )

        with p3:
            parts_changed = st.selectbox(
                "Parts Changed Last Service",
                df["parts_changed_last_service"].unique(),
                index=df["parts_changed_last_service"].unique().tolist().index(safe_value(latest["parts_changed_last_service"], df["parts_changed_last_service"].unique()[0])) if latest is not None else 0
            )

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:

        row = pd.DataFrame([{
            "vehicle_model": vehicle_model,
            "year_bought": year_bought,
            "total_km_run": total_km_run,
            "km_after_last_service": km_after_last_service,
            "avg_daily_km_est": avg_daily_km,
            "engine_temperature_c": engine_temp,
            "vibrations_level": vibrations,
            "oil_life_percent": oil_life,
            "battery_health_percent": battery,
            "brake_pad_thickness_mm": brake_pad,
            "tyre_health_percent": tyre,
            "fuel_efficiency_kmpl": fuel,
            "ambient_temp_c": ambient,
            "approx_past_services": past_services,
            "parts_changed_last_service": parts_changed,
            "route_type": route_type,
            "load_profile": load_profile,
        }])[model_features]  # FORCE CORRECT ORDER

        try:
            days = model.predict(row)[0]
            st.success(f"📅 Next Service In **{days:.1f} days**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



# SERVICE HISTORY

elif page == "Service History":
    st.title("📜 Service History Lookup")
    plate = st.text_input("Enter Number Plate")
    if st.button("Search"):
        m = df[df["truck_number_plate"].str.upper() == plate.upper()]
        if len(m) == 0:
            st.error("No records found.")
        else:
            st.success(f"{len(m)} records found.")
            st.dataframe(m)



# DATA EXPLORER

elif page == "Data Explorer":
    st.title("📂 Dataset")
    st.dataframe(df)

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt


# PAGE CONFIG

st.set_page_config(
    page_title="Predictive Vehicle Maintenance",
    page_icon="🚛",
    layout="wide",
)


# GLASS UI

def inject_css():
    st.markdown(
        """
        <style>

        /* SAFE BACKGROUND IMAGE */
        .main {
            background:
                linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                url('https://images.pexels.com/photos/2199293/pexels-photo-2199293.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        .glass-card {
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            background: rgba(255, 255, 255, 0.10);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.15);
            padding: 1.4rem 1.7rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
            margin-bottom: 1.4rem;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.4rem;
        }

        .badge {
            padding: 6px 12px;
            border-radius: 12px;
            font-weight: 600;
        }
        .good { background: rgba(34,197,94,0.2); color: #4ade80; }
        .warn { background: rgba(234,179,8,0.2); color: #facc15; }
        .bad  { background: rgba(239,68,68,0.2); color: #f87171; }

        label {
            color: #e5e7eb !important;
            font-weight: 600;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# LOAD DATA & MODEL

DATA_PATH = Path("predictive_truck_maintenance_2000.csv")
MODEL_PATH = Path("truck_maintenance_regressor.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()



# SIDEBAR NAVIGATION

st.sidebar.title("🚛 Predictive Vehicle Maintenance")

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Predict Next Service", "Service History", "Data Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Built for truck service centers • Demo prototype")


# UTILITY FUNCTIONS

def get_latest_record_for_plate(plate: str):
    plate = plate.strip().upper()
    match = df[df["truck_number_plate"].str.upper() == plate]
    if match.empty:
        return None, None
    latest = match.sort_values("service_date", ascending=False).iloc[0]
    return latest, match

def health_level(value, good, warn):
    if value >= good: return "🟢 Healthy", "good"
    elif value >= warn: return "🟡 Moderate", "warn"
    else: return "🔴 Critical", "bad"

def autofill(latest, col):
    """Return saved value or None for new truck (so fields stay empty)."""
    if latest is not None and col in latest:
        return latest[col]
    return None


# PAGE 1 — DASHBOARD

if page == "Dashboard":

    st.markdown("""
        <div style='padding: 2rem; 
             border-radius: 1rem;
             background: linear-gradient(135deg,#1e293b 0%,#0f172a 60%,#020617 100%);
             color: white;
             box-shadow: 0 18px 45px rgba(0,0,0,0.5);'>
            <h1 style='font-size:2.7rem;font-weight:800;'>🚛 Fleet Maintenance Intelligence</h1>
            <p style='font-size:1.1rem;opacity:0.85;'>Live insights • Smart predictions • Modern workshop analytics</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def metric_card(title, value, subtitle, icon):
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);padding:1.3rem;border-radius:1rem;
                     backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.12);
                     box-shadow:0 8px 22px rgba(0,0,0,0.45);color:#e2e8f0;">
            <div style='font-size:1.2rem;opacity:0.7;'>{icon} {title}</div>
            <div style='font-size:2rem;font-weight:700;color:#38bdf8;'>{value}</div>
            <div style='font-size:.80rem;opacity:.6;'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        metric_card("Avg. Days Until Service",
                    f"{df['days_until_next_service'].mean():.1f}", 
                    "Fleet-wide average", "⏳")
    with col2:
        metric_card("Critical Trucks",
                    f"{(df['days_until_next_service']<=15).mean()*100:.1f}%",
                    "Need urgent service", "🚨")
    with col3:
        metric_card("Avg KM Since Service",
                    f"{df['km_after_last_service'].mean():,.0f}",
                    "Per service event", "📍")
    with col4:
        metric_card("Unique Trucks", 
                    df["truck_number_plate"].nunique(),
                    "Tracked in system", "🚚")

    st.markdown("---")

    st.markdown("### 📊 Service Window Distribution")
    chart1 = alt.Chart(df).mark_area(opacity=0.7).encode(
        x=alt.X("days_until_next_service", bin=True),
        y="count()",
        color=alt.value("#38bdf8"),
    ).properties(height=300)
    st.altair_chart(chart1, use_container_width=True)

    st.markdown("### 🧠 Engine Temp vs Vibrations")
    chart2 = alt.Chart(df).mark_circle(size=60).encode(
        x="engine_temperature_c",
        y="vibrations_level",
        color=alt.Color("days_until_next_service", scale=alt.Scale(scheme="turbo")),
        tooltip=["truck_number_plate", "engine_temperature_c", "vibrations_level"],
    ).properties(height=350)
    st.altair_chart(chart2, use_container_width=True)




# PAGE 2 — PREDICT NEXT SERVICE

elif page == "Predict Next Service":

    st.markdown("<h1 style='color:white;'>🔮 Predict Next Service Interval</h1>", unsafe_allow_html=True)

    # -------- FETCH HISTORY BOX --------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    cA, cB = st.columns([3, 1])

    with cA:
        plate = st.text_input("Enter Truck Number Plate (optional for new truck)")
    with cB:
        fetch = st.button("Fetch History", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    latest = None
    history = None

    if fetch and plate.strip():
        latest, history = get_latest_record_for_plate(plate)
        if latest is None:
            st.info("No history found. New truck detected.")
        else:
            st.success("History found. Autofilling details.")

    # -------------- MAIN FORM --------------
    with st.form("predict_form"):

        # ========== VEHICLE INFO ==========
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🚚 Vehicle Information</div>", unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)

        with v1:
            vehicle_model = st.selectbox(
                "Vehicle Model",
                df["vehicle_model"].unique(),
                index=list(df["vehicle_model"].unique()).index(autofill(latest, "vehicle_model"))
                if latest is not None else None,
            )
        with v2:
            year_bought = st.number_input(
                "Year Bought",
                min_value=2010, max_value=2025,
                value=int(autofill(latest, "year_bought")) if latest else None,
            )
        with v3:
            total_km_run = st.number_input(
                "Total KM Run",
                min_value=0, max_value=2_000_000,
                value=int(autofill(latest, "total_km_run")) if latest else None,
            )
        with v4:
            km_after_last_service = st.number_input(
                "KM After Last Service",
                min_value=0, max_value=200_000,
                value=int(autofill(latest, "km_after_last_service")) if latest else None,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ========== ENGINE & SENSOR HEALTH ==========
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧪 Engine & Sensor Health</div>", unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)

        # Engine Temp
        with e1:
            engine_temp = st.slider(
                "Engine Temp (°C)", 60, 140,
                value=int(autofill(latest, "engine_temperature_c")) if latest else 90,
                step=1,
            )
            txt, badge = health_level(140 - engine_temp, 80, 40)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        # Vibrations
        with e2:
            vibrations = st.slider(
                "Vibration Level", 0.0, 10.0,
                value=float(autofill(latest, "vibrations_level")) if latest else 5.0,
                step=0.1,
            )
            txt, badge = health_level(10 - vibrations, 7, 4)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        # Oil Life
        with e3:
            oil_life = st.slider(
                "Oil Life (%)", 0, 100,
                value=int(autofill(latest, "oil_life_percent")) if latest else 60,
                step=1,
            )
            txt, badge = health_level(oil_life, 60, 30)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        # Battery
        with e4:
            battery = st.slider(
                "Battery Health (%)", 0, 100,
                value=int(autofill(latest, "battery_health_percent")) if latest else 75,
                step=1,
            )
            txt, badge = health_level(battery, 70, 40)
            st.markdown(f"<span class='badge {badge}'>{txt}</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # EXTRA MODEL FEATURES
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⚙️ Additional Performance Indicators</div>", unsafe_allow_html=True)

        x1, x2, x3, x4, x5, x6 = st.columns(6)

        with x1:
            avg_daily_km_est = st.number_input(
                "Avg Daily KM Estimate",
                min_value=0, max_value=2000,
                value=int(autofill(latest, "avg_daily_km_est")) if latest else None,
            )

        with x2:
            ambient_temp = st.slider(
                "Ambient Temp (°C)",
                min_value=-5, max_value=50,
                value=int(autofill(latest, "ambient_temp_c")) if latest else 30,
                step=1,
            )

        with x3:
            brake_pad = st.slider(
                "Brake Pad Thickness (mm)",
                min_value=1, max_value=25,
                value=int(autofill(latest, "brake_pad_thickness_mm")) if latest else 12,
                step=1,
            )

        with x4:
            tyre_health = st.slider(
                "Tyre Health (%)",
                min_value=0, max_value=100,
                value=int(autofill(latest, "tyre_health_percent")) if latest else 70,
                step=1,
            )

        with x5:
            fuel_eff = st.slider(
                "Fuel Efficiency (kmpl)",
                min_value=1.0, max_value=7.0,
                value=float(autofill(latest, "fuel_efficiency_kmpl")) if latest else 3.5,
                step=0.1,
            )

        with x6:
            past_services = st.number_input(
                "Past Service Count",
                min_value=0, max_value=50,
                value=int(autofill(latest, "approx_past_services")) if latest else None,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ========== WORKSHOP CONTEXT ==========
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🧰 Workshop Context</div>", unsafe_allow_html=True)

        w1, w2, w3, w4 = st.columns(4)

        with w1:
            service_type = st.selectbox(
                "Service Type",
                df["service_type"].unique(),
                index=list(df["service_type"].unique()).index(autofill(latest, "service_type"))
                if latest else None,
            )
        with w2:
            technician_exp = st.number_input(
                "Technician Experience (years)",
                min_value=0, max_value=40,
                value=int(autofill(latest, "technician_experience_years")) if latest else None,
            )
        with w3:
            queue_len = st.number_input(
                "Queue Length",
                min_value=0, max_value=50,
                value=int(autofill(latest, "current_queue_length")) if latest else None,
            )
        with w4:
            shift_hours = st.number_input(
                "Shift Hours Left",
                min_value=0.0, max_value=12.0,
                value=float(autofill(latest, "shift_hours_remaining")) if latest else None,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # SUBMIT BUTTON
        submitted = st.form_submit_button("🚀 Predict Next Service")

    # RUN PREDICTION 
    if submitted:

        if model is None:
            st.error("Model not found.")
        else:
            row = {
                "vehicle_model": vehicle_model,
                "year_bought": year_bought,
                "total_km_run": total_km_run,
                "km_after_last_service": km_after_last_service,
                "engine_temperature_c": engine_temp,
                "vibrations_level": vibrations,
                "oil_life_percent": oil_life,
                "battery_health_percent": battery,
                "avg_daily_km_est": avg_daily_km_est,
                "ambient_temp_c": ambient_temp,
                "brake_pad_thickness_mm": brake_pad,
                "tyre_health_percent": tyre_health,
                "fuel_efficiency_kmpl": fuel_eff,
                "approx_past_services": past_services,
                "service_type": service_type,
                "technician_experience_years": technician_exp,
                "current_queue_length": queue_len,
                "shift_hours_remaining": shift_hours,
            }

            pred = model.predict(pd.DataFrame([row]))[0]

            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown(f"""
                <h2 style='color:#38bdf8;font-size:2rem;'>📅 Next Service In  
                    <span style='font-size:2.8rem;'>{pred:.1f} days</span>
                </h2>
                <p style='color:#cbd5e1;'>Predicted using engine health, usage patterns, and workshop context.</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)



# PAGE 3 — SERVICE HISTORY

elif page == "Service History":
    st.title("📜 Service History Lookup")
    plate = st.text_input("Enter Truck Number Plate")

    if st.button("Get History") and plate:
        latest, hist = get_latest_record_for_plate(plate)
        if hist is None:
            st.error("No history found.")
        else:
            st.success(f"Found {len(hist)} record(s).")
            st.dataframe(hist, use_container_width=True)



# PAGE 4 — DATA EXPLORER

elif page == "Data Explorer":
    st.title("📂 Data Explorer")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        file_name="predictive_truck_maintenance_2000.csv",
        mime="text/csv",
    )

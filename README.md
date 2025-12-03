🚛 Predictive Vehicle Maintenance for Trucks
AI-powered system to predict next servicing, part failures, and maintenance needs
📝 Overview

This project predicts next service date (in days) and potential component failures for trucks using:

Engine temperature

Vibration levels

Oil life

Battery health

Brake pad wear

Tyre condition

Fuel efficiency

Load & route behavior

Workshop factors (queue, shift hours, technician experience)

Historical service records

It includes:
✔ A machine-learning model (Regression)
✔ A synthetic dataset (2000+ records)
✔ A beautiful Streamlit App
✔ History autofill using truck number plate
✔ Advanced UI with glassmorphism + truck background
✔ Real-time health analysis badges (Healthy / Warning / Critical)

🧠 Key Features
🔮 Predict Next Service / Part Failure

The system predicts:

Days until next service

Health risk bucket (Low / Medium / High / Critical)

🚚 Autofill Using Truck Number Plate

If a truck exists in historical records → auto-populates all fields

If new → user enters details manually

📚 Service History

Search by truck number plate

View full timeline of previous services

See technician details, parts changed, and sensor state

📊 Dashboard

Avg days to next service

Critical trucks (%)

Avg km after last service

Number of fleet trucks

Service window distribution

Engine temperature vs vibration heatmap

Vehicle model distribution chart

🧪 Engine & Sensor Health Levels

Each sensor is analyzed:

🟢 Healthy

🟡 Moderate

🔴 Critical

Includes:

Engine Temp

Vibration Level

Oil Life

Battery Health

Brake Thickness

Tyre Health

Fuel Efficiency

🧰 Tech Stack
Component	Technology
Frontend	Streamlit, HTML/CSS, Glassmorphism UI
ML Model	Scikit-Learn Regression Model
Data	2000+ Synthetic Truck Maintenance Records
Language	Python 3.10+
Storage	CSV-based history records
Deployment	Streamlit Cloud / Local execution

📁 Project Structure
Predictive-Vehicle-Maintenance/
│── app.py
│── predictive_truck_maintenance_2000.csv
│── truck_maintenance_regressor.pkl
│── README.md
│── assets/
│     └── truck_bg.jpg (optional)
│── model_training_notebook.ipynb (optional)

🗂️ Dataset Description

Your dataset includes (2000+ rows):

🔧 Vehicle Info

truck_number_plate

vehicle_model

year_bought

route_type

load_profile

🧪 Sensor Data

engine_temperature_c

vibrations_level

oil_life_percent

battery_health_percent

brake_pad_thickness_mm

tyre_health_percent

fuel_efficiency_kmpl

ambient_temp_c

🔧 Operational Data

total_km_run

km_after_last_service

avg_daily_km_est

🏭 Workshop / Technician Data

service_type

technician_id

technician_experience_years

current_queue_length

shift_hours_remaining

parts_in_stock_status

🎯 Target Column

days_until_next_service

⚙️ How to Run the Project
1️⃣ Create or use an existing environment
conda activate base

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit application
streamlit run app.py

4️⃣ Visit the app in your browser
http://localhost:8501

🧩 ML Model Training

The regression model:

one-hot encodes categorical data

scales numerical features

trains using RandomForestRegressor / GradientBoosting / LinearReg
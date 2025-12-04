# 🚛 Predictive Vehicle Maintenance for Trucks  
### 🔧 AI-powered analytics to predict servicing schedules, component failures & fleet health.

---

## 🧭 Overview  
This project uses **Machine Learning** + **real-time truck sensor data** to predict:

- 📅 **Days until next maintenance**
- 🚨 **Risk level** (Critical / High / Medium / Low)
- 🔩 **Probability of component failures**
- 🔄 **Workshop impact** (queue, technician skill, stocks)
- 📜 **Full truck service history**
- 🧠 Smart **autofill using truck number plate**

A modern **Streamlit application** with a **dark UI**, **glassmorphism**, and an optional **truck background image** powers the user experience.

---

## ⭐ Key Features  

### 🔮 1. Predictive Maintenance  
Machine learning model predicts:
- Days to next service  
- Maintenance urgency  
- Fault probability based on sensors  

### 🚚 2. Truck Plate Autofill  
When a plate number is entered:
- If **existing** → autofill past details  
- If **new** → manual entry  
Makes the system fast & efficient for real workshops.

### 🧪 3. Real-Time Sensor Health Levels  
Each sensor shows visual badges:
- 🟢 **Healthy**  
- 🟡 **Warning**  
- 🔴 **Critical**  

Evaluated for:
- Engine Temperature  
- Vibrations  
- Oil Life  
- Battery Health  
- Brake Pad Thickness  
- Tyre Health  
- Fuel Efficiency  
- Ambient Temperature Impact  

### 📊 4. Dynamic Dashboard  
Includes:
- ⭐ Average days until next service  
- 🚨 % of critical trucks  
- 📍 Avg KM after last service  
- 🚛 Count of unique trucks  
- 📉 Days-to-service distribution  
- 🌡️ Temperature vs Vibration heatmap  
- 🏷️ Trucks by model  

### 📚 5. Full Service History  
- View all past service logs  
- Most recent service snapshot  
- Technician details  
- Parts replaced  
- Service type  
- Timeline of repairs  

### 🗂️ 6. Data Explorer  
- View entire dataset  
- Filter & analyze  
- Download CSV  

---

## 🧠 Machine Learning Model  

### Model type  
Uses **Scikit-Learn** Regression Pipeline with:
- Numeric scaling  
- One-hot encoding  
- RandomForestRegressor (recommended)  

### Trained on  
✔ 2000+ synthetic truck maintenance records  

### Target predicted  
**days_until_next_service**

### Important engineered features  
- avg_daily_km_est  
- ambient_temp_c  
- brake_pad_thickness_mm  
- tyre_health_percent  
- fuel_efficiency_kmpl  
- approx_past_services  
- workshop metadata  

---

## 🧰 Tech Stack  

### Backend & ML  
- Python 3.10+  
- Pandas / NumPy  
- Scikit-Learn  
- Joblib  

### Frontend  
- Streamlit  
- Altair Charts  
- Custom CSS (dark + glassmorphism)  
- Background truck image  

### Storage  
- CSV for dataset  
- PKL model file  

---

---

## 📊 Dataset Description  

### 🚛 Vehicle Info  
- truck_number_plate  
- vehicle_model  
- year_bought  
- route_type  
- load_profile  

### 🔧 Sensors  
- engine_temperature_c  
- vibrations_level  
- oil_life_percent  
- battery_health_percent  
- brake_pad_thickness_mm  
- tyre_health_percent  
- fuel_efficiency_kmpl  
- ambient_temp_c  

### 🛣️ Operational Data  
- total_km_run  
- km_after_last_service  
- avg_daily_km_est  


### 🎯 Target  
- days_until_next_service  

---

## 🏗️ Architecture  
User Input / History Autofill
↓
Feature Preprocessing (Scaling + Encoding)
↓
ML Regression Model
↓
Predicted Days Until Next Service
↓
Risk Level Assignment
↓
Displayed in Streamlit App


---

## ▶️ How to Run  

### 1️⃣ Install dependencies 
pip install -r requirements.txt 

### 2️⃣ Run the app 
streamlit run app.py 

### 3️⃣ Open in browser
http://localhost:8501  

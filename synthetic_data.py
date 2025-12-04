import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---- Synthetic Data Generation ----

N = 2000
np.random.seed(42)

vehicle_models = ["Tata Prima", "Ashok Leyland 3718", "Eicher Pro 6035", "Mahindra Blazo X", "BharatBenz 2823R"]
route_types = ["Highway", "Mixed", "City"]
load_profiles = ["Light", "Medium", "Heavy"]
parts_list = ["Oil Filter", "Brake Pads", "Fuel Pump", "Air Filter", "Coolant", "None"]

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

start_date = datetime(2022,1,1)
end_date = datetime(2025,1,1)

data = {
    "truck_number_plate": [f"MH{random.randint(10,99)}AB{random.randint(1000,9999)}" for _ in range(N)],
    "vehicle_model": np.random.choice(vehicle_models, N),
    "year_bought": np.random.randint(2010, 2024, N),
    "service_date": [random_date(start_date, end_date) for _ in range(N)],
    "last_service_date": [random_date(start_date - timedelta(days=300), start_date) for _ in range(N)],
    "total_km_run": np.random.randint(50_000, 900_000, N),
    "km_after_last_service": np.random.randint(500, 40_000, N),
    "avg_daily_km_est": np.random.randint(50, 500, N),
    "engine_temperature_c": np.random.randint(70, 130, N),
    "vibrations_level": np.round(np.random.uniform(1, 9), 2),
    "oil_life_percent": np.random.randint(5, 100, N),
    "battery_health_percent": np.random.randint(20, 100, N),
    "route_type": np.random.choice(route_types, N),
    "load_profile": np.random.choice(load_profiles, N),
    "ambient_temp_c": np.random.randint(-5, 45, N),
    "brake_pad_thickness_mm": np.random.randint(2, 20, N),
    "tyre_health_percent": np.random.randint(10, 100, N),
    "fuel_efficiency_kmpl": np.round(np.random.uniform(2.5, 6.5, N), 2),
    "approx_past_services": np.random.randint(1, 20, N),
    "parts_changed_last_service": np.random.choice(parts_list, N),
    "days_until_next_service": np.random.randint(5, 120, N)
}

df = pd.DataFrame(data)
df.to_csv("truck_dataset.csv", index=False)

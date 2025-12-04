# train_test_split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("predictive_truck_maintenance_2000.csv")

# ==========================
# REMOVE UNUSED COLUMNS
# ==========================
df = df.drop(columns=[
    "record_id",
    "truck_id",
    "truck_number_plate",
    "service_date",
    "last_service_date",
    "technician_id"   # removed
], errors='ignore')

# ==========================
# TARGET VARIABLES
# ==========================
df["hours_until_next_service"] = df["days_until_next_service"] * 24

TARGET = "days_until_next_service"

# ==========================
# FEATURE LIST
# ==========================
FEATURES = [
    "vehicle_model",
    "year_bought",
    "total_km_run",
    "km_after_last_service",
    "avg_daily_km_est",
    "engine_temperature_c",
    "vibrations_level",
    "oil_life_percent",
    "battery_health_percent",
    "ambient_temp_c",
    "brake_pad_thickness_mm",
    "tyre_health_percent",
    "fuel_efficiency_kmpl",
    "route_type",
    "load_profile",
    "service_type",
    "parts_in_stock_status",
    "parts_changed_last_service",
    "technician_experience_years",
    "current_queue_length",
    "shift_hours_remaining",
    "approx_past_services"
]

df = df[FEATURES + [TARGET]]

# ==========================
# TRAIN / TEST SPLIT
# ==========================
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# PREPROCESSING
# ==========================
categorical_cols = [
    "vehicle_model",
    "route_type",
    "load_profile",
    "service_type",
    "parts_in_stock_status",
    "parts_changed_last_service"
]

numeric_cols = [col for col in FEATURES if col not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# ==========================
# MODEL
# ==========================
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ))
])

# ==========================
# TRAIN
# ==========================
model.fit(X_train, y_train)

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "truck_maintenance_regressor.pkl")

print("\nTraining completed successfully!")
print("Model saved as truck_maintenance_regressor.pkl")

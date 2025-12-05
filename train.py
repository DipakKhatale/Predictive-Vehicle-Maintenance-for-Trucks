import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# 1. Load CLEAN DATASET


df = pd.read_csv("truck_dataset.csv")


# 2. Select useful columns ONLY

useful_cols = [
    "truck_number_plate",          # ID (not used in model)
    "vehicle_model",
    "year_bought",
    "total_km_run",
    "km_after_last_service",
    "avg_daily_km_est",
    "engine_temperature_c",
    "vibrations_level",
    "oil_life_percent",
    "battery_health_percent",
    "route_type",
    "load_profile",
    "ambient_temp_c",
    "brake_pad_thickness_mm",
    "tyre_health_percent",
    "fuel_efficiency_kmpl",
    "approx_past_services",
    "parts_changed_last_service",
    "days_until_next_service"      # target
]

df = df[useful_cols]

# target
y = df["days_until_next_service"]

# features (drop target + truck_number_plate)
X = df.drop(columns=["days_until_next_service", "truck_number_plate"])


# 3. Identify Column Types

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# 4. Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# 5. Model

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# 6. Pipeline

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])


# 7. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipeline.fit(X_train, y_train)

# 8. Evaluate

y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 9. SAVE MODEL

joblib.dump(pipeline, "truck_maintenance_regressor.pkl")
print("Model saved ✔")

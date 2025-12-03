import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("predictive_truck_maintenance_2000.csv")

# 2. Features & target
target_col = "days_until_next_service"
y = df[target_col]

# Drop ID-type fields that don’t carry signal for prediction
X = df.drop(columns=[
    target_col,
    "record_id",
    "truck_id",
    "truck_number_plate",
    "service_date",
    "last_service_date",
    "parts_changed_last_service",  # could encode more
])

# 3. Column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# 4. Preprocess
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 6. Pipeline
regressor = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 7. Split: 1600 train, 400 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=1600,
    test_size=400,
    random_state=42
)

# 8. Train
regressor.fit(X_train, y_train)

# 9. Evaluate
y_pred = regressor.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 10. Save model for Streamlit
import joblib
joblib.dump(regressor, "truck_maintenance_regressor.pkl")
print("Model saved as truck_maintenance_regressor.pkl")

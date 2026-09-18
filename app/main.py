from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime, timezone
import csv



# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_calibrated_xgb.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.txt"

MODEL_VERSION = "xgb-calibrated-v1"

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.csv"

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

threshold = float(
    THRESHOLD_PATH.read_text().strip()
)

MODEL_VERSION = "xgb-calibrated-v1"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection using calibrated XGBoost",
    version="1.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
def log_prediction(
    timestamp,
    prediction,
    probability,
    model_version
):
    file_exists = LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "probability",
                "model_version"
            ])

        writer.writerow([
            timestamp,
            prediction,
            probability,
            model_version
        ])
    
@app.post("/predict")
def predict(transaction: Transaction):

    data = transaction.model_dump()

    # Feature engineering
    data["Amount_log"] = __import__("numpy").log1p(
        data["Amount"]
    )

    data["Time_hour"] = (
        data["Time"] // 3600
    ) % 24

    data["Time_day"] = (
        data["Time"] // (3600 * 24)
    )

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Fraud probability
    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    # Apply locked threshold
    prediction = int(
        probability >= threshold
    )
    
    timestamp = datetime.now(timezone.utc).isoformat()

    log_prediction(
        timestamp,
        prediction,
        probability,
        MODEL_VERSION
    )

    return {
        "fraud_prediction": prediction,
        "fraud_probability": round(probability, 6),
        "decision_threshold": threshold,
        "model_version": MODEL_VERSION,
        "timestamp": timestamp
    }
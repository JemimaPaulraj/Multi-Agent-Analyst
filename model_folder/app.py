# app.py
from pathlib import Path
from datetime import date, datetime, timedelta

import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

app = FastAPI()

# SageMaker mounts model artifacts to /opt/ml/model via ModelDataUrl
MODEL_PATH = Path("/opt/ml/model/model.joblib")

_model = None


class InvokePayload(BaseModel):
    horizon_days: int = Field(..., ge=1, le=365)
    start_date: str | None = None  # "YYYY-MM-DD"


def load_model():
    """Load model from SageMaker-mounted path."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Ensure ModelDataUrl is set in SageMaker Model configuration."
            )
        _model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Loaded from {MODEL_PATH}")
    return _model


@app.get("/ping")
def ping():
    # SageMaker health check endpoint
    try:
        load_model()
        return {"status": "ok"}
    except Exception as e:
        # return non-200 => endpoint considered unhealthy
        return Response(content=str(e), status_code=500)


@app.post("/invocations")
def invocations(payload: InvokePayload):
    # Parse start_date
    if payload.start_date:
        try:
            start = datetime.strptime(payload.start_date, "%Y-%m-%d").date()
        except ValueError:
            start = date.today()
    else:
        start = date.today()

    # Build future dataframe
    future_dates = [start + timedelta(days=i) for i in range(payload.horizon_days)]
    future_df = pd.DataFrame({"ds": pd.to_datetime(future_dates)})

    model = load_model()
    pred = model.predict(future_df)

    forecast = []
    for i, row in pred.iterrows():
        forecast.append(
            {
                "day": int(i) + 1,
                "date": row["ds"].strftime("%Y-%m-%d"),
                "forecast_ticket_count": int(round(row["yhat"])),
                "lower_bound": int(round(row["yhat_lower"])),
                "upper_bound": int(round(row["yhat_upper"])),
            }
        )

    return {"forecast": forecast}

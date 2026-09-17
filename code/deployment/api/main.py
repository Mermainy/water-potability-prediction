"""Serve the exact model packaged in the API image."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from schema import FEATURES


class WaterSample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ph: float = Field(ge=0, le=14)
    Hardness: float = Field(ge=0)
    Solids: float = Field(ge=0)
    Chloramines: float = Field(ge=0)
    Sulfate: float = Field(ge=0)
    Conductivity: float = Field(ge=0)
    Organic_carbon: float = Field(ge=0)
    Trihalomethanes: float = Field(ge=0)
    Turbidity: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "/app/models/model.joblib"))
    app.state.model = joblib.load(path)
    app.state.metadata = json.loads(path.with_name("metrics.json").read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Water Potability API", lifespan=lifespan)


@app.get("/health")
def health():
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "run_id": app.state.metadata["run_id"]}


@app.post("/predict")
def predict(sample: WaterSample):
    frame = pd.DataFrame([sample.model_dump()], columns=FEATURES)
    model = app.state.model
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
    return {"potability": prediction, "label": "Potable" if prediction else "Not potable",
            "probability_potable": probability, "run_id": app.state.metadata["run_id"]}

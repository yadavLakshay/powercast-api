from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi.responses import RedirectResponse
import numpy as np
import torch
from .inference import try_load_model, load_scaler, load_feature_schema, prepare_input, postprocess_preds

app = FastAPI(title="PowerCast Forecast API")

MODEL = None
LOAD_INFO: Dict | None = None
Y_SCALER = None
FEATURE_SCHEMA = None
DEVICE = None

@app.on_event("startup")  # deprecated; prefer lifespan
def startup():
    global MODEL, LOAD_INFO, Y_SCALER, FEATURE_SCHEMA, DEVICE
    MODEL, LOAD_INFO, DEVICE = try_load_model()
    try:
        Y_SCALER = load_scaler()
    except Exception:
        Y_SCALER = None
    FEATURE_SCHEMA = load_feature_schema()

class PredictRequest(BaseModel):
    inputs: List[List[List[float]]]
    return_scaled: Optional[bool] = False

class PredictResponse(BaseModel):
    preds: List
    model_info: Optional[Dict]

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    expected_f = len(FEATURE_SCHEMA["features"])
    if len(req.inputs[0][0]) != expected_f:
        raise HTTPException(status_code=400, detail=f"Expected {expected_f} features per timestep.")
    x = prepare_input(req.inputs, device=DEVICE)   # ensure same device as MODEL
    with torch.no_grad():
        preds = MODEL(x).detach().cpu().numpy()
    preds_out = preds if req.return_scaled else postprocess_preds(preds, y_scaler=Y_SCALER)
    return {"preds": preds_out.tolist(), "model_info": LOAD_INFO}
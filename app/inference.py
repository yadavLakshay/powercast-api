from app.models.model_def import GRUModel
import torch
import os
import joblib
import json
import numpy as np

def try_load_model():
    model_dir = os.path.join("app", "models")
    files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError("❌ No model checkpoint (.pt) found in app/models/")
    path = os.path.join(model_dir, files[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # ✅ Parameters aligned with training (Week4 final_results)
        model = GRUModel(
            input_dim=14,
            hidden_dim=128,
            horizon=6,
            target_dim=3,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        ).to(device)

        # load weights
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        return model, {"status": "✅ Model loaded successfully", "path": path}, device
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

# 🔹 Load the saved y_scaler
def load_scaler():
    path = os.path.join("app", "assets", "y_scaler_lb144_hr6.pkl")
    return joblib.load(path)

# 🔹 Load feature schema
def load_feature_schema():
    path = os.path.join("app", "assets", "feature_schema.json")
    with open(path, "r") as f:
        return json.load(f)

# 🔹 Prepare input (convert nested list to torch tensor)
def prepare_input(inputs: list, device=None):
    x = np.array(inputs, dtype=np.float32)
    tensor = torch.tensor(x, dtype=torch.float32)
    if device:
        tensor = tensor.to(device)
    return tensor

# 🔹 Postprocess predictions (inverse transform if scaler exists)
def postprocess_preds(preds: np.ndarray, y_scaler=None):
    if y_scaler is not None:
        preds_reshaped = preds.reshape(-1, preds.shape[-1])
        preds_inv = y_scaler.inverse_transform(preds_reshaped)
        return preds_inv.reshape(preds.shape)
    return preds
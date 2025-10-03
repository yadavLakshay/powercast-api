# ⚡ PowerCast API (Backend)

This is the **FastAPI backend** for the PowerCast project.  
It provides a REST API to serve deep learning models for **energy consumption forecasting**.

- **Backend API:** [PowerCast FastAPI Space](https://lakshay31-powercast-api.hf.space/docs)  
- **Frontend UI:** [PowerCast Gradio Space](https://lakshay31-powercast-ui.hf.space/)

## 🚀 How it works
1. The backend loads a pre-trained GRU-based time-series forecasting model.  
2. Clients send JSON payloads with weather/time-series data.  
3. The API responds with predicted energy consumption for three zones.

## Example Request
```json
{
  "inputs": [[[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.3, 0.4, 0.2, 0.1, 0.8, 0.9, 0.7, 0.5]]],
  "return_scaled": false
}
```

## Example Response
```json
{
  "preds": [[0.60, 0.26, 0.32]],
  "model_info": {
    "status": "✅ Model loaded successfully",
    "path": "app/models/GRUModel_best.pt"
  }
}
```
## 🗂 Folder Structure
```
powercast-api/
│
├── app/  
│   ├── main.py              # FastAPI entrypoint
│   ├── inference.py         # Model loading & preprocessing utils
│   ├── models/              # Saved GRU .pt model
│   ├── scalers/             # Saved scaler.pkl
│   ├── schemas/             # Feature schema JSON
│
├── requirements.txt
├── Dockerfile               # For Hugging Face Space (sdk: docker)
└── README.md
```

## 📦 Tech Stack
- Python 3.10  
- FastAPI  
- Uvicorn  
- PyTorch  
- NumPy  

## 🛠️ Run Locally
```bash
git clone https://github.com/Lakshay31/powercast-api.git
cd powercast-api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📦 Source Project

This deployment is based on my original submission to the SDS-CP036 PowerCast.

- [🔗 View the full project (EDA, modeling, experiments, deployment)](https://github.com/yadavLakshay/SDS-CP036-powercast/tree/main/advanced/submissions/team-members/lakshay-yadav)
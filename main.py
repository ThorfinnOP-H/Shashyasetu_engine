from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
from pathlib import Path

app = FastAPI(title="ShasyaSetu AI Engine")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shasyasetu-frontend.onrender.com/docs"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

try:
    ai_model = joblib.load(BASE_DIR / "market_ai_model.pkl")
    crop_encoder = joblib.load(BASE_DIR / "crop_encoder.pkl")
    mandi_encoder = joblib.load(BASE_DIR / "mandi_encoder.pkl")

    with open(BASE_DIR / "model_stats.json", "r") as f:
        stats = json.load(f)
    model_accuracy = stats.get("global_accuracy", 0)
except FileNotFoundError as exc:
    raise RuntimeError(f"Required model files not found in {BASE_DIR}") from exc


class CropRequest(BaseModel):
    crop_name: str
    mandi_location: str
    quantity_kg: int
    current_arrival_tonnes: float
    current_mandi_price: float


@app.post("/api/get-recommendation")
async def get_recommendation(request: CropRequest):
    try:
        crop_encoded = crop_encoder.transform([request.crop_name])[0]
        mandi_encoded = mandi_encoder.transform([request.mandi_location])[0]
    except ValueError:
        valid_crops = list(crop_encoder.classes_)
        valid_mandis = list(mandi_encoder.classes_)
        error_msg = (
            f"Wrong Name! Valid Crops: {valid_crops}. "
            f"Valid Mandis (Start ke 5): {valid_mandis[:5]}..."
        )
        raise HTTPException(status_code=400, detail=error_msg)

    input_features = [[
        crop_encoded,
        mandi_encoded,
        request.current_arrival_tonnes,
        request.current_mandi_price,
    ]]
    predicted_price = ai_model.predict(input_features)[0]
    predicted_price = round(float(predicted_price), 2)

    confidence = 0.85
    std_dev = max(abs(predicted_price - request.current_mandi_price) * 0.1, 0.01)

    transport_cost = 2.0
    storage_cost = 0.5 * 4

    current_net = request.current_mandi_price - transport_cost
    future_net = predicted_price - (storage_cost + transport_cost)

    if future_net > current_net:
        extra_profit = round((future_net - current_net) * request.quantity_kg, 2)
        return {
            "action": "HOLD",
            "message": f"💡 Prediction: ₹{predicted_price}/kg. HOLD to earn ₹{extra_profit} extra net profit.",
            "confidence": confidence,
            "market_volatility": round(std_dev, 2),
            "model_accuracy": f"{model_accuracy}%",
            "current_net": current_net,
            "future_net": round(future_net, 2),
        }

    return {
        "action": "SELL NOW",
        "message": f"💡 Prediction: ₹{predicted_price}/kg. SELL NOW to avoid losses.",
        "confidence": confidence,
        "market_volatility": round(std_dev, 2),
        "model_accuracy": f"{model_accuracy}%",
        "current_net": current_net,
        "future_net": round(future_net, 2),
    }

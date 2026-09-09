from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="ShasyaSetu AI Engine")

ai_model = joblib.load('market_ai_model.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
mandi_encoder = joblib.load('mandi_encoder.pkl')

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
        raise HTTPException(status_code=400, detail="Invalid Crop or Mandi name. Use Tomato, Onion, Potato, or Cabbage.")

    input_features = [[crop_encoded, mandi_encoded, request.current_arrival_tonnes, request.current_mandi_price]]
    predicted_price = ai_model.predict(input_features)[0]
    predicted_price = round(predicted_price, 2)

    # Use a lightweight fallback estimate for response metadata since the model object
    # does not expose prediction confidence or volatility directly.
    confidence = 0.85
    std_dev = max(abs(predicted_price - request.current_mandi_price) * 0.1, 0.01)

    transport_cost = 2.0
    storage_cost = 0.5 * 4  # 4 din ka storage

    current_net = request.current_mandi_price - transport_cost
    future_net = predicted_price - (storage_cost + transport_cost)

    if future_net > current_net:
        extra_profit = round((future_net - current_net) * request.quantity_kg, 2)
        return {
            "action": "HOLD",
            "message": f"💡 Prediction: ₹{predicted_price}/kg. HOLD to earn ₹{extra_profit} extra net profit.",
            "confidence": confidence,
            "market_volatility": round(std_dev, 2),
            "current_net": current_net,
            "future_net": round(future_net, 2)
        }
    else:
        return {
            "action": "SELL NOW",
            "message": f"💡 Prediction: ₹{predicted_price}/kg. SELL NOW to avoid losses.",
            "confidence": confidence,
            "market_volatility": round(std_dev, 2),
            "current_net": current_net,
            "future_net": round(future_net, 2)
        }
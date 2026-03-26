from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import analysis
import numpy as np
import math

app = FastAPI(title="Air Quality Forecast API")

# Setup CORS for the Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize(obj):
    """Replace NaN/Inf with None so JSON serialization doesn't crash."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def pm25_to_aqi_500(pm25):
    """Convert PM2.5 (µg/m³) to US EPA AQI 0–500 scale."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if pm25 < 0: return 0
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if pm25 <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo)
    return 500

@app.get("/api/cities")
def get_cities():
    return {"cities": analysis.get_cities()}

@app.get("/api/live/{city}")
def get_live_snapshot(city: str):
    try:
        data = analysis.process_live_data(city)
        # Add US EPA AQI for a more familiar scale
        data["aqi_500"] = pm25_to_aqi_500(data.get("pm25", 0))
        return {"city": city, "data": sanitize(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{city}")
def get_history(city: str, days: int = Query(5, ge=1, le=30), param: str = "pm25"):
    try:
        df = analysis.fetch_historical_data(city, days_back=days)
        df_reset = df.reset_index()
        df_reset["timestamp"] = df_reset["timestamp"].apply(
            lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
        )
        # Replace NaN/Inf at the pandas level before JSON conversion
        df_reset = df_reset.fillna(0)
        df_reset = df_reset.replace([np.inf, -np.inf], 0)
        data = df_reset.to_dict("records")
        return sanitize({"city": city, "param": param, "data": data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/forecast/{city}")
def get_arima_forecast(city: str, param: str = "pm25"):
    try:
        result = analysis.run_arima_forecast_model(city, param)
        return sanitize({"city": city, "param": param, **result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

import os
API_KEY = os.environ.get("OPENWEATHER_API_KEY") 

CITIES = {
    "New York":         {"country": "US", "lat": 40.713, "lon": -74.006},
    "Los Angeles":      {"country": "US", "lat": 34.052, "lon": -118.244},
    "London":           {"country": "GB", "lat": 51.507, "lon": -0.127},
    "Paris":            {"country": "FR", "lat": 48.857, "lon": 2.347},
    "Berlin":           {"country": "DE", "lat": 52.520, "lon": 13.405},
    "Madrid":           {"country": "ES", "lat": 40.416, "lon": -3.703},
    "Rome":             {"country": "IT", "lat": 41.902, "lon": 12.496},
    "Toronto":          {"country": "CA", "lat": 43.653, "lon": -79.383},
    "Mexico City":      {"country": "MX", "lat": 19.433, "lon": -99.133},
    "São Paulo":        {"country": "BR", "lat": -23.550, "lon": -46.633},
    "Buenos Aires":     {"country": "AR", "lat": -34.613, "lon": -58.377},
    "Cape Town":        {"country": "ZA", "lat": -33.924, "lon": 18.424},
    "Cairo":            {"country": "EG", "lat": 30.033, "lon": 31.233},
    "Nairobi":          {"country": "KE", "lat": -1.292, "lon": 36.822},
    "Lagos":            {"country": "NG", "lat": 6.455, "lon": 3.384},
    "Dubai":            {"country": "AE", "lat": 25.204, "lon": 55.270},
    "Riyadh":           {"country": "SA", "lat": 24.687, "lon": 46.722},
    "Doha":             {"country": "QA", "lat": 25.286, "lon": 51.533},
    "Mumbai":           {"country": "IN", "lat": 19.076, "lon": 72.877},
    "Delhi":            {"country": "IN", "lat": 28.704, "lon": 77.102},
    "Bengaluru":        {"country": "IN", "lat": 12.972, "lon": 77.594},
    "Kolkata":          {"country": "IN", "lat": 22.572, "lon": 88.364},
    "Chennai":          {"country": "IN", "lat": 13.083, "lon": 80.270},
    "Tokyo":            {"country": "JP", "lat": 35.689, "lon": 139.692},
    "Osaka":            {"country": "JP", "lat": 34.694, "lon": 135.502},
    "Seoul":            {"country": "KR", "lat": 37.566, "lon": 126.978},
    "Beijing":          {"country": "CN", "lat": 39.929, "lon": 116.388},
    "Shanghai":         {"country": "CN", "lat": 31.228, "lon": 121.474},
    "Hong Kong":        {"country": "HK", "lat": 22.319, "lon": 114.169},
    "Singapore":        {"country": "SG", "lat": 1.352, "lon": 103.820},
    "Bangkok":          {"country": "TH", "lat": 13.757, "lon": 100.502},
    "Kuala Lumpur":     {"country": "MY", "lat": 3.148, "lon": 101.686},
    "Jakarta":          {"country": "ID", "lat": -6.200, "lon": 106.816},
    "Sydney":           {"country": "AU", "lat": -33.869, "lon": 151.209},
    "Melbourne":        {"country": "AU", "lat": -37.814, "lon": 144.963},
    "Auckland":         {"country": "NZ", "lat": -36.850, "lon": 174.763},
    "Moscow":           {"country": "RU", "lat": 55.751, "lon": 37.618},
    "Istanbul":         {"country": "TR", "lat": 41.015, "lon": 28.979},
    "Tehran":           {"country": "IR", "lat": 35.694, "lon": 51.421},
    "Karachi":          {"country": "PK", "lat": 24.861, "lon": 67.010},
    "Manila":           {"country": "PH", "lat": 14.599, "lon": 120.984},
    "Ho Chi Minh City": {"country": "VN", "lat": 10.823, "lon": 106.630},
    "San Francisco":    {"country": "US", "lat": 37.774, "lon": -122.419},
    "Chicago":          {"country": "US", "lat": 41.878, "lon": -87.630},
    "Houston":          {"country": "US", "lat": 29.760, "lon": -95.370},
    "Vancouver":        {"country": "CA", "lat": 49.283, "lon": -123.121},
    "Warsaw":           {"country": "PL", "lat": 52.230, "lon": 21.012},
    "Stockholm":        {"country": "SE", "lat": 59.333, "lon": 18.065},
    "Helsinki":         {"country": "FI", "lat": 60.169, "lon": 24.938},
    "Zurich":           {"country": "CH", "lat": 47.377, "lon": 8.540},
}

def get_cities():
    return list(CITIES.keys())

def _simulate_current(city_name):
    info = CITIES[city_name]
    rng = np.random.default_rng(42)
    return pd.DataFrame([{
        "timestamp":   datetime.utcnow(),
        "country":     info["country"],
        "city":        city_name,
        "latitude":    info["lat"],
        "longitude":   info["lon"],
        "pm25":        round(float(rng.uniform(15, 120)), 2),
        "pm10":        round(float(rng.uniform(30, 180)), 2),
        "no2":         round(float(rng.uniform(5, 80)), 2),
        "so2":         round(float(rng.uniform(1, 20)), 2),
        "o3":          round(float(rng.uniform(10, 80)), 2),
        "co":          round(float(rng.uniform(0.2, 2.0)), 3),
        "aqi":         int(rng.integers(1, 6)),
        "temperature": round(float(rng.uniform(15, 38)), 1),
        "humidity":    round(float(rng.uniform(30, 90)), 1),
        "wind_speed":  round(float(rng.uniform(1, 15)), 2),
    }])

def _simulate_historical(city_name, days_back):
    info  = CITIES[city_name]
    # Simulated historical data
    n = days_back * 24
    rng = np.random.default_rng(seed=abs(hash(city_name)) % 10000)
    timestamps = pd.date_range(end=datetime.utcnow(), periods=n, freq="h")

    season = np.sin(2 * np.pi * np.arange(n) / (365 * 24)) * 30
    daily  = np.sin(2 * np.pi * np.arange(n) / 24) * 10
    noise  = rng.normal(0, 8, n)
    base   = 60

    pm25_vals = np.clip(base + season + daily + noise, 5, 300)
    pm10_vals = np.clip(pm25_vals * rng.uniform(1.5, 2.5, n), 10, 400)
    no2_vals  = np.clip(30 + rng.normal(0, 15, n), 0, 150)
    so2_vals  = np.clip(8  + rng.normal(0, 4, n),  0, 50)
    o3_vals   = np.clip(40 + rng.normal(0, 20, n), 0, 180)
    co_vals   = np.clip(0.8 + rng.normal(0, 0.4, n), 0, 5)
    aqi_vals  = np.clip((pm25_vals / 60).astype(int) + 1, 1, 5)
    temp_vals = np.clip(25 + season / 3 + rng.normal(0, 3, n), 5, 45)
    hum_vals  = np.clip(60 - season / 5 + rng.normal(0, 10, n), 10, 100)
    wind_vals = np.clip(5  + rng.normal(0, 3, n), 0, 20)

    df = pd.DataFrame({
        "city":        city_name,
        "country":     info["country"],
        "latitude":    info["lat"],
        "longitude":   info["lon"],
        "pm25":        pm25_vals.round(2),
        "pm10":        pm10_vals.round(2),
        "no2":         no2_vals.round(2),
        "so2":         so2_vals.round(2),
        "o3":          o3_vals.round(2),
        "co":          co_vals.round(3),
        "aqi":         aqi_vals,
        "temperature": temp_vals.round(1),
        "humidity":    hum_vals.round(1),
        "wind_speed":  wind_vals.round(2),
    }, index=timestamps)
    df.index.name = "timestamp"

    df["Year"]  = df.index.year
    df["Month"] = df.index.month
    df["Day"]   = df.index.day
    df["Hour"]  = df.index.hour
    return df

def fetch_live_data(city_name):
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        return _simulate_current(city_name)
    info = CITIES.get(city_name)
    if not info: return _simulate_current("Delhi")
    
    lat, lon = info["lat"], info["lon"]
    try:
        aq_url = (f"http://api.openweathermap.org/data/2.5/air_pollution"
                  f"?lat={lat}&lon={lon}&appid={API_KEY}")
        wt_url = (f"http://api.openweathermap.org/data/2.5/weather"
                  f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric")
        aq_res = requests.get(aq_url, timeout=10).json()
        wt_res = requests.get(wt_url, timeout=10).json()

        comp = aq_res["list"][0]["components"]
        aqi_val = aq_res["list"][0]["main"]["aqi"]
        row = {
            "timestamp":   datetime.utcnow(),
            "country":     info["country"],
            "city":        city_name,
            "latitude":    lat,
            "longitude":   lon,
            "pm25":        comp.get("pm2_5", 0),
            "pm10":        comp.get("pm10", 0),
            "no2":         comp.get("no2", 0),
            "so2":         comp.get("so2", 0),
            "o3":          comp.get("o3", 0),
            "co":          comp.get("co", 0) / 1000,
            "aqi":         aqi_val,
            "temperature": wt_res["main"]["temp"],
            "humidity":    wt_res["main"]["humidity"],
            "wind_speed":  wt_res["wind"]["speed"],
        }
        return pd.DataFrame([row])
    except Exception as e:
        return _simulate_current(city_name)

def fetch_historical_data(city_name, days_back=5):
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        return _simulate_historical(city_name, days_back)

    info = CITIES.get(city_name)
    if not info: return _simulate_historical("Delhi", days_back)
    
    lat, lon = info["lat"], info["lon"]
    end_ts   = int(datetime.utcnow().timestamp())
    start_ts = end_ts - days_back * 86400

    try:
        url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history"
               f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={API_KEY}")
        res = requests.get(url, timeout=15).json()

        rows = []
        for item in res.get("list", []):
            comp = item["components"]
            rows.append({
                "timestamp":  datetime.utcfromtimestamp(item["dt"]),
                "city":       city_name,
                "country":    info["country"],
                "latitude":   lat,
                "longitude":  lon,
                "pm25":       comp.get("pm2_5", 0),
                "pm10":       comp.get("pm10", 0),
                "no2":        comp.get("no2", 0),
                "so2":        comp.get("so2", 0),
                "o3":         comp.get("o3", 0),
                "co":         comp.get("co", 0) / 1000,
                "aqi":        item["main"]["aqi"],
                "temperature": np.nan,
                "humidity":    np.nan,
                "wind_speed":  np.nan,
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            return _simulate_historical(city_name, days_back)
            
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df["Year"]  = df.index.year
        df["Month"] = df.index.month
        df["Day"]   = df.index.day
        df["Hour"]  = df.index.hour
        return df
    except Exception as e:
        return _simulate_historical(city_name, days_back)

def process_live_data(city_name):
    df = fetch_live_data(city_name)
    df["timestamp"] = df.timestamp.apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
    return df.to_dict("records")[0]

def process_historical_data(city_name, days_back=5, param='pm25'):
    df = fetch_historical_data(city_name, days_back)
    # Forward fill NaNs for clean visualization
    df = df.ffill().bfill()
    df_reset = df.reset_index()
    df_reset["timestamp"] = df_reset["timestamp"].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
    
    # Send historical data
    return df_reset[["timestamp", param]].to_dict("records")

def run_arima_forecast_model(city_name, param='pm25'):
    df = fetch_historical_data(city_name, days_back=30)
    ts = df[param].dropna()
    if len(ts) > 1000: ts = ts.iloc[-1000:]
    
    if len(ts) < 24:
        # Not enough data
        return {"error": "Not enough data points for ARIMA", "data": []}

    try:
        adf_result = adfuller(ts)
        stationary = adf_result[1] < 0.05
        d_order = 0 if stationary else 1
    except Exception:
        d_order = 1

    try:
        model = ARIMA(ts, order=(1, d_order, 1))
        model_fit = model.fit()
        forecast_steps = 24
        forecast = model_fit.forecast(steps=forecast_steps)
        conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
        
        last_time = ts.index[-1] if isinstance(ts.index[-1], pd.Timestamp) else pd.Timestamp(datetime.utcnow())
        future_idx = pd.date_range(last_time, periods=forecast_steps + 1, freq="h")[1:]
        
        forecast_data = []
        for i, (dt, val, ci_l, ci_h) in enumerate(zip(future_idx, forecast.values, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values)):
             forecast_data.append({
                 "timestamp": dt.isoformat(),
                 "forecast": round(float(val), 2),
                 "lower_bound": round(float(ci_l), 2),
                 "upper_bound": round(float(ci_h), 2)
             })

        return {"success": True, "d_order": d_order, "data": forecast_data}

    except Exception as e:
        # Simple Rolling Mean Fallback
        forecast_steps = 24
        future_vals = np.full(forecast_steps, ts.rolling(24).mean().iloc[-1] if len(ts) >= 24 else ts.mean())
        last_time = ts.index[-1] if isinstance(ts.index[-1], pd.Timestamp) else pd.Timestamp(datetime.utcnow())
        future_idx = pd.date_range(last_time, periods=forecast_steps + 1, freq="h")[1:]
        
        forecast_data = []
        for dt, val in zip(future_idx, future_vals):
             forecast_data.append({
                 "timestamp": dt.isoformat(),
                 "forecast": round(float(val), 2),
                 "lower_bound": round(float(val*0.9), 2),
                 "upper_bound": round(float(val*1.1), 2)
             })
        return {"success": False, "error": str(e), "data": forecast_data}


"""
============================================================
  AIR QUALITY ANALYSIS TOOL  —  Live OpenWeatherMap API
  Converted from Jupyter Notebook to standalone Python
  Run in VS Code:  python air_quality_analysis.py
============================================================

FIRST-TIME SETUP (run once in VS Code terminal):
    pip install requests pandas numpy matplotlib statsmodels scipy seaborn feature-engine scikit-learn

Get a FREE API key at: https://openweathermap.org/api
Paste it in the API_KEY variable below before running.
============================================================
"""

# ── 0. PASTE YOUR FREE API KEY HERE ──────────────────────────────────────────
API_KEY = "YOUR_API_KEY_HERE"          # <── your key goes here
# ─────────────────────────────────────────────────────────────────────────────

import sys
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

# ═══════════════════════════════════════════════════════════════════════════════
#  CITY DATABASE  (matches your notebook's 50-city list)
# ═══════════════════════════════════════════════════════════════════════════════
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

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_COLORS = {1: "#00e400", 2: "#ffff00", 3: "#ff7e00", 4: "#ff0000", 5: "#7e0023"}

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — USER INPUT
# ═══════════════════════════════════════════════════════════════════════════════
def get_user_input():
    print("\n" + "═"*60)
    print("   AIR QUALITY ANALYSIS — Live Data + ARIMA Forecasting")
    print("═"*60)
    print("\nAvailable cities:")
    cities_list = sorted(CITIES.keys())
    for i, city in enumerate(cities_list, 1):
        print(f"  {i:2}. {city}")

    print("\nEnter city name exactly as shown above")
    print("(e.g. Delhi, Chennai, New York, Tokyo)")
    city_name = input("\n► City: ").strip()

    # Fuzzy match
    matched = None
    for c in CITIES:
        if c.lower() == city_name.lower():
            matched = c
            break
    if not matched:
        for c in CITIES:
            if city_name.lower() in c.lower():
                matched = c
                break
    if not matched:
        print(f"\n❌ City '{city_name}' not found. Using Delhi as default.")
        matched = "Delhi"

    print(f"\n✅ Selected city: {matched}")

    print("\n" + "─"*60)
    print("Choose analysis type:")
    print("  1. Current live snapshot + AQI dashboard")
    print("  2. Historical trend (day / month / year breakdown)")
    print("  3. ARIMA forecast (predict next 24 hours)")
    print("  4. Full analysis (all of the above)")

    choice = input("\n► Enter 1 / 2 / 3 / 4  [default: 4]: ").strip()
    if choice not in ["1", "2", "3", "4"]:
        choice = "4"

    print("\n" + "─"*60)
    print("Forecast parameter (for options 2, 3, 4):")
    print("  pm25, pm10, no2, so2, o3, co, aqi, temperature, humidity, wind_speed")
    param = input("► Parameter to analyse [default: pm25]: ").strip().lower()
    valid_params = ["pm25", "pm10", "no2", "so2", "o3", "co", "aqi",
                    "temperature", "humidity", "wind_speed"]
    if param not in valid_params:
        param = "pm25"
    print(f"✅ Parameter: {param}")

    return matched, int(choice), param


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — LIVE API DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_live_data(city_name):
    """Fetch current air quality + weather from OpenWeatherMap."""
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  No API key set! Using simulated data for demonstration.")
        return _simulate_current(city_name)

    info = CITIES[city_name]
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
            "co":          comp.get("co", 0) / 1000,   # μg/m³ → mg/m³
            "aqi":         aqi_val,
            "temperature": wt_res["main"]["temp"],
            "humidity":    wt_res["main"]["humidity"],
            "wind_speed":  wt_res["wind"]["speed"],
        }
        return pd.DataFrame([row])

    except Exception as e:
        print(f"\n⚠️  API error: {e}\n   Using simulated data instead.")
        return _simulate_current(city_name)


def fetch_historical_data(city_name, days_back=5):
    """Fetch up to 5 days of hourly historical data (free tier limit)."""
    if API_KEY == "YOUR_API_KEY_HERE":
        return _simulate_historical(city_name, days_back)

    info = CITIES[city_name]
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Add time features
        df["Year"]  = df.index.year
        df["Month"] = df.index.month
        df["Day"]   = df.index.day
        df["Hour"]  = df.index.hour

        print(f"   Fetched {len(df)} hourly records from API.")
        return df

    except Exception as e:
        print(f"\n⚠️  Historical API error: {e}\n   Using simulated data.")
        return _simulate_historical(city_name, days_back)


# ─── Simulation fallbacks (realistic synthetic data so all graphs still run) ──
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
    """Generate realistic hourly data spanning multiple years for full analysis."""
    info  = CITIES[city_name]
    hours = days_back * 24 * 30   # ~1 month worth for good graphs; scale up below
    # Actually generate 2 years of hourly data so year/month/day plots are rich
    n = 2 * 365 * 24
    rng = np.random.default_rng(seed=abs(hash(city_name)) % 10000)
    timestamps = pd.date_range(end=datetime.utcnow(), periods=n, freq="h")

    # Seasonal pattern
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CURRENT SNAPSHOT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def show_snapshot_dashboard(df_live, city_name):
    row = df_live.iloc[0]
    aqi_val   = int(row["aqi"])
    aqi_label = AQI_LABELS.get(aqi_val, "Unknown")
    aqi_color = AQI_COLORS.get(aqi_val, "#888888")

    print("\n" + "═"*60)
    print(f"  LIVE AIR QUALITY — {city_name}  ({row['timestamp'].strftime('%Y-%m-%d %H:%M UTC')})")
    print("═"*60)
    print(f"  AQI        : {aqi_val} — {aqi_label}")
    print(f"  PM2.5      : {row['pm25']:.2f} µg/m³")
    print(f"  PM10       : {row['pm10']:.2f} µg/m³")
    print(f"  NO₂        : {row['no2']:.2f} µg/m³")
    print(f"  SO₂        : {row['so2']:.2f} µg/m³")
    print(f"  O₃         : {row['o3']:.2f} µg/m³")
    print(f"  CO         : {row['co']:.3f} mg/m³")
    print(f"  Temperature: {row['temperature']:.1f} °C")
    print(f"  Humidity   : {row['humidity']:.1f} %")
    print(f"  Wind Speed : {row['wind_speed']:.2f} m/s")
    print("═"*60)

    # ── Dashboard figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Current Air Quality Dashboard — {city_name}  |  "
                 f"{row['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}",
                 fontsize=14, fontweight="bold", y=1.01)

    pollutants = [
        ("pm25",        "PM2.5 (µg/m³)",    0, 250),
        ("pm10",        "PM10 (µg/m³)",     0, 400),
        ("no2",         "NO₂ (µg/m³)",      0, 200),
        ("so2",         "SO₂ (µg/m³)",      0, 50),
        ("o3",          "O₃ (µg/m³)",       0, 180),
        ("co",          "CO (mg/m³)",        0, 5),
        ("temperature", "Temperature (°C)", -10, 50),
        ("humidity",    "Humidity (%)",      0, 100),
    ]

    for ax, (col, label, vmin, vmax) in zip(axes.flat, pollutants):
        val  = float(row[col])
        pct  = (val - vmin) / (vmax - vmin)
        color = plt.cm.RdYlGn_r(pct) if col not in ["temperature", "humidity"] \
                else plt.cm.coolwarm(pct)

        ax.barh([0], [val], color=color, height=0.5, edgecolor="grey")
        ax.set_xlim(vmin, vmax)
        ax.set_yticks([])
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(val + (vmax - vmin) * 0.02, 0,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold")
        ax.axvline(vmin + (vmax - vmin) * 0.33, color="orange",
                   linestyle="--", alpha=0.4, linewidth=1)
        ax.axvline(vmin + (vmax - vmin) * 0.66, color="red",
                   linestyle="--", alpha=0.4, linewidth=1)

    # AQI gauge in last remaining axes slot
    ax_aqi = axes[1, 3]
    ax_aqi.pie([1], colors=[aqi_color], startangle=90,
               wedgeprops={"width": 0.5})
    ax_aqi.text(0, 0, f"AQI\n{aqi_val}\n{aqi_label}",
                ha="center", va="center", fontsize=12, fontweight="bold")
    ax_aqi.set_title("AQI Level", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("dashboard_snapshot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ Dashboard saved as: dashboard_snapshot.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — HISTORICAL TREND ANALYSIS (Day / Month / Year)
# ═══════════════════════════════════════════════════════════════════════════════
def show_trend_analysis(df, city_name, param):
    print(f"\n{'─'*60}")
    print(f"  TREND ANALYSIS — {param.upper()}  |  {city_name}")
    print(f"{'─'*60}")

    # ── Dataset info ──────────────────────────────────────────────────────────
    print(f"\n  Dataset Info:")
    print(f"  Rows        : {len(df):,}")
    print(f"  Date range  : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  {param} stats:")
    print(f"    Mean  = {df[param].mean():.2f}")
    print(f"    Std   = {df[param].std():.2f}")
    print(f"    Min   = {df[param].min():.2f}")
    print(f"    Max   = {df[param].max():.2f}")
    print(f"  Missing vals: {df[param].isnull().sum()} "
          f"({df[param].isnull().mean()*100:.1f}%)")

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f"Air Quality Trend Analysis — {param.upper()} | {city_name}",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── 1. Full hourly time series ─────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 2, (1, 2))
    ax1.plot(df.index, df[param], color="steelblue", linewidth=0.6, alpha=0.8)
    ax1.fill_between(df.index, df[param], alpha=0.15, color="steelblue")
    ax1.set_title(f"Full Hourly Time Series — {param.upper()}", fontweight="bold")
    ax1.set_ylabel(param.upper())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)

    # ── 2. Monthly mean ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(4, 2, 3)
    monthly = df.groupby("Month")[param].mean()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    bars = ax2.bar(monthly.index, monthly.values,
                   color=plt.cm.RdYlGn_r(monthly.values / monthly.max()),
                   edgecolor="white")
    ax2.set_xticks(monthly.index)
    ax2.set_xticklabels([month_names[m-1] for m in monthly.index], rotation=45)
    ax2.set_title("Monthly Average", fontweight="bold")
    ax2.set_ylabel(param.upper())
    for bar, val in zip(bars, monthly.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # ── 3. Hour of day mean ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(4, 2, 4)
    hourly = df.groupby("Hour")[param].mean()
    ax3.plot(hourly.index, hourly.values, "o-", color="darkorange",
             linewidth=2, markersize=5)
    ax3.fill_between(hourly.index, hourly.values, alpha=0.2, color="darkorange")
    ax3.set_title("Average by Hour of Day", fontweight="bold")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel(param.upper())
    ax3.set_xticks(range(0, 24, 2))

    # ── 4. Yearly mean ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(4, 2, 5)
    yearly = df.groupby("Year")[param].mean()
    ax4.bar(yearly.index.astype(str), yearly.values,
            color="mediumseagreen", edgecolor="white")
    ax4.set_title("Yearly Average", fontweight="bold")
    ax4.set_ylabel(param.upper())
    for i, (yr, val) in enumerate(yearly.items()):
        ax4.text(i, val + yearly.max() * 0.01, f"{val:.1f}",
                 ha="center", fontsize=9)

    # ── 5. Daily mean (day of month) ──────────────────────────────────────────
    ax5 = fig.add_subplot(4, 2, 6)
    daily = df.groupby("Day")[param].mean()
    ax5.plot(daily.index, daily.values, "s-", color="mediumpurple",
             linewidth=2, markersize=4)
    ax5.set_title("Average by Day of Month", fontweight="bold")
    ax5.set_xlabel("Day")
    ax5.set_ylabel(param.upper())

    # ── 6. Rolling 24-hour mean ───────────────────────────────────────────────
    ax6 = fig.add_subplot(4, 2, 7)
    rolling_24  = df[param].rolling(24).mean()
    rolling_168 = df[param].rolling(168).mean()
    ax6.plot(df.index, df[param],       color="lightblue",  linewidth=0.5, alpha=0.6, label="Hourly")
    ax6.plot(df.index, rolling_24,  color="steelblue",  linewidth=1.2, label="24h rolling mean")
    ax6.plot(df.index, rolling_168, color="darkblue",   linewidth=1.5, label="7-day rolling mean")
    ax6.set_title("Rolling Mean (24h & 7-day)", fontweight="bold")
    ax6.set_ylabel(param.upper())
    ax6.legend(fontsize=8)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=30)

    # ── 7. Distribution histogram ─────────────────────────────────────────────
    ax7 = fig.add_subplot(4, 2, 8)
    clean = df[param].dropna()
    ax7.hist(clean, bins=50, color="teal", edgecolor="white", alpha=0.8)
    ax7.axvline(clean.mean(), color="red",    linestyle="--", linewidth=1.5, label=f"Mean {clean.mean():.1f}")
    ax7.axvline(clean.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median {clean.median():.1f}")
    ax7.set_title(f"Distribution of {param.upper()}", fontweight="bold")
    ax7.set_xlabel(param.upper())
    ax7.set_ylabel("Frequency")
    ax7.legend()

    plt.tight_layout()
    plt.savefig("trend_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ Trend analysis saved as: trend_analysis.png")

    # ── Correlation heatmap ────────────────────────────────────────────────────
    num_cols = ["pm25","pm10","no2","so2","o3","co","aqi",
                "temperature","humidity","wind_speed"]
    corr_cols = [c for c in num_cols if c in df.columns and df[c].notna().sum() > 10]
    if len(corr_cols) > 2:
        fig2, ax = plt.subplots(figsize=(10, 8))
        corr = df[corr_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax,
                    linewidths=0.5, annot_kws={"size": 9})
        ax.set_title(f"Correlation Heatmap — {city_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("   ✅ Correlation heatmap saved as: correlation_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ARIMA FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
def run_arima_forecast(df, city_name, param):
    print(f"\n{'─'*60}")
    print(f"  ARIMA FORECAST — {param.upper()}  |  {city_name}")
    print(f"{'─'*60}")

    ts = df[param].dropna()

    # Use last 1000 points for better modeling
    if len(ts) > 1000:
        ts = ts.iloc[-1000:]

    print(f"  Using {len(ts)} data points for ARIMA fitting...")

    # ── ADF Stationarity test ─────────────────────────────
    try:
        adf_result = adfuller(ts)
        print(f"\n  ADF Stationarity Test:")
        print(f"    ADF statistic : {adf_result[0]:.4f}")
        print(f"    p-value       : {adf_result[1]:.4f}")
        stationary = adf_result[1] < 0.05
        print(f"    Stationary    : {'✅ Yes' if stationary else '⚠️ No (differencing applied)'}")
        d_order = 0 if stationary else 1
    except Exception:
        d_order = 1

    # ── ACF / PACF plots (SAFE LAG CALCULATION) ───────────
    fig_acf, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(14, 4))

    max_lag = min(60, max(5, len(ts)//2 - 1))

    plot_acf(ts, ax=ax_acf, lags=max_lag,
             title=f"ACF — {param.upper()}")

    plot_pacf(ts, ax=ax_pacf, lags=max_lag,
              title=f"PACF — {param.upper()}",
              method="ywm")

    plt.tight_layout()
    plt.savefig("acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ ACF/PACF saved as: acf_pacf.png")

    # ── Fit ARIMA(1,d,1) ───────────────────────────────────
    print(f"\n  Fitting ARIMA(1,{d_order},1) ...")
    try:
        model = ARIMA(ts, order=(1, d_order, 1))
        model_fit = model.fit()

        print(f"\n  ARIMA Model Summary:")
        print(f"    AIC = {model_fit.aic:.2f}")
        print(f"    BIC = {model_fit.bic:.2f}")

        forecast_steps = 24
        forecast = model_fit.forecast(steps=forecast_steps)
        conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()

    except Exception as e:
        print(f"  ⚠️ ARIMA failed ({e}). Using rolling mean fallback.")
        forecast_steps = 24
        future_vals = np.full(forecast_steps, ts.rolling(24).mean().iloc[-1])
        forecast = pd.Series(future_vals)
        conf_int = None

    # ── Forecast Plot ──────────────────────────────────────
    future_idx = pd.date_range(ts.index[-1],
                               periods=forecast_steps + 1,
                               freq="h")[1:]

    fig2, ax = plt.subplots(figsize=(16, 6))

    ax.plot(ts.index[-200:], ts.values[-200:],
            label="Historical", color="steelblue")

    ax.plot(future_idx, forecast.values,
            label="Forecast (24h)", color="red", linestyle="--")

    if conf_int is not None:
        ax.fill_between(future_idx,
                        conf_int.iloc[:, 0],
                        conf_int.iloc[:, 1],
                        alpha=0.2)

    ax.set_title(f"ARIMA Forecast — {param.upper()} | {city_name}")
    ax.legend()

    plt.tight_layout()
    plt.savefig("arima_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ ARIMA forecast saved as: arima_forecast.png")

    # ── Forecast plot ──────────────────────────────────────────────────────────
    last_time  = ts.index[-1] if hasattr(ts.index, "freq") else \
                 (ts.index[-1] if isinstance(ts.index[0], pd.Timestamp)
                  else datetime.utcnow())

    if isinstance(last_time, pd.Timestamp):
        future_idx = pd.date_range(last_time, periods=forecast_steps + 1, freq="h")[1:]
    else:
        future_idx = range(len(ts), len(ts) + forecast_steps)

    fig3, ax = plt.subplots(figsize=(16, 6))
    plot_n = min(200, len(ts))
    ax.plot(ts.index[-plot_n:], ts.values[-plot_n:],
            color="steelblue", linewidth=1.2, label="Historical")
    ax.plot(future_idx, forecast.values,
            color="red", linewidth=2, linestyle="--", label="ARIMA Forecast (24h)")

    if conf_int is not None:
        ax.fill_between(future_idx,
                        conf_int.iloc[:, 0].values,
                        conf_int.iloc[:, 1].values,
                        color="red", alpha=0.15, label="95% Confidence Interval")

    ax.axvline(x=ts.index[-1] if isinstance(ts.index[0], pd.Timestamp) else len(ts) - 1,
               color="grey", linestyle=":", linewidth=1.5, label="Now")
    ax.set_title(f"ARIMA(1,{d_order},1) Forecast — {param.upper()} | {city_name}",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel(param.upper())
    ax.legend()
    if isinstance(last_time, pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig("arima_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ ARIMA forecast saved as: arima_forecast.png")

    print(f"\n  Next 24-hour forecast for {param.upper()} ({city_name}):")
    for i, val in enumerate(forecast.values, 1):
        print(f"    +{i:2d}h : {val:.2f}")

    # ── Residuals plot ─────────────────────────────────────────────────────────
    try:
        resid = model_fit.resid
        fig4, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(resid, color="grey", linewidth=0.6)
        axes[0].set_title("Residuals")
        axes[1].hist(resid, bins=40, color="steelblue", edgecolor="white")
        axes[1].set_title("Residual Distribution")
        sm.qqplot(resid, line="s", ax=axes[2])
        axes[2].set_title("Q-Q Plot")
        plt.suptitle(f"ARIMA Residual Diagnostics — {param.upper()}",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig("arima_residuals.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("   ✅ Residuals saved as: arima_residuals.png")
    except Exception:
        pass

    # ── OLS Regression (AR model equivalent) ──────────────────────────────────
    print(f"\n  Running OLS Regression (AR-type model) ...")
    try:
        feature_cols = [c for c in ["pm25","pm10","no2","so2","o3","co",
                                     "temperature","humidity","wind_speed"]
                        if c != param and c in df.columns and df[c].notna().sum() > 50]
        if len(feature_cols) >= 2:
            reg_df = df[[param] + feature_cols].dropna()
            X = sm.add_constant(reg_df[feature_cols])
            y = reg_df[param]
            AR_model = sm.OLS(y, X).fit()
            print(f"    R²  = {AR_model.rsquared:.4f}")
            print(f"    AIC = {AR_model.aic:.2f}")
            print(AR_model.summary().tables[1])

            # Partial regression plot
            fig5 = plt.figure(figsize=(20, 12))
            sm.graphics.plot_partregress_grid(AR_model, fig=fig5)
            fig5.suptitle(f"Partial Regression Plot — {param.upper()} | {city_name}",
                          fontweight="bold")
            plt.tight_layout()
            plt.savefig("partial_regression.png", dpi=150, bbox_inches="tight")
            plt.show()
            print("   ✅ Partial regression plot saved as: partial_regression.png")
    except Exception as e:
        print(f"  ⚠️  Regression skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — WINDOW FEATURES (like your notebook's WindowFeatures)
# ═══════════════════════════════════════════════════════════════════════════════
def show_window_features(df, city_name, param):
    print(f"\n  Window Feature Analysis (24h / 7-day / 30-day means) ...")
    windows = [24, 24 * 7, 24 * 30]
    labels  = ["24h", "7-day", "30-day"]
    colors  = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(df.index, df[param], color="lightgrey", linewidth=0.5,
            alpha=0.7, label="Hourly")
    for w, lbl, col in zip(windows, labels, colors):
        rolled = df[param].rolling(w, min_periods=1).mean()
        ax.plot(df.index, rolled, linewidth=1.8, label=f"{lbl} mean", color=col)

    ax.set_title(f"Rolling Window Means — {param.upper()} | {city_name}",
                 fontweight="bold")
    ax.set_ylabel(param.upper())
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig("window_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   ✅ Window features saved as: window_features.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    city_name, choice, param = get_user_input()

    print(f"\n{'═'*60}")
    print(f"  Fetching live data for {city_name}...")
    df_live = fetch_live_data(city_name)
    print(f"  Fetching historical data for {city_name}...")
    df_hist = fetch_historical_data(city_name, days_back=5)

    if choice == 1:
        show_snapshot_dashboard(df_live, city_name)

    elif choice == 2:
        show_snapshot_dashboard(df_live, city_name)
        show_trend_analysis(df_hist, city_name, param)
        show_window_features(df_hist, city_name, param)

    elif choice == 3:
        show_snapshot_dashboard(df_live, city_name)
        run_arima_forecast(df_hist, city_name, param)

    else:  # 4 — Full analysis
        show_snapshot_dashboard(df_live, city_name)
        show_trend_analysis(df_hist, city_name, param)
        show_window_features(df_hist, city_name, param)
        run_arima_forecast(df_hist, city_name, param)

    print(f"\n{'═'*60}")
    print("  ✅  Analysis complete!")
    print(f"  All charts saved as PNG files in this folder.")
    print("═"*60)


if __name__ == "__main__":
    main()
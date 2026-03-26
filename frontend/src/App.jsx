import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  ComposedChart, Area, Legend
} from 'recharts';
import { 
  Wind, ThermometerSun, Droplets, CloudFog, Cloud, 
  Activity, AlertCircle, RefreshCw, ChevronDown, CheckCircle2
} from 'lucide-react';

const API_BASE = "/api";

const PARAMS = [
  { value: 'pm25', label: 'PM2.5' },
  { value: 'pm10', label: 'PM10' },
  { value: 'no2', label: 'Nitrogen Dioxide (NO2)' },
  { value: 'so2', label: 'Sulfur Dioxide (SO2)' },
  { value: 'o3', label: 'Ozone (O3)' },
  { value: 'co', label: 'Carbon Monoxide (CO)' },
  { value: 'temperature', label: 'Temperature' },
  { value: 'humidity', label: 'Humidity' },
  { value: 'wind_speed', label: 'Wind Speed' }
];

const getAqiInfo = (val) => {
  // US EPA AQI 0-500 scale
  if (val <= 50) return { label: 'Good', color: 'var(--aqi-1)', icon: <CheckCircle2 size={16} /> };
  if (val <= 100) return { label: 'Moderate', color: 'var(--aqi-2)', icon: <CheckCircle2 size={16} /> };
  if (val <= 150) return { label: 'Unhealthy for Sensitive', color: 'var(--aqi-3)', icon: <AlertCircle size={16} /> };
  if (val <= 200) return { label: 'Unhealthy', color: 'var(--aqi-4)', icon: <AlertCircle size={16} /> };
  if (val <= 300) return { label: 'Very Unhealthy', color: 'var(--aqi-5)', icon: <AlertCircle size={16} /> };
  if (val > 300) return { label: 'Hazardous', color: '#7e0023', icon: <AlertCircle size={16} /> };
  return { label: 'Unknown', color: '#888', icon: <Activity size={16} /> };
};

const getParamUnit = (param) => {
  if (['pm25', 'pm10', 'no2', 'so2', 'o3'].includes(param)) return 'µg/m³';
  if (param === 'co') return 'mg/m³';
  if (param === 'temperature') return '°C';
  if (param === 'humidity') return '%';
  if (param === 'wind_speed') return 'm/s';
  return '';
};

export default function App() {
  const [cities, setCities] = useState([]);
  const [city, setCity] = useState("Delhi");
  const [param, setParam] = useState("pm25");
  
  const [loadingLive, setLoadingLive] = useState(true);
  const [loadingHist, setLoadingHist] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(true);
  
  const [liveData, setLiveData] = useState(null);
  const [histData, setHistData] = useState([]);
  const [forecastData, setForecastData] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/cities`)
      .then(res => res.json())
      .then(d => setCities(d.cities || []))
      .catch(console.error);
  }, []);

  const fetchData = async () => {
    setLoadingLive(true);
    setLoadingHist(true);
    setLoadingForecast(true);

    try {
      const liveRes = await fetch(`${API_BASE}/live/${city}`);
      const liveJson = await liveRes.json();
      setLiveData(liveJson.data);
    } catch (e) { console.error(e); setLiveData(null); } finally { setLoadingLive(false); }

    try {
      const histRes = await fetch(`${API_BASE}/history/${city}?days=5&param=${param}`);
      if (!histRes.ok) throw new Error('History fetch failed');
      const histJson = await histRes.json();
      setHistData(histJson.data || []);
    } catch (e) { console.error(e); setHistData([]); } finally { setLoadingHist(false); }

    try {
      const forecastRes = await fetch(`${API_BASE}/forecast/${city}?param=${param}`);
      if (!forecastRes.ok) throw new Error('Forecast fetch failed');
      const forecastJson = await forecastRes.json();
      setForecastData(forecastJson);
    } catch (e) { console.error(e); setForecastData(null); } finally { setLoadingForecast(false); }
  };

  useEffect(() => {
    if (city) fetchData();
    // eslint-disable-next-line
  }, [city, param]);

  // Merge history and forecast for the combined chart
  const combinedChartData = React.useMemo(() => {
    if (!histData.length) return [];
    
    // We only take the last 48 hours for a cleaner zoom on the forecast junction
    const recentHist = histData.slice(-48).map(d => ({
      timestamp: new Date(d.timestamp).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}),
      history_val: d[param],
      forecast_val: null,
      lower_bound: null,
      upper_bound: null
    }));

    const future = (forecastData?.data || []).map(d => ({
      timestamp: new Date(d.timestamp).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}),
      history_val: null,
      forecast_val: d.forecast,
      lower_bound: d.lower_bound,
      upper_bound: d.upper_bound
    }));

    // To make the line connect perfectly, the last history point should also exist as the first forecast point
    if (recentHist.length > 0 && future.length > 0) {
      const lastHist = recentHist[recentHist.length - 1];
      future.unshift({
        timestamp: lastHist.timestamp,
        history_val: null,
        forecast_val: lastHist.history_val,
        lower_bound: lastHist.history_val,
        upper_bound: lastHist.history_val
      });
    }

    return [...recentHist, ...future];
  }, [histData, forecastData, param]);


  const aqiInfo = liveData ? getAqiInfo(liveData.aqi_500 || 0) : getAqiInfo(0);
  const isRefreshing = loadingLive || loadingHist || loadingForecast;

  return (
    <div className="app-wrapper">
      
      {/* HEADER */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon">
            <CloudFog color="#fff" size={24} />
          </div>
          <div className="brand-text">
            <h1>AeroSpect</h1>
            <p>Predictive Air Quality Dashboard</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="select-wrapper">
            <select className="city-select" value={city} onChange={e => setCity(e.target.value)}>
              {cities.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <ChevronDown className="select-arrow" size={16} />
          </div>
          
          <div className="select-wrapper">
            <select className="param-select" value={param} onChange={e => setParam(e.target.value)}>
              {PARAMS.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
            </select>
            <ChevronDown className="select-arrow" size={16} />
          </div>

          <button className="btn-refresh" onClick={fetchData} disabled={isRefreshing}>
            <RefreshCw size={16} /> <span>Refresh</span>
          </button>
        </div>
      </header>

      {/* STATUS BAR */}
      <div className="status-bar">
        <div className={`status-dot ${isRefreshing ? 'loading' : ''}`} style={!isRefreshing ? {background: aqiInfo.color, boxShadow: `0 0 8px ${aqiInfo.color}`} : {}} />
        <span>{isRefreshing ? 'Syncing live telemetry & ARIMA forecasting...' : `Live connection established • Last updated: ${new Date().toLocaleTimeString()}`}</span>
      </div>

      {/* LIVE SNAPSHOT */}
      <div className="section-label">
        <h2>Live Air Quality Index</h2>
        <div className="label-line" />
      </div>

      {loadingLive ? (
        <div className="skeleton skeleton-banner" />
      ) : liveData ? (
        <div className="aqi-banner" style={{'--aqi-color': aqiInfo.color, '--aqi-glow': `${aqiInfo.color}30`}}>
          <div className="aqi-city-info">
            <h2>{city}, {liveData.country}</h2>
            <div className="city-meta">Lat: {liveData.latitude?.toFixed(2)} • Lon: {liveData.longitude?.toFixed(2)}</div>
            <div className="aqi-timestamp">
              <Activity size={12} /> {new Date(liveData.timestamp).toLocaleString()}
            </div>
          </div>
          
          <div className="aqi-pill">
            <div>
              <div className="aqi-label-text" style={{color: aqiInfo.color}}>
                {aqiInfo.label}
              </div>
              <div className="aqi-sub">US EPA AQI (based on PM2.5)</div>
            </div>
            <div className="aqi-score" style={{'--aqi-color': aqiInfo.color, '--aqi-glow': `${aqiInfo.color}40`}}>
              <span className="score-num">{liveData.aqi_500 || 0}</span>
              <span className="score-label">AQI</span>
            </div>
          </div>
        </div>
      ) : <div className="error-state">No live data available.</div>}

      {/* METRICS GRID */}
      {loadingLive ? (
        <div className="skeleton-grid">
          {[...Array(6)].map((_,i) => <div key={i} className="skeleton skeleton-card" />)}
        </div>
      ) : liveData && (
        <div className="metrics-grid">
          <MetricCard title="PM2.5" val={liveData.pm25} unit="µg/m³" icon={<Cloud size={18}/>} max={250} />
          <MetricCard title="PM10" val={liveData.pm10} unit="µg/m³" icon={<CloudFog size={18}/>} max={400} />
          <MetricCard title="NO₂" val={liveData.no2} unit="µg/m³" icon={<AlertCircle size={18}/>} max={200} />
          <MetricCard title="SO₂" val={liveData.so2} unit="µg/m³" icon={<AlertCircle size={18}/>} max={50} />
          <MetricCard title="O₃" val={liveData.o3} unit="µg/m³" icon={<Activity size={18}/>} max={180} />
          <MetricCard title="CO" val={liveData.co} unit="mg/m³" icon={<Cloud size={18}/>} max={5} />
          <MetricCard title="Temperature" val={liveData.temperature} unit="°C" icon={<ThermometerSun size={18}/>} />
          <MetricCard title="Humidity" val={liveData.humidity} unit="%" icon={<Droplets size={18}/>} />
          <MetricCard title="Wind Speed" val={liveData.wind_speed} unit="m/s" icon={<Wind size={18}/>} />
        </div>
      )}

      {/* HISTORICAL TREND */}
      <div className="section-label">
        <h2>Historical Pulse</h2>
        <div className="label-line" />
      </div>

      <div className="chart-card">
        <div className="chart-header">
          <div>
            <div className="chart-title">
              <div className="chart-title-icon"><Activity size={18} /></div>
              5-Day Historical Trend ({param.toUpperCase()})
            </div>
            <div className="chart-subtitle">Hourly sampled concentration data</div>
          </div>
        </div>
        
        {loadingHist ? <div className="skeleton skeleton-chart" /> : (
          <div style={{ width: '100%', height: 350 }}>
            <ResponsiveContainer>
              <LineChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="var(--text-muted)" 
                  fontSize={12} 
                  tickFormatter={t => new Date(t).toLocaleString([], {month:'short', day:'numeric', hour:'2-digit'})}
                  minTickGap={40}
                />
                <YAxis stroke="var(--text-muted)" fontSize={12} domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border)', color: 'var(--text-primary)' }}
                  itemStyle={{color: 'var(--text-primary)'}}
                  labelFormatter={t => new Date(t).toLocaleString()}
                />
                <Line 
                  type="monotone" 
                  dataKey={param} 
                  stroke="var(--accent-cyan)" 
                  strokeWidth={2} 
                  dot={false}
                  activeDot={{ r: 6, fill: 'var(--accent-cyan)', stroke: '#000' }}
                  animationDuration={1500}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ARIMA FORECAST */}
      <div className="section-label">
        <h2>ARIMA Predictive Model</h2>
        <div className="label-line" />
      </div>

      <div className="chart-card">
        <div className="chart-header">
          <div>
            <div className="chart-title">
              <div className="chart-title-icon"><Wind size={18} /></div>
              24-Hour Predictive Forecast ({param.toUpperCase()})
            </div>
            <div className="chart-subtitle">Using AutoRegressive Integrated Moving Average</div>
          </div>
          
          <div className="chart-legend">
            <div className="legend-item"><div className="legend-line" style={{background: 'var(--text-muted)'}}/> History</div>
            <div className="legend-item"><div className="legend-dash" style={{'--c': 'var(--accent-purple)'}}/> Forecast</div>
            <div className="legend-item"><div className="legend-dot" style={{background: 'rgba(167,139,250,0.2)'}}/> 95% Confidence</div>
          </div>
        </div>

        {loadingForecast ? <div className="skeleton skeleton-chart" /> : forecastData?.success ? (
          <>
            <div className="forecast-stats">
              <div className="fstat-card">
                <div className="fstat-label">Model Type</div>
                <div className="fstat-value">ARIMA (1, {forecastData.d_order}, 1)</div>
              </div>
              <div className="fstat-card">
                <div className="fstat-label">Param Evaluated</div>
                <div className="fstat-value">{param.toUpperCase()}</div>
              </div>
              <div className="fstat-card">
                <div className="fstat-label">Forecast Horizon</div>
                <div className="fstat-value">24 Hours</div>
              </div>
            </div>

            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <ComposedChart data={combinedChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis 
                    dataKey="timestamp" 
                    stroke="var(--text-muted)" 
                    fontSize={12} 
                    minTickGap={50}
                  />
                  <YAxis stroke="var(--text-muted)" fontSize={12} domain={['auto', 'auto']} />
                  <Tooltip 
                     contentStyle={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border)', color: 'var(--text-primary)' }}
                     itemStyle={{color: 'var(--text-primary)'}}
                  />
                  
                  {/* Historical Line */}
                  <Line 
                    type="monotone" 
                    dataKey="history_val" 
                    name="Historical" 
                    stroke="var(--text-muted)" 
                    strokeWidth={2} 
                    dot={false} 
                    activeDot={{ r: 4 }}
                    isAnimationActive={false}
                  />
                  
                  {/* Confidence Interval Area */}
                  <Area 
                    type="monotone" 
                    dataKey="upper_bound" 
                    stroke="none" 
                    fill="rgba(167,139,250,0.15)" 
                    isAnimationActive={true}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="lower_bound" 
                    stroke="none" 
                    fill="var(--bg-card)" 
                    isAnimationActive={true}
                  />
                  
                  {/* Forecast Line */}
                  <Line 
                    type="monotone" 
                    dataKey="forecast_val" 
                    name="Forecast" 
                    stroke="var(--accent-purple)" 
                    strokeWidth={2} 
                    strokeDasharray="5 5"
                    dot={{ r: 2, fill: 'var(--accent-purple)', stroke: 'none' }}
                    activeDot={{ r: 6, fill: '#fff', stroke: 'var(--accent-purple)' }}
                    animationDuration={2000}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </>
        ) : (
          <div className="error-state">
            <AlertCircle className="error-icon" />
            <div className="error-msg">ARIMA Model Failed to Converge</div>
            <div className="error-sub">{forecastData?.error || "Unknown error during statsmodels execution. Try another parameter."}</div>
          </div>
        )}
      </div>

    </div>
  );
}

function MetricCard({ title, val, unit, icon, max }) {
  const getLvlClass = () => {
    if (!max) return '';
    const pct = val / max;
    if (pct < 0.2) return 'level-good';
    if (pct < 0.4) return 'level-fair';
    if (pct < 0.6) return 'level-moderate';
    if (pct < 0.8) return 'level-poor';
    return 'level-very-poor';
  };
  
  const getBarColor = () => {
    if (!max) return 'var(--accent-blue)';
    const pct = val / max;
    if (pct < 0.2) return 'var(--aqi-1)';
    if (pct < 0.4) return 'var(--aqi-2)';
    if (pct < 0.6) return 'var(--aqi-3)';
    if (pct < 0.8) return 'var(--aqi-4)';
    return 'var(--aqi-5)';
  };

  const pct = max ? Math.min(100, (val / max) * 100) : 0;

  return (
    <div className="metric-card">
      <div className={`metric-icon ${getLvlClass()}`}>
        {icon}
      </div>
      <div className="metric-label">{title}</div>
      <div>
        <span className="metric-value">{val?.toFixed?.(1) || val}</span>
        <span className="metric-unit">{unit}</span>
      </div>
      {max && (
        <div className="metric-bar-track">
          <div className="metric-bar-fill" style={{ width: `${pct}%`, backgroundColor: getBarColor() }} />
        </div>
      )}
    </div>
  );
}

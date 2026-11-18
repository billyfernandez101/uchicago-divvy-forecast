"""UChicago Campus Divvy Forecast: a Streamlit app.

Predicts hourly bike departures at 9 Divvy stations on the University of Chicago campus,
using a LightGBM model trained on Nov 2023 - Oct 2026 trip data, the academic calendar,
and the Open-Meteo weather forecast. Live availability comes from Divvy's GBFS feed.
"""

import json
from pathlib import Path

import altair as alt
import folium
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from streamlit_folium import st_folium
from urllib3.util.retry import Retry

st.set_page_config(page_title="UChicago Campus Divvy Forecast", page_icon="🚲", layout="wide")

APP_DIR = Path(__file__).parent  # model and data files sit next to app.py in the repo
TZ = "America/Chicago"
GBFS_ROOT = "https://gbfs.lyft.com/gbfs/2.3/chi/gbfs.json"
FORECAST_HOURS = 48

TERM_LABELS = {
    "instruction": "Classes in session",
    "reading_period": "Reading period",
    "finals": "Finals week",
    "post_finals": "After finals",
    "thanksgiving": "Thanksgiving break",
    "winter_break": "Winter break",
    "spring_break": "Spring break",
    "early_summer_break": "Early summer break",
    "summer_quarter": "Summer quarter",
    "september_term": "September term",
    "late_summer_gap": "Late summer break",
    "pre_orientation": "Pre-orientation",
    "o_week": "O-Week",
}

# ---------------------------------------------------------------------------
# Text for the "How it works" tab. Edit this freely; it's plain Markdown.
# ---------------------------------------------------------------------------
ABOUT_TEXT = """
### What this is

This site predicts how many Divvy bikes will be taken out of each station on the
UChicago campus over the next 48 hours. I built it to answer a simple question:
will there be a bike when I need one?

### The data

- **Trips:** every Divvy trip from Nov 2023 to Oct 2026 that started at one of the 9 docked
  stations within 100 meters of the university's official campus boundary (the Clery Act map).
  That's about 300,000 trips. Undocked e-bike trips are left out because Divvy rounds
  their locations to about 1 km.
- **Weather:** hourly temperature, rain, snow, and wind from Open-Meteo.
- **Academic calendar:** quarter dates, reading periods, finals, breaks, O-Week, and
  pre-orientation from the registrar.

### What drives campus demand

Holding weather constant, demand during winter break is about 84% lower than during a
normal week of classes. Thanksgiving is 72% lower and spring break is 65% lower. Most of
the drop comes from students leaving campus, not from the cold.

Rain cuts demand by about 28%, and so does snow. Demand rises with temperature until about
90°F (feels-like) and then levels off. Weeks 9 and 10 of the quarter are about 12% slower
than earlier weeks even after accounting for weather, and finals week is about 24% slower.

O-Week is busier than summer quarter despite having a much smaller population on campus,
which suggests new students ride a lot. With only three O-Weeks in the data, that's
suggestive rather than proven.

### Comparing models

I compared three models on six months of data they never saw (May - Oct 2026):

{results_table}

LightGBM, a gradient-boosted tree model, beat a standard Poisson regression by about 10%.
When I let the regression give each station its own hour-by-hour pattern, the gap fell by
half. Most of what machine learning added came from that one interaction. Its remaining edge
is mostly in summer and rare parts of the calendar.

Standard errors in the regression are clustered by day, since stations share the same
weather and campus events within a day. That made the confidence intervals several times wider
than the default ones.

### Limitations

- The model predicts **departures**, not whether a station will run out. It doesn't account
  for bikes being returned, so the "may run short" warning is only a rough guide.
- The data only shows trips that happened. When a station is empty, demand gets undercounted.
- Forecasts for a single station are noisier than the campus-wide total.
- The academic calendar runs through {calendar_through}. After that the forecasts stop until
  new registrar dates are added.

### Credits

Trip and live station data: [Divvy](https://divvybikes.com/system-data) (live data via the
GBFS feed, CC BY 4.0). Weather: [Open-Meteo](https://open-meteo.com) (CC BY 4.0).
Campus boundary: UChicago Department of Safety & Security Clery Act map (March 2024).
"""


# ---------------------------------------------------------------------------
# Loading data and models
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(APP_DIR / "model.txt"))


@st.cache_data
def load_static():
    meta = json.loads((APP_DIR / "meta.json").read_text())
    stations = pd.read_json(APP_DIR / "stations.json")
    boundary = json.loads((APP_DIR / "boundary.json").read_text())
    cal = pd.read_csv(APP_DIR / "calendar.csv", parse_dates=["date"])
    return meta, stations, boundary, cal


@st.cache_resource
def http_session():
    """A shared connection that automatically retries slow or failed requests."""
    s = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "uchicago-campus-divvy-forecast (student project)"
    return s


@st.cache_resource
def last_good_forecast():
    """Keeps the most recent successful weather forecast in case the weather service is slow."""
    return {}


@st.cache_data(ttl=1800)  # refresh the weather forecast every 30 minutes
def get_weather_forecast():
    params = {
        "latitude": 41.789, "longitude": -87.599,
        "hourly": "temperature_2m,apparent_temperature,precipitation,snowfall,"
                  "wind_speed_10m,weather_code",
        "timezone": TZ, "forecast_days": 3,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
    r = http_session().get("https://api.open-meteo.com/v1/forecast", params=params,
                           timeout=(10, 45))
    r.raise_for_status()
    w = pd.DataFrame(r.json()["hourly"])
    w["hour"] = pd.to_datetime(w["time"])
    return w.drop(columns="time").rename(columns={
        "temperature_2m": "temp_f", "apparent_temperature": "feels_like_f",
        "precipitation": "precip_in", "snowfall": "snow_in", "wind_speed_10m": "wind_mph",
    })


def distance_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


@st.cache_data(ttl=60)  # refresh live availability every minute
def get_live_status(stations):
    s = http_session()
    root = s.get(GBFS_ROOT, timeout=(10, 30)).json()["data"]
    lang = "en" if "en" in root else next(iter(root))
    urls = {f["name"]: f["url"] for f in root[lang]["feeds"]}
    info = pd.DataFrame(s.get(urls["station_information"], timeout=(10, 30))
                        .json()["data"]["stations"])
    status = pd.DataFrame(s.get(urls["station_status"], timeout=(10, 30))
                          .json()["data"]["stations"])
    live = info.merge(status, on="station_id", suffixes=("", "_status"))

    # Match each campus station to the nearest live station (within 75 m)
    rows = []
    for _, s in stations.iterrows():
        d = distance_m(s["lat"], s["lng"], live["lat"].values, live["lon"].values)
        i = int(d.argmin())
        if d[i] > 75:
            continue
        row = live.iloc[i]
        rows.append({
            "campus_station": s["campus_station"],
            "bikes": int(row.get("num_bikes_available", 0) or 0),
            "ebikes": int(row.get("num_ebikes_available", 0) or 0),
            "docks": int(row.get("num_docks_available", 0) or 0),
            "renting": bool(row.get("is_renting", 1)),
        })
    return pd.DataFrame(rows)


def make_predictions(model, meta, cal, weather, station_names):
    now = pd.Timestamp.now(tz=TZ).floor("h").tz_localize(None)
    w = weather[(weather["hour"] >= now)
                & (weather["hour"] < now + pd.Timedelta(hours=FORECAST_HOURS))]
    grid = w.merge(pd.DataFrame({"campus_station": station_names}), how="cross")
    grid["date"] = grid["hour"].dt.normalize()
    grid = grid.merge(cal, on="date", how="left")

    missing_calendar = grid["term_period"].isna().any()
    grid = grid.dropna(subset=["term_period"]).copy()

    grid["hour_of_day"] = grid["hour"].dt.hour
    grid["day_of_week"] = grid["hour"].dt.dayofweek
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["month"] = grid["hour"].dt.month
    grid["week_of_quarter"] = grid["week_of_quarter"].astype(int)
    grid["is_holiday"] = grid["is_holiday"].astype(int)
    grid["campus_station"] = pd.Categorical(grid["campus_station"],
                                            categories=meta["station_categories"])
    grid["term_period"] = pd.Categorical(grid["term_period"],
                                         categories=meta["term_categories"])

    if len(grid):
        grid["pred"] = model.predict(grid[meta["features"]])
    grid["campus_station"] = grid["campus_station"].astype(str)
    grid["term_period"] = grid["term_period"].astype(str)
    return grid, missing_calendar


def hourly_chart(df, title):
    return (alt.Chart(df, title=title)
            .mark_bar(color="#800000")
            .encode(
                x=alt.X("hour:T", title=None, axis=alt.Axis(format="%a %-I %p", labelAngle=0)),
                y=alt.Y("pred:Q", title="Expected departures"),
                tooltip=[
                    alt.Tooltip("hour:T", title="Hour", format="%a %b %-d, %-I %p"),
                    alt.Tooltip("pred:Q", title="Expected departures", format=".1f"),
                    alt.Tooltip("temp_f:Q", title="Temp (°F)", format=".0f"),
                    alt.Tooltip("precip_in:Q", title="Rain (in)", format=".2f"),
                ])
            .properties(height=280))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
model = load_model()
meta, stations, boundary, cal = load_static()
station_names = stations["campus_station"].tolist()

st.title("🚲 UChicago Campus Divvy Forecast")
st.caption("Hourly bike demand forecasts for the 9 Divvy stations on campus, "
           "based on the weather forecast and the academic calendar.")

backup = last_good_forecast()
try:
    weather = get_weather_forecast()
    backup["weather"] = weather
    backup["fetched_at"] = pd.Timestamp.now(tz=TZ)
except Exception:
    if "weather" in backup:
        weather = backup["weather"]
        st.info("The weather service is slow right now, so these forecasts use the weather "
                f"forecast from {backup['fetched_at']:%-I:%M %p}.")
    else:
        st.error("The weather service isn't responding right now, so forecasts can't load. "
                 "Refresh the page in a minute or two.")
        st.stop()

preds, missing_calendar = make_predictions(model, meta, cal, weather, station_names)

if missing_calendar:
    st.warning(f"The academic calendar in this app only runs through {meta['calendar_through']}. "
               "Forecasts past that date aren't available until new registrar dates are added.")
if preds.empty:
    st.stop()

try:
    live = get_live_status(stations)
except Exception:
    live = pd.DataFrame(columns=["campus_station", "bikes", "ebikes", "docks", "renting"])

today = preds.iloc[0]
period = TERM_LABELS.get(today["term_period"], today["term_period"])
if today["term_period"] == "instruction" and today["week_of_quarter"] > 0:
    period += f", week {today['week_of_quarter']}"
st.markdown(f"**Academic calendar today:** {period}")

tab_forecast, tab_map, tab_about = st.tabs(["Forecast", "Live map", "How it works"])

# ---- Forecast tab ----
with tab_forecast:
    station = st.selectbox("Station", station_names)
    sp = preds[preds["campus_station"] == station].sort_values("hour")
    next_hour = sp["pred"].iloc[0]
    next_6 = sp["pred"].iloc[:6].sum()

    lv = live[live["campus_station"] == station]
    c1, c2, c3 = st.columns(3)
    if len(lv):
        bikes = int(lv["bikes"].iloc[0])
        c1.metric("Bikes available now", bikes,
                  help=f"Includes {int(lv['ebikes'].iloc[0])} e-bikes. Live from Divvy.")
    else:
        bikes = None
        c1.metric("Bikes available now", "n/a", help="Live data unavailable right now.")
    c2.metric("Expected departures, next hour", f"{next_hour:.1f}")
    c3.metric("Expected departures, next 6 hours", f"{next_6:.0f}")

    if bikes is not None and bikes < next_hour:
        st.warning("Bikes may run short here in the next hour. This doesn't account for "
                   "bikes being returned, so treat it as a rough guide.")

    st.altair_chart(hourly_chart(sp, f"{station}: next {FORECAST_HOURS} hours"),
                    use_container_width=True)

    campus = (preds.groupby("hour", as_index=False)
              .agg(pred=("pred", "sum"), temp_f=("temp_f", "first"),
                   precip_in=("precip_in", "first")))
    st.altair_chart(hourly_chart(campus, "All 9 campus stations combined"),
                    use_container_width=True)
    st.caption("Campus-wide totals are more reliable than single-station forecasts.")

# ---- Live map tab ----
with tab_map:
    next_hr = preds[preds["hour"] == preds["hour"].min()][["campus_station", "pred"]]
    mp = stations.merge(next_hr, on="campus_station").merge(live, on="campus_station", how="left")

    m = folium.Map(location=[41.789, -87.597], zoom_start=15, tiles="cartodbpositron")
    folium.Polygon(boundary, color="#800000", weight=3, fill=True, fill_opacity=0.05,
                   tooltip="UChicago campus (Clery Act boundary)").add_to(m)
    for _, r in mp.iterrows():
        if pd.isna(r.get("bikes")):
            color, bikes_txt = "gray", "live data unavailable"
        else:
            color = "red" if r["bikes"] == 0 else "orange" if r["bikes"] <= 2 else "green"
            bikes_txt = f"{int(r['bikes'])} bikes ({int(r['ebikes'])} e-bikes), {int(r['docks'])} open docks"
        folium.CircleMarker(
            [r["lat"], r["lng"]], radius=6 + 3 * min(r["pred"], 6),
            color=color, fill=True, fill_opacity=0.8,
            tooltip=f"<b>{r['campus_station']}</b><br>{bikes_txt}"
                    f"<br>Expected departures next hour: {r['pred']:.1f}",
        ).add_to(m)
    st_folium(m, height=560, use_container_width=True, returned_objects=[])
    st.caption("Color = bikes available now (green 3+, orange 1-2, red 0). "
               "Size = expected departures in the next hour.")

# ---- How it works tab ----
with tab_about:
    results = pd.DataFrame(meta["test_results"]).T.round(3)
    table_md = results.to_markdown() if hasattr(results, "to_markdown") else results.to_string()
    st.markdown(ABOUT_TEXT.format(results_table=table_md,
                                  calendar_through=meta["calendar_through"]))

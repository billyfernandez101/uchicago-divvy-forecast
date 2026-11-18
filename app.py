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

st.set_page_config(page_title="UChicago Campus Divvy Forecast", layout="centered")

APP_DIR = Path(__file__).parent  # model and data files sit next to app.py in the repo
TZ = "America/Chicago"
GBFS_ROOT = "https://gbfs.lyft.com/gbfs/2.3/chi/gbfs.json"
FORECAST_HOURS = 48
AVAILABILITY_HOURS = 12
MAROON = "#800000"

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

# Light styling: tighter spacing and smaller headings on phones.
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
@media (max-width: 640px) {
  .block-container {padding-left: 1rem; padding-right: 1rem; padding-top: 1.25rem;}
  h1 {font-size: 1.65rem !important;}
  h3 {font-size: 1.15rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Text for the "How it works" tab. Edit freely; it's plain Markdown.
# The results table is shown between the two parts.
# ---------------------------------------------------------------------------
ABOUT_PART_1 = """
### What this site does

This site forecasts how many Divvy bikes will be taken out of each station on the
UChicago campus, hour by hour, for the next two days. It also shows how many bikes are
at each station right now. I built it because I wanted to know whether a bike would be
there when I needed one.

### How to read the forecast

**Expected departures** is the number of bikes the model expects people to take from a
station during that hour. It's an average, so a forecast of 2.4 means you'd usually see
2 or 3 bikes taken out, sometimes more and sometimes fewer.

**Bikes available now** comes from Divvy's live feed and updates about once a minute.

**Bikes left if none are returned** starts from the live count and subtracts the expected
departures hour by hour. It's a worst case. People return bikes all day, especially at
busy stations, so the real number is usually higher. If this line stays above zero, you
can be fairly confident a bike will be there.

### How a forecast is made

Each forecast combines four things: the station, the time of day and day of the week,
where we are in the academic calendar (classes, finals, breaks, O-Week), and the weather
forecast for that hour. These go into a LightGBM model, a type of machine learning model
that builds hundreds of small decision trees, each one correcting the mistakes of the
ones before it.

The model learned from about 300,000 Divvy trips taken between November 2023 and
October 2026. The weather forecast refreshes every 30 minutes, so the predictions shift
as the forecast changes.

### The data

- **Trips:** every Divvy trip from Nov 2023 to Oct 2026 that started at one of the 9 docked
  stations within 100 meters of the university's official campus boundary (the Clery Act
  map). Undocked e-bike trips are left out because Divvy rounds their locations to about 1 km.
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

### Testing the model

To test the model fairly, I trained it on data through April 2026 and then checked its
forecasts against May to October 2026, six months it had never seen. I compared it with a
simple historical average and with two versions of a Poisson regression, a standard
statistical model for count data:
"""

ABOUT_PART_2 = """
Lower is better in every column. MAE is the average miss, in trips per station per hour.
RMSE weighs big misses more heavily. Poisson deviance is a standard accuracy measure for
count data.

LightGBM beat the standard Poisson regression by about 10%. When I let the regression give
each station its own hour-by-hour pattern, the gap fell by half, so most of what machine
learning added came from that one pattern. Its remaining edge is mostly in summer and in
rare parts of the calendar.

Standard errors in the regression are clustered by day, since all stations share the same
weather and campus events on a given day. That made the confidence intervals several times
wider than the default ones.

### Limitations

- The model predicts **departures**, not returns, so it can't say exactly how many bikes a
  station will have later. The "bikes left" chart is a worst case for that reason.
- The data only shows trips that happened. When a station is empty, demand gets undercounted.
- Forecasts for a single station are noisier than the campus-wide total.
- The academic calendar runs through {calendar_through}. After that, forecasts stop until
  new registrar dates are added.

### Credits

Trip and live station data: [Divvy](https://divvybikes.com/system-data) (live data via the
GBFS feed, CC BY 4.0). Weather: [Open-Meteo](https://open-meteo.com) (CC BY 4.0).
Campus boundary: UChicago Department of Safety & Security Clery Act map (March 2024).
"""


# ---------------------------------------------------------------------------
# Loading data, models, weather, and live station status
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
    for _, stn in stations.iterrows():
        d = distance_m(stn["lat"], stn["lng"], live["lat"].values, live["lon"].values)
        i = int(d.argmin())
        if d[i] > 75:
            continue
        row = live.iloc[i]
        rows.append({
            "campus_station": stn["campus_station"],
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def describe_weather(code):
    """Plain-English description of an Open-Meteo weather code."""
    code = int(code)
    if code == 0:
        return "clear"
    if code in (1, 2):
        return "partly cloudy"
    if code == 3:
        return "cloudy"
    if code in (45, 48):
        return "fog"
    if 51 <= code <= 57:
        return "drizzle"
    if 61 <= code <= 67:
        return "rain"
    if 71 <= code <= 77:
        return "snow"
    if 80 <= code <= 82:
        return "rain showers"
    if code in (85, 86):
        return "snow showers"
    if code >= 95:
        return "thunderstorms"
    return ""


def project_bikes(bikes_now, station_preds, hours=AVAILABILITY_HOURS):
    """Worst case: start from the live count and subtract expected departures,
    assuming no bikes are returned. Only the rest of the current hour counts."""
    now = pd.Timestamp.now(tz=TZ).tz_localize(None)
    sp = station_preds.iloc[:hours]
    share_of_hour = np.ones(len(sp))
    share_of_hour[0] = 1 - now.minute / 60
    left = bikes_now - np.cumsum(sp["pred"].values * share_of_hour)
    times = [now] + list(sp["hour"] + pd.Timedelta(hours=1))
    bikes = [bikes_now] + list(np.maximum(np.floor(left), 0))
    return pd.DataFrame({"time": times, "bikes": bikes})


TIME_AXIS = alt.Axis(format="%a %-I %p", labelAngle=0, labelOverlap="greedy", grid=False)


def departures_chart(df):
    return (alt.Chart(df)
            .mark_bar(color=MAROON)
            .encode(
                x=alt.X("hour:T", title=None, axis=TIME_AXIS),
                y=alt.Y("pred:Q", title="Expected departures"),
                tooltip=[
                    alt.Tooltip("hour:T", title="Hour", format="%a %b %-d, %-I %p"),
                    alt.Tooltip("pred:Q", title="Expected departures", format=".1f"),
                    alt.Tooltip("temp_f:Q", title="Temperature (°F)", format=".0f"),
                    alt.Tooltip("precip_in:Q", title="Rain (in)", format=".2f"),
                ])
            .properties(height=230))


def availability_chart(df):
    return (alt.Chart(df)
            .mark_line(color=MAROON, point=alt.OverlayMarkDef(color=MAROON, size=35))
            .encode(
                x=alt.X("time:T", title=None, axis=TIME_AXIS),
                y=alt.Y("bikes:Q", title="Bikes", scale=alt.Scale(domainMin=0)),
                tooltip=[
                    alt.Tooltip("time:T", title="Time", format="%a %-I:%M %p"),
                    alt.Tooltip("bikes:Q", title="Bikes left (worst case)", format=".0f"),
                ])
            .properties(height=210))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
model = load_model()
meta, stations, boundary, cal = load_static()
station_names = stations["campus_station"].tolist()

st.title("UChicago Campus Divvy Forecast")
st.write("Hourly forecasts of bike demand at the 9 Divvy stations on campus, "
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

current = preds.iloc[0]
period = TERM_LABELS.get(current["term_period"], current["term_period"])
if current["term_period"] == "instruction" and current["week_of_quarter"] > 0:
    period += f", week {current['week_of_quarter']}"
conditions = describe_weather(current["weather_code"])
weather_now = f"{current['temp_f']:.0f}°F" + (f", {conditions}" if conditions else "")
st.caption(f"Academic calendar: {period}  ·  Weather now: {weather_now}")

tab_forecast, tab_map, tab_about = st.tabs(["Forecast", "Map", "How it works"])

# ---- Forecast tab ----
with tab_forecast:
    station = st.selectbox("Choose a station", station_names)
    window = st.radio("Show the next", ["12 hours", "24 hours", "48 hours"],
                      index=1, horizontal=True)
    n_hours = int(window.split()[0])

    sp = preds[preds["campus_station"] == station].sort_values("hour")
    sp_window = sp.iloc[:n_hours]

    lv = live[live["campus_station"] == station]
    bikes = int(lv["bikes"].iloc[0]) if len(lv) else None

    c1, c2, c3 = st.columns(3)
    if bikes is not None:
        c1.metric("Bikes available now", bikes,
                  help=f"Includes {int(lv['ebikes'].iloc[0])} e-bikes. Live from Divvy.")
    else:
        c1.metric("Bikes available now", "n/a", help="Live data is unavailable right now.")
    c2.metric("Expected departures, next hour", f"{sp['pred'].iloc[0]:.1f}")
    c3.metric("Expected departures, next 6 hours", f"{sp['pred'].iloc[:6].sum():.0f}")

    peak = sp_window.loc[sp_window["pred"].idxmax()]
    st.write(f"Busiest hour in the next {n_hours} hours: **{peak['hour']:%A %-I %p}**, "
             f"with about {peak['pred']:.1f} expected departures.")

    if bikes is not None and bikes < sp["pred"].iloc[0]:
        st.warning("Bikes may run short here in the next hour. This doesn't account for "
                   "bikes being returned, so treat it as a rough guide.")

    st.subheader("Bikes left if none are returned")
    if bikes is not None:
        st.altair_chart(availability_chart(project_bikes(bikes, sp)), width="stretch")
        st.caption(f"A worst case for the next {AVAILABILITY_HOURS} hours: today's live count "
                   "minus expected departures. Returns usually keep the real number higher.")
    else:
        st.caption("Live bike counts are unavailable right now, so this chart can't be shown.")

    st.subheader("Expected departures by hour")
    st.altair_chart(departures_chart(sp_window), width="stretch")

    st.subheader("All campus stations combined")
    campus = (preds.groupby("hour", as_index=False)
              .agg(pred=("pred", "sum"), temp_f=("temp_f", "first"),
                   precip_in=("precip_in", "first"))
              .iloc[:n_hours])
    st.altair_chart(departures_chart(campus), width="stretch")
    st.caption("Campus-wide totals are more reliable than forecasts for a single station.")

# ---- Map tab ----
with tab_map:
    next_hr = preds[preds["hour"] == preds["hour"].min()][["campus_station", "pred"]]
    mp = stations.merge(next_hr, on="campus_station").merge(live, on="campus_station", how="left")

    m = folium.Map(location=[41.789, -87.597], zoom_start=15, tiles="OpenStreetMap")
    folium.Polygon(boundary, color=MAROON, weight=3, fill=True, fill_opacity=0.05,
                   tooltip="UChicago campus (Clery Act boundary)").add_to(m)
    for _, r in mp.iterrows():
        if pd.isna(r.get("bikes")):
            color, bikes_txt = "gray", "Live data unavailable"
        else:
            color = "red" if r["bikes"] == 0 else "orange" if r["bikes"] <= 2 else "green"
            bikes_txt = (f"{int(r['bikes'])} bikes ({int(r['ebikes'])} e-bikes), "
                         f"{int(r['docks'])} open docks")
        label = (f"<b>{r['campus_station']}</b><br>{bikes_txt}"
                 f"<br>Expected departures next hour: {r['pred']:.1f}")
        folium.CircleMarker(
            [r["lat"], r["lng"]], radius=6 + 3 * min(r["pred"], 6),
            color=color, fill=True, fill_opacity=0.8,
            tooltip=label, popup=folium.Popup(label, max_width=260),
        ).add_to(m)
    st_folium(m, height=450, use_container_width=True, returned_objects=[])
    st.caption("Tap or hover over a station for details. Color shows bikes available now "
               "(green 3 or more, orange 1 to 2, red none). Size shows expected departures "
               "in the next hour.")

# ---- How it works tab ----
with tab_about:
    st.markdown(ABOUT_PART_1)
    results = pd.DataFrame(meta["test_results"]).T.round(3)
    results.index.name = "Model"
    st.dataframe(results, width="stretch")
    st.markdown(ABOUT_PART_2.format(calendar_through=meta["calendar_through"]))

st.caption("Live bike counts update every minute. Weather from Open-Meteo, updated every "
           "30 minutes. Not affiliated with Divvy or the University of Chicago.")

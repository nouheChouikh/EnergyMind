"""Step 2 - hourly context of the street: temperature, weekend, holidays, period.

Temperature: Open-Meteo historical weather archive (ERA5 reanalysis, CC BY 4.0), one point at
the capital of the street's province. The download is cached in
dataset/data/sources/openmeteo_temperatures.csv, so a rebuild needs no internet.
Holidays: Spanish national + regional calendar from the `holidays` package.
Output: dataset/data/system_inputs/hourly_context.csv
"""
from collections import Counter

import numpy as np
import pandas as pd

import config as C
from shared.helpers import hours, period_labels, province_label, write_csv

API = "https://archive-api.open-meteo.com/v1/archive"


def street_province():
    homes = pd.read_csv(C.SELECTED_HOMES, dtype=str, keep_default_na=False)
    keys = [k for k in homes["province_key"] if k in C.PROVINCES]
    return Counter(keys).most_common(1)[0][0] if keys else C.DEFAULT_PROVINCE


def download(lat, lon):
    import requests
    params = dict(latitude=lat, longitude=lon, start_date="2017-12-31", end_date="2020-01-01",
                  hourly="temperature_2m", timezone="GMT")
    resp = requests.get(API, params=params, timeout=120)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return pd.DataFrame({"latitude": lat, "longitude": lon, "time_utc": hourly["time"],
                         "temperature_c": hourly["temperature_2m"]})


def temperature(spec):
    raw = None
    if C.WEATHER_CACHE.exists():
        raw = pd.read_csv(C.WEATHER_CACHE)
        if not (np.isclose(raw["latitude"].iat[0], spec["lat"]) and np.isclose(raw["longitude"].iat[0], spec["lon"])):
            raw = None
        else:
            print(f"  temperatures: cached file {C.WEATHER_CACHE.name}")
    if raw is None:
        try:
            raw = download(spec["lat"], spec["lon"])
            write_csv(raw, C.WEATHER_CACHE)
            print(f"  temperatures: downloaded {len(raw)} hours from Open-Meteo")
        except Exception as exc:
            print(f"  WARNING: temperature download failed ({type(exc).__name__}). temperature_c stays empty;"
                  " rerun `python dataset/builder/build_dataset.py` with internet to fill it.")
            return pd.Series(np.nan, index=hours())
    # UTC -> Spanish wall-clock time, handled exactly like GoiEner did:
    # the repeated autumn hour is averaged, the skipped spring hour is filled.
    local = (pd.to_datetime(raw["time_utc"]).dt.tz_localize("UTC")
             .dt.tz_convert("Europe/Madrid").dt.tz_localize(None))
    s = pd.Series(raw["temperature_c"].to_numpy(dtype=float), index=pd.DatetimeIndex(local))
    s = s.groupby(level=0).mean()
    s = s.reindex(hours()).interpolate(limit_direction="both")
    return s


def holiday_names(region):
    import holidays
    try:
        cal = holidays.country_holidays("ES", subdiv=region, years=[2018, 2019], language="en_US")
    except Exception:
        cal = holidays.country_holidays("ES", subdiv=region, years=[2018, 2019])
    return {pd.Timestamp(day): name for day, name in sorted(cal.items())}


def build_context():
    key = street_province()
    spec = C.PROVINCES[key]
    print(f"  street province: {province_label(key)} (weather point {spec['lat']}, {spec['lon']},"
          f" holiday calendar ES-{spec['holidays']})")
    idx = hours()
    days = idx.normalize()
    names = holiday_names(spec["holidays"])
    temp = temperature(spec)
    return pd.DataFrame({
        "timestamp": idx,
        "temperature_c": np.round(temp.to_numpy(dtype=float), 1),
        "is_weekend": (idx.dayofweek >= 5).astype("int8"),
        "is_holiday": days.isin(list(names)).astype("int8"),
        "holiday_name": [names.get(d, "") for d in days],
        "period": period_labels(idx),
    })


def main():
    ctx = build_context()
    write_csv(ctx, C.HOURLY_CONTEXT)
    n_hol = ctx.loc[ctx["is_holiday"] == 1, "timestamp"].dt.normalize().nunique()
    print(f"  {C.HOURLY_CONTEXT.name}: {len(ctx)} hours, {n_hol} holidays, "
          f"temperature {ctx['temperature_c'].min()} to {ctx['temperature_c'].max()} C")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cyclical_encode(values, period):
    sin = np.sin(2 * np.pi * values / period)
    cos = np.cos(2 * np.pi * values / period)
    return sin, cos

def hourly_data_loader() -> pd.DataFrame:

    # --- Electric: hourly kWh ---
    elec1 = pd.read_csv("data/UsageData-01_01_2022-11_26_2024-clean.csv", parse_dates=["timestamp"])
    elec2 = pd.read_csv("data/UsageData-11_27_2024-03_31_2026-clean.csv", parse_dates=["timestamp"])
    elec2["timestamp"] = pd.to_datetime(elec2["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elec = pd.concat([elec1, elec2], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    # --- Meteorological: hourly observations ---
    # Keeping: temp, rhum, prcp, wspd, pres, wdir — dropping sparse/derived cols (snwd, wpgt, tsun, cldc, coco, wchill)
    WEATHER_COLS = ["temp", "rhum", "prcp", "wspd", "pres", "wdir"]
    met1 = pd.read_csv("data/meteorological_observations_1_1_2022-11_24_2024.csv", parse_dates=["time"], usecols=["time"] + WEATHER_COLS)
    met2 = pd.read_csv("data/meteorological_observations_11_27_2024-3_31_2026.csv", parse_dates=["time"], usecols=["time"] + WEATHER_COLS)
    met = pd.concat([met1, met2], ignore_index=True).sort_values("time").reset_index(drop=True)

    # Fill nan prcp values with 0
    met["prcp"] = met["prcp"].fillna(0)

    # --- Merge on hour ---
    df = pd.merge(elec.rename(columns={"timestamp": "time"}), met, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)

    # Time behavior
    df["hour"]       = df["time"].dt.hour
    df["dayofweek"]  = df["time"].dt.dayofweek   # 0=Mon … 6=Sun
    df["month"]      = df["time"].dt.month
    df['is_weekend'] = (df["dayofweek"] >= 5).astype(int)
    df['awake'] = ((df['hour'] >= 6) & (df['hour'] <= 23)).astype(int)
    df['evening'] = ((df['hour'] >= 16) & (df['hour'] <= 21)).astype(int)

    # Temperatue lag to model thermal effects
    df["temp_lag1"]  = df["temp"].shift(1)
    df["temp_lag3"]  = df["temp"].shift(3)
    df["temp_lag6"] = df["temp"].shift(6)
    df["temp_avg_12"] = df["temp"].shift(1).rolling(12).mean() #previous 12 hours avg temp
    
    # Humidity lag to capture drying/humidification
    df["rhum_lag3"] = df["rhum"].shift(3)
    
    # Cumulative precip
    df["prcp_sum_12"] = df["prcp"].shift(1).rolling(12).sum()
    
    # cyclically encoded values
    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour"], 24)
    df["dayofweek_sin"],  df["dayofweek_cos"]  = cyclical_encode(df["dayofweek"], 7)
    df["month_sin"],  df["month_cos"]  = cyclical_encode(df["month"], 12)
    df["wdir_sin"], df["wdir_cos"] = cyclical_encode(df["wdir"], 360)

    # Drop missing values
    df = df.dropna(axis=0)
    return df

def daily_data_loader() -> pd.DataFrame:
    
    # --- Electric: hourly kWh → aggregate to daily totals ---
    elec1 = pd.read_csv("data/UsageData-01_01_2022-11_26_2024-clean.csv", parse_dates=["timestamp"])
    elec2 = pd.read_csv("data/UsageData-11_27_2024-03_31_2026-clean.csv", parse_dates=["timestamp"])
    elec2["timestamp"] = pd.to_datetime(elec2["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    elec = pd.concat([elec1, elec2], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    elec["date"] = elec["timestamp"].dt.normalize()
    elec_daily = elec.groupby("date", as_index=False)["kwh"].sum()

    # --- Meteorological: hourly observations → aggregate to daily ---
    WEATHER_COLS = ["temp", "rhum", "prcp", "wspd", "pres", "wdir"]
    met1 = pd.read_csv("data/meteorological_observations_1_1_2022-11_24_2024.csv", parse_dates=["time"], usecols=["time"] + WEATHER_COLS)
    met2 = pd.read_csv("data/meteorological_observations_11_27_2024-3_31_2026.csv", parse_dates=["time"], usecols=["time"] + WEATHER_COLS)
    met = pd.concat([met1, met2], ignore_index=True).sort_values("time").reset_index(drop=True)
    met["date"] = met["time"].dt.normalize()
    met["prcp"] = met["prcp"].fillna(0)

    met_daily = met.groupby("date", as_index=False).agg(
        temp_mean=("temp", "mean"),
        temp_min = ("temp", "min"),
        temp_max = ("temp", "max"),
        rhum_mean=("rhum", "mean"),
        rhum_min=("rhum", "min"),
        rhum_max=("rhum", "max"),
        prcp_sum=("prcp", "sum"),
        wspd_mean=("wspd", "mean"),
        wspd_min=("wspd", "min"),
        wspd_max=("wspd", "max"),
        pres_mean=("pres", "mean"),
        wdir_mean=("wdir", "mean"),
    )

    # --- Merge on date ---
    df = pd.merge(elec_daily, met_daily, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    df["dayofweek"]  = df["date"].dt.dayofweek   # 0=Mon … 6=Sun
    df["month"]      = df["date"].dt.month
    df['is_weekend'] = (df["dayofweek"] >= 5).astype(int)
    
    df["dayofweek_sin"],  df["dayofweek_cos"]  = cyclical_encode(df["dayofweek"], 7)
    df["month_sin"],  df["month_cos"]  = cyclical_encode(df["month"], 12)
    df["wdir_mean_sin"], df["wdir_mean_cos"] = cyclical_encode(df["wdir_mean"], 360)

    # Drop missing values
    df = df.dropna(axis=0)
    return df
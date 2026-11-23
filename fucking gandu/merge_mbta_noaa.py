"""
DS 2500 Team Project - Step 1 (Pushan): merge MBTA performance data with NOAA weather.

Output: one row per (service_date, line) with daily reliability measures + weather
columns + flags (rain/dry/snow/heat/cold) that every teammate's analysis can reuse.

Usage (from the project folder):
    python merge_mbta_noaa.py \
        --travel "TravelTimes_2023/*HR*TravelTimes*.csv" \
                 "TravelTimes_2025/*HR*TravelTimes*.csv" \
        --noaa USW00014739.csv --out merged_daily.csv

Travel-time files are read in chunks. Their empirical baseline is estimated
from a reproducible 5% sample before a second pass creates daily summaries.

Data notes
  * MBTA files: https://www.mbta.com/developers/historical-performance-data
    (Rapid Transit Headways + Travel Times, monthly CSVs).
  * NOAA: GHCND:USW00014739 (Boston Logan). Pull with units=standard so that
    PRCP/SNOW/SNWD are in inches and TMAX/TMIN in deg F, e.g.
    https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries
      &stations=USW00014739&startDate=2022-01-01&endDate=2025-12-31
      &dataTypes=PRCP,SNOW,SNWD,TMAX,TMIN,AWND,WSF2&units=standard&format=csv
  * MBTA column names have changed between releases, so each column is looked up
    from a list of candidates below. If a file uses another name, add it here.
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd

# ---- candidate column names (first match wins) -------------------------------
COLS = {
    "date":        ["service_date", "service date"],
    "route":       ["route_id", "route"],
    "headway":     ["headway_time_sec", "headway"],
    "bench_hw":    ["benchmark_headway_time_sec", "scheduled_headway"],
    "travel":      ["travel_time_sec"],
    "bench_tt":    ["benchmark_travel_time_sec"],
}
LINES = {"Red", "Orange", "Blue", "Green"}      # Mattapan trolley is excluded
LONG_GAP_FACTOR = 1.5                           # a "long gap" = headway > 1.5x scheduled
MAX_HEADWAY_SEC = 7200                          # drop >2h headways (overnight / data errors)


def pick(cols, key, path):
    lowered = {c.lower().strip(): c for c in cols}
    for cand in COLS[key]:
        if cand in lowered:
            return lowered[cand]
    sys.exit(f"[{path}] none of {COLS[key]} found. Columns present: {list(cols)}")


def add_line(df, route_col):
    # Green-B/C/D/E -> Green ; Red/Orange/Blue unchanged
    df["line"] = df[route_col].astype(str).str.split("-").str[0]
    return df[df["line"].isin(LINES)]


CHUNK = 200_000   # rows read at a time, so multi-GB files never sit fully in memory


def _combine(parts, sum_cols, count_col):
    t = pd.concat(parts).groupby(["date", "line"], as_index=False)[sum_cols + [count_col]].sum()
    return t


def summarize_headways(path):
    head = pd.read_csv(path, nrows=0)
    d, r = pick(head.columns, "date", path), pick(head.columns, "route", path)
    h, b = pick(head.columns, "headway", path), pick(head.columns, "bench_hw", path)
    parts = []
    for df in pd.read_csv(path, usecols=[d, r, h, b], chunksize=CHUNK):
        df = df.rename(columns={d: "date", r: "route", h: "hw", b: "bench"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = add_line(df, "route").dropna(subset=["hw", "bench"])
        df = df[(df["bench"] > 0) & (df["hw"] > 0) & (df["hw"] <= MAX_HEADWAY_SEC)]
        df["hw_dev"] = df["hw"] - df["bench"]            # seconds longer than scheduled
        df["hw_ratio"] = df["hw"] / df["bench"]          # 1.0 = exactly on schedule
        df["long_gap"] = (df["hw_ratio"] > LONG_GAP_FACTOR).astype(float)
        parts.append(df.groupby(["date", "line"], as_index=False)
                       .agg(s_dev=("hw_dev", "sum"), s_ratio=("hw_ratio", "sum"),
                            s_gap=("long_gap", "sum"), n_headways=("hw", "size")))
    t = _combine(parts, ["s_dev", "s_ratio", "s_gap"], "n_headways")
    t["hw_dev_mean"] = t["s_dev"] / t["n_headways"]
    t["hw_ratio_mean"] = t["s_ratio"] / t["n_headways"]
    t["pct_long_gaps"] = t["s_gap"] / t["n_headways"]
    return t[["date", "line", "hw_dev_mean", "hw_ratio_mean", "pct_long_gaps", "n_headways"]]


BASELINE_KEYS = [
    "route_id", "direction_id", "from_stop_id", "to_stop_id", "hour", "is_weekend",
]
ANALYSIS_LINES = {"Red", "Orange", "Blue"}
BASELINE_SAMPLE_FRAC = 0.05


def _read_travel_chunks(path):
    columns = pd.read_csv(path, nrows=0).columns
    time_col = next(
        (col for col in ["start_time_sec", "from_stop_departure_sec"] if col in columns),
        None,
    )
    if time_col is None:
        raise ValueError(f"[{path}] no supported departure-time column found")
    usecols = [
        "service_date", "from_stop_id", "to_stop_id", "route_id",
        "direction_id", time_col, "travel_time_sec",
    ]
    for df in pd.read_csv(
            path, usecols=usecols,
            dtype={"from_stop_id": "string", "to_stop_id": "string",
                   "direction_id": "string"},
            chunksize=CHUNK):
        yield df.rename(columns={time_col: "start_time_sec"})


def _prepare_travel_chunk(df):
    df = df[df["route_id"].isin(ANALYSIS_LINES)].copy()
    df["date"] = pd.to_datetime(df["service_date"]).dt.normalize()
    df["travel_time_sec"] = pd.to_numeric(df["travel_time_sec"], errors="coerce")
    df["start_time_sec"] = pd.to_numeric(df["start_time_sec"], errors="coerce")
    df["hour"] = (df["start_time_sec"] // 3600).mod(24)
    df["is_weekend"] = df["date"].dt.day_of_week >= 5
    return df


def summarize_travel_files(paths):
    # Pass 1: take a reproducible 5% sample from each file to estimate baselines.
    sample_parts = []
    for file_number, path in enumerate(paths):
        print(f"  Baseline sample: {path}")
        for chunk_number, df in enumerate(_read_travel_chunks(path)):
            df = _prepare_travel_chunk(df)
            sample = df[df["travel_time_sec"] > 0].sample(
                frac=BASELINE_SAMPLE_FRAC,
                random_state=2023 + file_number * 100_003 + chunk_number,
            )
            sample_parts.append(sample[BASELINE_KEYS + ["travel_time_sec"]])

    samples = pd.concat(sample_parts, ignore_index=True)
    baselines = (samples.groupby(BASELINE_KEYS, as_index=False)["travel_time_sec"]
                 .median().rename(columns={"travel_time_sec": "baseline"}))
    del samples, sample_parts

    # Pass 2: apply the sampled baselines and accumulate daily sums and counts.
    parts = []
    dropped_rows = 0
    for path in paths:
        print(f"  Summarizing: {path}")
        for df in _read_travel_chunks(path):
            df = _prepare_travel_chunk(df)
            dropped_rows += len(df)
            df = df.merge(baselines, on=BASELINE_KEYS, how="left", validate="m:1")
            valid = (
                df["baseline"].notna()
                & (df["baseline"] > 0)
                & (df["travel_time_sec"] > 0)
                & (df["travel_time_sec"] <= 3 * df["baseline"])
            )
            dropped_rows -= int(valid.sum())
            df = df.loc[valid].copy()
            df["tt_dev"] = df["travel_time_sec"] - df["baseline"]
            df["tt_ratio"] = df["travel_time_sec"] / df["baseline"]
            parts.append(df.groupby(["date", "route_id"], as_index=False)
                         .agg(s_dev=("tt_dev", "sum"), s_ratio=("tt_ratio", "sum"),
                              n_travel_times=("travel_time_sec", "size"))
                         .rename(columns={"route_id": "line"}))

    t = (pd.concat(parts, ignore_index=True)
         .groupby(["date", "line"], as_index=False)[
             ["s_dev", "s_ratio", "n_travel_times"]].sum())
    t["tt_dev_mean"] = t["s_dev"] / t["n_travel_times"]
    t["tt_ratio_mean"] = t["s_ratio"] / t["n_travel_times"]
    print(f"  Dropped {dropped_rows:,} HR trip rows (invalid values, missing baseline, or >3x baseline).")
    return t[["date", "line", "tt_dev_mean", "tt_ratio_mean", "n_travel_times"]]


def load_many(pattern, fn):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched {pattern}")
    parts = []
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f}")
        parts.append(fn(f))
    out = pd.concat(parts, ignore_index=True)
    # if a service date appears in two files, recombine by simple mean of the duplicates
    return out.groupby(["date", "line"], as_index=False).mean(numeric_only=True)


def load_noaa(path):
    """Accepts three weather formats (auto-detected):
      1. NOAA GHCN-Daily 'by_station' file (USW00014739.csv or .csv.gz, no header, long format)
      2. NOAA Climate Data Online / NCEI Data Service CSV (columns DATE, PRCP, SNOW, TMAX...)
      3. Open-Meteo archive CSV (columns time, precipitation_sum (inch), ...)
    Output columns: date, prcp (in), snow (in), snwd (in), tmax (F), tmin (F), awnd (mph)
    """
    with open(path, "rb") as fh:
        first = fh.readline().decode("utf-8", "ignore") if not path.endswith(".gz") else ""
    if path.endswith(".gz") or first.startswith("USW"):
        return _load_ghcn_by_station(path)
    if "latitude" in first.lower() or first.lower().startswith("time"):
        return _load_open_meteo(path)
    w = pd.read_csv(path)
    w.columns = [c.strip().upper() for c in w.columns]
    w["date"] = pd.to_datetime(w["DATE"]).dt.normalize()
    keep = ["date"] + [c for c in ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND", "WSF2"]
                       if c in w.columns]
    w = w[keep].rename(columns=str.lower)
    for c in w.columns.drop("date"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    return w.drop_duplicates("date")


def _load_ghcn_by_station(path):
    cols = ["id", "ymd", "element", "value", "mflag", "qflag", "sflag", "obstime"]
    g = pd.read_csv(path, header=None, names=cols, dtype={"ymd": str, "qflag": str})
    g = g[g["element"].isin(["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND"])]
    g = g[g["qflag"].isna()]                         # drop values that failed NOAA quality checks
    g["date"] = pd.to_datetime(g["ymd"], format="%Y%m%d")
    w = g.pivot_table(index="date", columns="element", values="value", aggfunc="first").reset_index()
    out = pd.DataFrame({"date": w["date"]})
    if "PRCP" in w: out["prcp"] = w["PRCP"] / 254.0          # tenths of mm -> inches
    if "SNOW" in w: out["snow"] = w["SNOW"] / 25.4           # mm -> inches
    if "SNWD" in w: out["snwd"] = w["SNWD"] / 25.4
    if "TMAX" in w: out["tmax"] = w["TMAX"] / 10 * 9 / 5 + 32  # tenths of C -> F
    if "TMIN" in w: out["tmin"] = w["TMIN"] / 10 * 9 / 5 + 32
    if "AWND" in w: out["awnd"] = w["AWND"] / 10 * 2.23694     # tenths of m/s -> mph
    return out


def _load_open_meteo(path):
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.lower().startswith("time,"))
    from io import StringIO
    w = pd.read_csv(StringIO("\n".join(lines[i:])))
    out = pd.DataFrame({"date": pd.to_datetime(w["time"]).dt.normalize()})
    for col in w.columns:
        c = col.lower()
        if c.startswith("precipitation_sum"): out["prcp"] = w[col]
        elif c.startswith("snowfall_sum"): out["snow"] = w[col]
        elif c.startswith("temperature_2m_max"): out["tmax"] = w[col]
        elif c.startswith("temperature_2m_min"): out["tmin"] = w[col]
        elif c.startswith("wind_speed_10m_max"): out["awnd"] = w[col]
    return out


def add_weather_flags(df):
    snow = df["snow"].fillna(0) if "snow" in df else 0
    # Snow days are kept OUT of the rain comparison so snow/slush isn't counted as "rain".
    rain = (df["prcp"] >= 0.10) & (snow == 0) & (df["tmax"] > 35)
    dry = (df["prcp"] == 0) & (snow == 0)
    df["weather_group"] = np.select(
        [df["prcp"].isna(), snow > 0, rain, dry],
        ["unknown", "snow", "rain", "dry"], default="trace_or_light")
    df["rain_intensity"] = pd.cut(df["prcp"], [-0.01, 0, 0.099, 0.49, np.inf],
                                  labels=["none", "trace/light", "moderate", "heavy"])
    df["hot_day"] = df["tmax"] >= 90
    df["cold_day"] = df["tmax"] <= 32
    df["dow"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.day_of_week >= 5
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headways", default=None, help="optional; omit if you only have travel times")
    ap.add_argument("--travel", required=True, nargs="+")
    ap.add_argument("--noaa", required=True)
    ap.add_argument("--out", default="data/merged_daily.csv")
    a = ap.parse_args()

    tt = None
    hw = None
    if a.headways:
        print("Headways:");     hw = load_many(a.headways, summarize_headways)
    files = sorted({f for pattern in a.travel for f in glob.glob(pattern)})
    if not files:
        sys.exit(f"No files matched {a.travel}")
    print(f"Travel times ({len(files)} HR files):")
    tt = summarize_travel_files(files)
    wx = load_noaa(a.noaa)

    # outer join so a day with only one measure is not silently dropped
    mbta = tt if hw is None else hw.merge(tt, on=["date", "line"], how="outer")
    merged = mbta.merge(wx, on="date", how="left", validate="m:1")
    merged = add_weather_flags(merged).sort_values(["date", "line"])

    # ---- data-quality report (useful for the "missing data" paragraph) ----
    print("\n=== Merge report ===")
    print(f"line-days: {len(merged):,}   date range: {merged['date'].min().date()} -> {merged['date'].max().date()}")
    print(f"line-days with no weather row: {merged['prcp'].isna().sum():,}")
    print("weather_group counts (line-days):")
    print(merged["weather_group"].value_counts().to_string())
    print("missing values per column:")
    print(merged.isna().sum()[lambda s: s > 0].to_string())
    merged.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")

    print("\n=== Sanity checks ===")
    dry_mean = merged.loc[merged["weather_group"] == "dry", "tt_ratio_mean"].mean()
    print(f"Mean tt_ratio_mean on dry days: {dry_mean:.4f}")
    days_per_line = merged.groupby("line")["date"].nunique().reindex(
        ["Red", "Orange", "Blue"])
    print("Unique service dates per line:")
    print(days_per_line.to_string())


if __name__ == "__main__":
    main()

"""
Day-ahead electricity price forecasting (DE-LU)
Step 1: data loading, seasonal-naive baselines, walk-forward evaluation.
Step 2: calendar + lag features, Ridge regression and gradient boosting.

Data: Energy-Charts API (Fraunhofer ISE), day-ahead prices, source data from
Bundesnetzagentur | SMARD.de (CC BY 4.0, please keep the attribution).

Requirements: pandas, numpy, matplotlib, scikit-learn, holidays
Run:          python da_price_forecast.py            (uses the cached CSV after the first run)
              python da_price_forecast.py --refresh  (downloads the data again)
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import holidays
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Global Variables
BZN = "DE-LU"                       # bidding zone: Germany + Luxembourg
DATA_START = "2023-01-01"           # starting after the 2022 energy crisis is a
                                    # deliberate choice: the price regime was different
DATA_END = pd.Timestamp.today().strftime("%Y-%m-%d")
TEST_START = "2025-01-01"           # everything before is "history", from here we evaluate
LOCAL_TZ = "Europe/Berlin"          # delivery days are defined in local time
CACHE = Path("data/day_ahead_prices_hourly.csv")
PLOT_DIR = Path("plots")


# Load data
def fetch_chunk(start: str, end: str) -> pd.Series:
    """Download one date range from the Energy-Charts API (prices in EUR/MWh)."""
    query = urllib.parse.urlencode({"bzn": BZN, "start": start, "end": end})
    url = f"https://api.energy-charts.info/price?{query}"
    request = urllib.request.Request(
        url, headers={"accept": "application/json", "User-Agent": "da-price-forecast-student-project"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    return pd.Series(pd.to_numeric(payload["price"], errors="coerce"), index=index, name="price")


def download_prices() -> pd.Series:
    """Download year by year (the public API is rate limited) and resample to hourly means.

    Note: since 1 Oct 2025 the day-ahead market uses 15-minute products. Taking the
    hourly mean makes the whole history comparable (60-min before, 4 x 15-min after).
    """
    years = range(pd.Timestamp(DATA_START).year, pd.Timestamp(DATA_END).year + 1)
    parts = []
    for year in years:
        start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(DATA_START)).strftime("%Y-%m-%d")
        end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(DATA_END)).strftime("%Y-%m-%d")
        print(f"  downloading {start} to {end} ...")
        parts.append(fetch_chunk(start, end))
        time.sleep(1.5)  # be polite to the public API
    raw = pd.concat(parts)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw.resample("1h").mean()


def load_prices(refresh: bool = False) -> pd.Series:
    """Return a complete hourly price series (UTC index), cached on disk."""
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        prices = df["price"]
    else:
        print("Downloading data from Energy-Charts ...")
        prices = download_prices()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        prices.to_frame("price").to_csv(CACHE)

    # make sure there are no silent holes in the time axis
    full_index = pd.date_range(prices.index.min(), prices.index.max(), freq="1h", tz="UTC")
    prices = prices.reindex(full_index)
    n_missing = int(prices.isna().sum())
    prices = prices.interpolate(method="time", limit=3)   # only fill short gaps
    prices = prices.dropna() if prices.isna().any() else prices
    print(f"  {len(prices):,} hourly values, {n_missing} missing before gap filling")
    return prices.tz_convert(LOCAL_TZ)


def describe(prices: pd.Series) -> None:
    print("\n--- Data summary ---")
    print(f"Range:               {prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}")
    print(f"Mean / std:          {prices.mean():.1f} / {prices.std():.1f} EUR/MWh")
    print(f"Min / max:           {prices.min():.1f} / {prices.max():.1f} EUR/MWh")
    print(f"Hours with price<0:  {(prices < 0).mean():.1%}")


def make_seasonal_naive(lag_hours: int):
    """Forecast = the price exactly `lag_hours` earlier (24 = yesterday, 168 = last week)."""
    assert lag_hours >= 24, "lags below 24 h would use information that is not yet known"

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        source_times = target_index - pd.Timedelta(hours=lag_hours)
        values = history.reindex(source_times).to_numpy()
        return pd.Series(values, index=target_index)

    return forecast


def make_hybrid_naive():
    """ Tue-Fri -> yesterday's price, Mon/Sat/Sun -> last week's price. """
    naive_day, naive_week = make_seasonal_naive(24), make_seasonal_naive(168)

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        use_week = np.isin(target_index.day_of_week, [0, 5, 6])      # Mon=0 ... Sun=6
        values = np.where(use_week,
                          naive_week(history, target_index).to_numpy(),
                          naive_day(history, target_index).to_numpy())
        return pd.Series(values, index=target_index)

    return forecast


# Features
NUMERIC_COLS = ["lag_24h", "lag_48h", "lag_168h", "mean_24h", "mean_168h", "doy_sin", "doy_cos"]
CATEGORICAL_COLS = ["hour", "dow"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ["is_holiday"]
GERMAN_HOLIDAYS = holidays.Germany(years=range(2022, 2031))   # national holidays


def build_features(prices: pd.Series) -> pd.DataFrame:
    """One row per hour of `prices.index`; every price-based feature looks back >= 24 h."""
    p = prices.asfreq("1h")                     # guarantees a gap-free hourly axis
    X = pd.DataFrame(index=p.index)

    # --- price features (all shifted by at least 24 h) ---
    X["lag_24h"] = p.shift(24)                  # same hour yesterday
    X["lag_48h"] = p.shift(48)                  # same hour two days ago
    X["lag_168h"] = p.shift(168)                # same hour last week
    X["mean_24h"] = p.shift(24).rolling(24).mean()     # average level of the last known day
    X["mean_168h"] = p.shift(24).rolling(168).mean()   # average level of the last known week

    # --- calendar features ---
    X["hour"] = p.index.hour
    X["dow"] = p.index.day_of_week                # Monday = 0
    doy = p.index.day_of_year.to_numpy()
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)    # smooth yearly seasonality
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["is_holiday"] = [
        int(d in GERMAN_HOLIDAYS or (d.month == 12 and d.day in (24, 31))) for d in p.index.date
    ]
    return X


# Models
def make_ridge(alpha: float = 10.0):
    """Regularised linear model: L2 penalty shrinks the coefficients."""
    preprocess = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(categories=[list(range(24)), list(range(7))],
                                     handle_unknown="ignore"), CATEGORICAL_COLS),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_COLS),
        ],
        remainder="passthrough",                # is_holiday (0/1) is passed on unchanged
    )
    return make_pipeline(preprocess, Ridge(alpha=alpha))


def make_hist_gb():
    """Gradient-boosted trees: can learn non-linear effects and interactions"""
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, random_state=0,
    )


def make_ml_forecaster(model_factory, retrain_every_days: int = 7, train_window_days: int = 730):
    """Wrap a scikit-learn model in the same interface as the baselines.

    The model is refitted every `retrain_every_days` days on the last `train_window_days`
    days of history. In between, it only receives fresh lag features. Refitting daily
    would be more faithful but much slower.
    """
    state = {"model": None, "last_fit_day": None}

    def forecast(history: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        day = target_index[0].normalize()

        # (re)train if necessary, using ONLY history
        if state["model"] is None or (day - state["last_fit_day"]).days >= retrain_every_days:
            train = history.iloc[-train_window_days * 24:]
            X_train = build_features(train).dropna()
            y_train = train.reindex(X_train.index)
            model = model_factory()
            model.fit(X_train[FEATURE_COLS], y_train)
            state.update(model=model, last_fit_day=day)

        # features for the target day: recent history + empty target rows
        recent = history.iloc[-24 * 21:]
        target_rows = pd.Series(np.nan, index=target_index)
        X_target = build_features(pd.concat([recent, target_rows])).reindex(target_index)
        return pd.Series(state["model"].predict(X_target[FEATURE_COLS]), index=target_index)

    return forecast


# Metrics
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Walk-Forward Evaluation
def walk_forward(prices: pd.Series, forecast_fn, test_start: str) -> pd.DataFrame:
    """Forecast every delivery day in the test period using only data from before that day.
    Returns a DataFrame with columns y_true and y_pred (one row per hour).
    """
    local_days = prices.loc[test_start:].index.normalize().unique()
    rows = []
    for i, day_start in enumerate(local_days):
        if i % 100 == 0:
            print(f"    day {i + 1} / {len(local_days)}")
        target = prices[prices.index.normalize() == day_start]    # the 23 to 25 hours of this day
        history = prices[prices.index < day_start]                # everything strictly before
        prediction = forecast_fn(history, target.index)
        rows.append(pd.DataFrame({"y_true": target, "y_pred": prediction}))
    result = pd.concat(rows).dropna()
    return result


def align_results(results: dict) -> dict:
    common = None
    for df in results.values():
        common = df.index if common is None else common.intersection(df.index)
    return {name: df.loc[common] for name, df in results.items()}


def summarize(results: dict, reference: str) -> pd.DataFrame:
    table = {
        name: {"MAE": mae(df.y_true, df.y_pred), "RMSE": rmse(df.y_true, df.y_pred)}
        for name, df in results.items()
    }
    out = pd.DataFrame(table).T
    out["MAE_gain_vs_ref_%"] = 100 * (1 - out["MAE"] / out.loc[reference, "MAE"])
    out["n_hours"] = len(next(iter(results.values())))
    return out.round(2)


def mae_by_weekday(results: dict) -> pd.DataFrame:
    table = pd.DataFrame({
        name: (df.y_true - df.y_pred).abs().groupby(df.index.day_of_week).mean()
        for name, df in results.items()
    }).round(1)
    table.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return table


# Plots
def plot_example_week(results: dict, week_start: str, filename: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    first = next(iter(results.values()))
    start = pd.Timestamp(week_start, tz=LOCAL_TZ)
    window = first.loc[start:start + pd.Timedelta(days=7)]
    ax.plot(window.index, window.y_true, color="black", linewidth=2, label="actual")
    for name, df in results.items():
        w = df.loc[start:start + pd.Timedelta(days=7)]
        ax.plot(w.index, w.y_pred, linewidth=1, label=name)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Day-ahead price {BZN}: actual vs. baselines, week of {week_start}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)


def plot_error_by_hour(results: dict) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, df in results.items():
        by_hour = (df.y_true - df.y_pred).abs().groupby(df.index.hour).mean()
        ax.plot(by_hour.index, by_hour.values, marker="o", label=name)
    ax.set_xlabel("hour of day (local time)")
    ax.set_ylabel("MAE in EUR/MWh")
    ax.set_title("Where do the baselines make their errors?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "error_by_hour.png", dpi=150)
    plt.close(fig)


def main() -> None:
    prices = load_prices(refresh="--refresh" in sys.argv)
    describe(prices)

    reference = "naive_24h (yesterday)"
    models = {
        # Baselines
        reference: make_seasonal_naive(24),
        "naive_168h (last week)": make_seasonal_naive(168),
        "naive_hybrid (Tue-Fri 24h, else 168h)": make_hybrid_naive(),
        # Models
        "ridge": make_ml_forecaster(lambda: make_ridge(alpha=10.0)),
        "hist_gb": make_ml_forecaster(make_hist_gb),
    }

    results = {}
    for name, forecast_fn in models.items():
        print(f"Evaluating {name} ...")
        results[name] = walk_forward(prices, forecast_fn, TEST_START)
    results = align_results(results)

    print("\n--- Results on the test period ---")
    print(summarize(results, reference).to_string())
    print("\n--- MAE by weekday ---")
    print(mae_by_weekday(results).to_string())

    shown = {k: results[k] for k in (reference, "ridge", "hist_gb")}
    plot_example_week(shown, week_start="2025-02-03", filename="example_week_winter.png")
    plot_example_week(shown, week_start="2025-06-02", filename="example_week_summer.png")
    plot_error_by_hour(results)
    print(f"\nPlots saved in {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()

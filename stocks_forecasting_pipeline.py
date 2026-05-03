"""
╔══════════════════════════════════════════════════════════════════╗
║        S&P 500 Stock Forecasting Pipeline                       ║
║  Bronze → Silver → Gold → Model Training → Evaluation          ║
╚══════════════════════════════════════════════════════════════════╝

STAGES:
  1. Data Ingestion          (Bronze)
  2. Data Cleaning           (Silver)
  3. Feature Engineering     (Gold)
  4. EDA & Visualization
  5. Model Training          (XGBoost)
  6. Evaluation & Reporting
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS & GLOBAL CONFIG
# ─────────────────────────────────────────────
import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Visual style (dark theme matching the original notebooks)
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#444",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})
COLORS = ["#00d4ff","#ff6b6b","#ffd93d","#6bcb77","#c77dff",
          "#f4845f","#48cae4","#e9c46a","#2a9d8f","#e76f51"]

# ── Output directories
for d in ["bronze", "silver", "gold", "models", "reports"]:
    os.makedirs(d, exist_ok=True)

# ── Target stock
TARGET_STOCK = "AAPL"
TEST_RATIO   = 0.20
LAG_PERIODS  = [1, 2, 3]
RANDOM_STATE = 42


# ══════════════════════════════════════════════
# STAGE 1 — DATA INGESTION  (Bronze)
# ══════════════════════════════════════════════
def stage1_ingest(csv_path: str) -> pd.DataFrame:
    """
    Load raw CSV  →  minimal type casting  →  save bronze parquet.
    Accepts either 'all_stocks_5yr.csv' format  OR  yfinance AAPL format.
    """
    print("\n" + "="*60)
    print("  STAGE 1 · Data Ingestion (Bronze)")
    print("="*60)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    # ── Normalise column names ──────────────────────────────────
    rename_map = {
        "ticker": "Name", "symbol": "Name",        # yfinance variant
        "adj close": "adj_close",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # If single-stock CSV (no Name column) add it
    if "name" not in df.columns and "Name" not in df.columns:
        df["Name"] = TARGET_STOCK

    df.rename(columns={"name": "Name"}, inplace=True)   # unify casing

    df["date"] = pd.to_datetime(df["date"])

    print(f"  ✅ Loaded  {len(df):,} rows  ·  {df['Name'].nunique()} stocks")
    print(f"     Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"     Columns   : {list(df.columns)}")

    df.to_parquet("bronze/raw_stocks.parquet", index=False)
    print("  💾 Saved → bronze/raw_stocks.parquet")
    return df


# ══════════════════════════════════════════════
# STAGE 2 — DATA CLEANING  (Silver)
# ══════════════════════════════════════════════
def stage2_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Null analysis & forward-fill
    • Duplicate removal
    • Outlier detection (IQR on volume)
    • Business-day validation
    """
    print("\n" + "="*60)
    print("  STAGE 2 · Data Cleaning (Silver)")
    print("="*60)

    df = df.copy()
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)

    # ── 2.1  Null analysis ─────────────────────────────────────
    null_counts = df.isnull().sum()
    print("\n  Null counts:\n", null_counts[null_counts > 0].to_string() or "  (none)")

    # Forward-fill numeric price/volume columns per stock
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby("Name")[col].transform(
                lambda s: s.ffill().bfill()
            )

    # Drop rows still missing critical fields
    critical = [c for c in ["date","open","high","low","close","volume","Name"]
                if c in df.columns]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    print(f"\n  Dropped {before - len(df)} rows with missing critical fields")

    # ── 2.2  Duplicates ────────────────────────────────────────
    dup_count = df.duplicated(subset=["date","Name"]).sum()
    print(f"  Duplicate (date,Name) pairs: {dup_count}")
    df.drop_duplicates(subset=["date","Name"], inplace=True)

    # ── 2.3  Outliers (volume IQR) ─────────────────────────────
    q25, q75 = df["volume"].quantile([0.25, 0.75])
    iqr       = q75 - q25
    lo, hi    = q25 - 1.5*iqr, q75 + 1.5*iqr
    outliers  = ((df["volume"] < lo) | (df["volume"] > hi)).sum()
    pct       = outliers / len(df) * 100
    print(f"\n  Volume IQR  lower={lo:,.0f}  upper={hi:,.0f}")
    print(f"  Outliers   : {outliers:,}  ({pct:.2f}%)  — kept (informative)")

    # ── 2.4  Business-day check ────────────────────────────────
    weekend_rows = df[df["date"].dt.dayofweek >= 5]
    print(f"  Weekend rows: {len(weekend_rows)} — removed")
    df = df[df["date"].dt.dayofweek < 5]

    # ── 2.5  Type enforcement ──────────────────────────────────
    df["Name"] = df["Name"].str.upper().str.strip()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df = df.reset_index(drop=True)
    df.to_parquet("silver/cleaned_stocks.parquet", index=False)
    print(f"\n  ✅ Clean dataset: {len(df):,} rows")
    print("  💾 Saved → silver/cleaned_stocks.parquet")
    return df


# ══════════════════════════════════════════════
# STAGE 3 — FEATURE ENGINEERING  (Gold)
# ══════════════════════════════════════════════
def stage3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar, lag, rolling, technical, and target features.
    """
    print("\n" + "="*60)
    print("  STAGE 3 · Feature Engineering (Gold)")
    print("="*60)

    df = df.copy().sort_values(["Name", "date"]).reset_index(drop=True)

    # ── 3.1  Calendar features ─────────────────────────────────
    dt = df["date"].dt
    df["year"]           = dt.year
    df["month"]          = dt.month
    df["day"]            = dt.day
    df["dayofweek"]      = dt.dayofweek
    df["quarter"]        = dt.quarter
    df["weekofyear"]     = dt.isocalendar().week.astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"]   = dt.is_month_end.astype(int)
    df["is_quarter_start"] = ((df["month"].isin([1,4,7,10])) & (df["day"] == 1)).astype(int)
    df["is_quarter_end"]   = (
        ((df["month"] == 3)  & (df["day"] == 31)) |
        ((df["month"] == 6)  & (df["day"] == 30)) |
        ((df["month"] == 9)  & (df["day"] == 30)) |
        ((df["month"] == 12) & (df["day"] == 31))
    ).astype(int)

    # ── 3.2  US Holidays ──────────────────────────────────────
    try:
        import holidays
        years = range(df["year"].min(), df["year"].max() + 1)
        us_hols = set(holidays.US(years=list(years)).keys())
        df["is_holiday"] = df["date"].dt.date.astype("datetime64[ns]") \
                             .dt.normalize().isin(
                                 pd.to_datetime(list(us_hols))
                             ).astype(int)
    except ImportError:
        df["is_holiday"] = 0
        print("  ⚠️  holidays package not found — is_holiday set to 0")

    # ── 3.3  Per-stock lag & rolling features ─────────────────
    grp = df.groupby("Name")["close"]

    for lag in LAG_PERIODS:
        df[f"lag_{lag}"] = grp.shift(lag)

    for win in [5, 10, 20]:
        df[f"sma_{win}"]    = grp.transform(lambda s: s.rolling(win).mean())
        df[f"std_{win}"]    = grp.transform(lambda s: s.rolling(win).std())

    # EMA
    df["ema_12"] = grp.transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema_26"] = grp.transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"]   = df["ema_12"] - df["ema_26"]

    # Daily return & volatility
    df["daily_return"] = grp.pct_change()
    df["volatility_5"] = df.groupby("Name")["daily_return"] \
                           .transform(lambda s: s.rolling(5).std())

    # Price range
    df["price_range"] = df["high"] - df["low"]
    df["days_since_prev"] = df.groupby("Name")["date"] \
                              .transform(lambda s: s.diff().dt.days)

    # ── 3.4  Target variable (two variants) ───────────────────
    df["target_next_close"]  = grp.shift(-1)                     # raw price
    df["target_next_return"] = grp.pct_change().shift(-1)        # % return

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet("gold/stocks_features.parquet", index=False)
    print(f"  ✅ Feature set: {df.shape[1]} columns  ·  {len(df):,} rows")
    print("  💾 Saved → gold/stocks_features.parquet")
    return df


# ══════════════════════════════════════════════
# STAGE 4 — EDA  (5 charts → gold/)
# ══════════════════════════════════════════════
def stage4_eda(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  STAGE 4 · Exploratory Data Analysis")
    print("="*60)

    # ── Plot 1: Market volume + avg price over time ────────────
    monthly = (
        df.assign(ym=df["date"].dt.to_period("M"))
          .groupby("ym")
          .agg(total_volume=("volume","sum"), avg_close=("close","mean"))
          .reset_index()
    )
    monthly["date"] = monthly["ym"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("📈 S&P 500 Market Demand Over Time",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    ax1 = axes[0]
    ax1.fill_between(monthly["date"], monthly["total_volume"]/1e9, alpha=0.4, color=COLORS[0])
    ax1.plot(monthly["date"],  monthly["total_volume"]/1e9, color=COLORS[0], lw=2.5)
    ax1.set_ylabel("Total Volume (Billions)")
    ax1.set_title("Monthly Total Trading Volume", color="#aaa")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.fill_between(monthly["date"], monthly["avg_close"], alpha=0.3, color=COLORS[2])
    ax2.plot(monthly["date"],  monthly["avg_close"], color=COLORS[2], lw=2.5)
    ax2.set_ylabel("Average Close Price (USD)")
    ax2.set_title("Monthly Average Close Price", color="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("gold/plot1_market_overview.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  ✅ Plot 1 saved")

    # ── Plot 2: Top-10 stocks by volume ───────────────────────
    top10 = (
        df.groupby("Name")
          .agg(total_volume=("volume","sum"), avg_close=("close","mean"))
          .nlargest(10, "total_volume")
          .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("🏆 Top 10 Stocks by Trading Volume",
                 fontsize=16, fontweight="bold", color="white")

    ax = axes[0]
    bars = ax.barh(top10["Name"][::-1].values, top10["total_volume"][::-1].values/1e9,
                   color=COLORS[:10], height=0.6)
    for bar, val in zip(bars, top10["total_volume"][::-1].values/1e9):
        ax.text(val+0.2, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}B", va="center", fontsize=9, color="white")
    ax.set_xlabel("Total Volume (Billions)")
    ax.set_title("5-Year Total Volume", color="#aaa")

    ax2 = axes[1]
    ax2.scatter(top10["avg_close"], top10["total_volume"]/1e9,
                c=COLORS[:len(top10)], s=250, edgecolors="white", lw=0.8)
    for _, row in top10.iterrows():
        ax2.annotate(row["Name"],
                     xy=(row["avg_close"], row["total_volume"]/1e9),
                     xytext=(5,5), textcoords="offset points",
                     fontsize=8, color="white")
    ax2.set_xlabel("Average Close Price (USD)")
    ax2.set_ylabel("Total Volume (Billions)")
    ax2.set_title("Price vs Volume", color="#aaa")

    plt.tight_layout()
    plt.savefig("gold/plot2_top_stocks.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  ✅ Plot 2 saved")

    # ── Plot 3: Seasonal patterns ─────────────────────────────
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    dow_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    monthly_avg = df.groupby("month")["volume"].mean().reset_index()
    dow_avg     = df.groupby("dayofweek")["volume"].mean().reset_index()
    heatmap_data = df.groupby(["month","dayofweek"])["volume"].mean().unstack()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("🗓️ Seasonal Demand Patterns",
                 fontsize=16, fontweight="bold", color="white")

    ax = axes[0]
    ax.bar([month_names[m-1] for m in monthly_avg["month"]],
           monthly_avg["volume"]/1e6, color=COLORS[:12])
    ax.set_title("Avg Demand by Month", color="#aaa")
    ax.set_ylabel("Avg Volume (Millions)")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.bar([dow_names[d] for d in dow_avg["dayofweek"]],
            dow_avg["volume"]/1e6, color=COLORS[:7])
    ax2.set_title("Avg Demand by Day", color="#aaa")
    ax2.set_ylabel("Avg Volume (Millions)")

    ax3 = axes[2]
    heatmap_data.columns = [dow_names[d] for d in heatmap_data.columns]
    heatmap_data.index   = [month_names[m-1] for m in heatmap_data.index]
    sns.heatmap(heatmap_data/1e6, ax=ax3, cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label":"Avg Volume (M)"})
    ax3.set_title("Heatmap: Month × Day", color="#aaa")

    plt.tight_layout()
    plt.savefig("gold/plot3_seasonal.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  ✅ Plot 3 saved")

    # ── Plot 4: Year-over-year ─────────────────────────────────
    yoy = (
        df.groupby("year")
          .agg(total_vol=("volume","sum"), avg_price=("close","mean"))
          .reset_index()
    )
    yoy["vol_growth"] = yoy["total_vol"].pct_change() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("📅 Year-over-Year Market Analysis",
                 fontsize=15, fontweight="bold", color="white")

    ax = axes[0]
    bars = ax.bar(yoy["year"].astype(str), yoy["total_vol"]/1e9,
                  color=COLORS[:len(yoy)], width=0.5)
    for bar, val in zip(bars, yoy["total_vol"]/1e9):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
                f"{val:.0f}B", ha="center", fontsize=10, color="white",
                fontweight="bold")
    ax.set_ylabel("Total Volume (Billions)")
    ax.set_title("Annual Trading Volume", color="#aaa")

    ax2 = axes[1]
    growth = yoy.dropna(subset=["vol_growth"])
    gc = ["#6bcb77" if v >= 0 else "#ff6b6b" for v in growth["vol_growth"]]
    ax2.bar(growth["year"].astype(str), growth["vol_growth"],
            color=gc, width=0.5)
    ax2.axhline(0, color="white", lw=0.8, ls="--")
    ax2.set_ylabel("YoY Growth (%)")
    ax2.set_title("Year-over-Year Volume Growth", color="#aaa")

    plt.tight_layout()
    plt.savefig("gold/plot4_yoy.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print("  ✅ Plot 4 saved")
    print(f"\n  EDA complete — charts in gold/")


# ══════════════════════════════════════════════
# STAGE 5 — MODEL TRAINING  (XGBoost)
# ══════════════════════════════════════════════
def stage5_train(df: pd.DataFrame, stock: str = TARGET_STOCK):
    """
    Train on a single stock's data using time-series cross-validation.
    Returns model, X_test, y_test, predictions.
    """
    print("\n" + "="*60)
    print(f"  STAGE 5 · Model Training  [{stock}]")
    print("="*60)

    # ── Filter stock ──────────────────────────────────────────
    sdf = df[df["Name"] == stock].copy().sort_values("date")
    if len(sdf) < 100:
        raise ValueError(f"Too few rows for {stock}: {len(sdf)}")

    # ── Feature / target split ─────────────────────────────────
    drop_cols = ["date", "Name", "target_next_close", "target_next_return",
                 "adj_close"]
    feature_cols = [c for c in sdf.columns if c not in drop_cols]
    sdf_clean = sdf.select_dtypes(include=[np.number])
    feature_cols = [c for c in feature_cols if c in sdf_clean.columns]

    X = sdf[feature_cols]
    y = sdf["target_next_return"]   # predict % return (more robust)

    # ── Train / test split (time-based) ───────────────────────
    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"\n  Train  : {len(X_train):,} rows  |  Test : {len(X_test):,} rows")
    print(f"  Features: {len(feature_cols)}")

    # ── TimeSeriesSplit cross-validation ──────────────────────
    tscv   = TimeSeriesSplit(n_splits=5)
    cv_mae = []
    print("\n  Cross-validation (TimeSeriesSplit, k=5):")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = XGBRegressor(n_estimators=100, learning_rate=0.05,
                         max_depth=5, random_state=RANDOM_STATE,
                         objective="reg:squarederror", verbosity=0)
        m.fit(Xtr, ytr)
        cv_mae.append(mean_absolute_error(yval, m.predict(Xval)))
        print(f"    Fold {fold+1}  MAE={cv_mae[-1]:.6f}")
    print(f"  Mean CV MAE: {np.mean(cv_mae):.6f}")

    # ── Final model on full train set ─────────────────────────
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)

    # ── Save model ────────────────────────────────────────────
    model.save_model(f"models/{stock}_xgb_model.json")
    print(f"\n  💾 Model saved → models/{stock}_xgb_model.json")

    return model, X_train, X_test, y_train, y_test, sdf, feature_cols


# ══════════════════════════════════════════════
# STAGE 6 — EVALUATION & REPORTING
# ══════════════════════════════════════════════
def stage6_evaluate(model, X_train, X_test, y_train, y_test,
                    sdf, feature_cols, stock=TARGET_STOCK):
    print("\n" + "="*60)
    print(f"  STAGE 6 · Evaluation & Reporting  [{stock}]")
    print("="*60)

    # ── Predict % returns, then convert to prices ─────────────
    split = len(X_train)
    test_rows = sdf.iloc[split:]

    pred_returns  = model.predict(X_test)
    actual_prices = test_rows["close"].shift(-1).values[:-1]
    pred_prices   = test_rows["close"].values[:-1] * (1 + pred_returns[:-1])
    plot_dates    = test_rows["date"].values[:-1]

    # ── Metrics ───────────────────────────────────────────────
    mae  = mean_absolute_error(actual_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    r2   = r2_score(actual_prices, pred_prices)

    print(f"\n  MAE  : ${mae:.2f}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ── Plot A: Forecast vs Actual ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_dates, actual_prices, label="Actual Price", color=COLORS[0], lw=2)
    ax.plot(plot_dates, pred_prices,   label="Predicted Price", color=COLORS[1],
            alpha=0.8, lw=1.5, ls="--")
    ax.set_title(f"{stock} Stock Price Forecast — XGBoost (Returns Method)",
                 color="white", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=11)
    ax.text(0.02, 0.95, f"MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}",
            transform=ax.transAxes, color="white", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#1a1d2e", alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"reports/{stock}_forecast.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"  ✅ Forecast chart → reports/{stock}_forecast.png")

    # ── Plot B: Feature Importance ────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    top15 = fi.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    top15.plot(kind="barh", ax=ax, color=COLORS[0], edgecolor="none")
    ax.set_title(f"Top 15 Feature Importances — {stock}", color="white", fontsize=13)
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f"reports/{stock}_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"  ✅ Feature importance → reports/{stock}_feature_importance.png")

    # ── Plot C: Residuals ─────────────────────────────────────
    residuals = actual_prices - pred_prices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Analysis — {stock}", color="white", fontsize=14)

    axes[0].scatter(pred_prices, residuals, alpha=0.5, color=COLORS[2], s=15)
    axes[0].axhline(0, color="white", lw=1, ls="--")
    axes[0].set_xlabel("Predicted Price"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted", color="#aaa")

    axes[1].hist(residuals, bins=40, color=COLORS[3], edgecolor="none", alpha=0.8)
    axes[1].axvline(0, color="white", lw=1.5, ls="--")
    axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution", color="#aaa")

    plt.tight_layout()
    plt.savefig(f"reports/{stock}_residuals.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"  ✅ Residuals chart → reports/{stock}_residuals.png")

    # ── Summary report ────────────────────────────────────────
    report = f"""
╔══════════════════════════════════════════════╗
║   PIPELINE SUMMARY REPORT                   ║
╠══════════════════════════════════════════════╣
║  Stock         : {stock:<28}║
║  Test samples  : {len(actual_prices):<28}║
║  MAE           : ${mae:<27.2f}║
║  RMSE          : ${rmse:<27.2f}║
║  R²            : {r2:<28.4f}║
╠══════════════════════════════════════════════╣
║  Top 5 Features:                            ║
"""
    for feat, imp in fi.nlargest(5).items():
        line = f"    {feat:<24}  {imp:.4f}"
        report += f"║  {line:<44}║\n"
    report += "╚══════════════════════════════════════════════╝\n"
    print(report)

    with open(f"reports/{stock}_summary.txt", "w") as f:
        f.write(report)
    print(f"  💾 Summary → reports/{stock}_summary.txt")

    return {"mae": mae, "rmse": rmse, "r2": r2}


# ══════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════
def run_pipeline(csv_path: str, stock: str = TARGET_STOCK):
    """
    End-to-end pipeline entry point.

    Args:
        csv_path : Path to raw CSV  (e.g. 'all_stocks_5yr.csv')
        stock    : Ticker to model  (default: AAPL)
    """
    print("""
╔══════════════════════════════════════════════════════════╗
║   S&P 500 STOCK FORECASTING PIPELINE  v1.0              ║
║   Bronze → Silver → Gold → Train → Evaluate             ║
╚══════════════════════════════════════════════════════════╝
""")
    t0 = datetime.now()

    raw_df       = stage1_ingest(csv_path)
    clean_df     = stage2_clean(raw_df)
    feature_df   = stage3_features(clean_df)
    stage4_eda(feature_df)
    model, X_tr, X_te, y_tr, y_te, sdf, fcols = stage5_train(feature_df, stock)
    metrics      = stage6_evaluate(model, X_tr, X_te, y_tr, y_te, sdf, fcols, stock)

    elapsed = (datetime.now() - t0).seconds
    print(f"\n  🏁  Pipeline complete in {elapsed}s")
    print("  Output directories: bronze/  silver/  gold/  models/  reports/")
    return metrics


# ══════════════════════════════════════════════
# DEMO — runs with synthetic AAPL data if no
#         CSV is present (for testing)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "all_stocks_5yr.csv"
    stk      = sys.argv[2] if len(sys.argv) > 2 else TARGET_STOCK

    if not os.path.exists(csv_file):
        print(f"\n  ⚠️  '{csv_file}' not found — generating synthetic AAPL data …")

        np.random.seed(42)
        n   = 1_500
        dates = pd.date_range("2013-01-02", periods=n, freq="B")   # business days
        close = 100 + np.cumsum(np.random.randn(n) * 0.8)
        synth = pd.DataFrame({
            "date"   : dates,
            "open"   : close * (1 + np.random.randn(n)*0.003),
            "high"   : close * (1 + np.abs(np.random.randn(n)*0.005)),
            "low"    : close * (1 - np.abs(np.random.randn(n)*0.005)),
            "close"  : close,
            "volume" : np.random.randint(5_000_000, 50_000_000, n),
            "Name"   : "AAPL",
        })
        csv_file = "all_stocks_5yr.csv"
        synth.to_csv(csv_file, index=False)
        print(f"  Synthetic data saved → {csv_file}")
        stk = "AAPL"

    run_pipeline(csv_file, stk)

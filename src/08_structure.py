```python
import os
import math
from datetime import datetime, timezone

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)

GREEKS_FILE = os.path.join(
    ANALYSIS_DIR,
    "options_greeks.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

SEARCH_FILE = os.path.join(
    ANALYSIS_DIR,
    "option_search.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv"
)


# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    print(f"[08 STRUCTURE] {now} | {message}")


# ============================================================
# HELPERS
# ============================================================

def first_existing(df, columns, default=np.nan):
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def numeric_series(df, column, default=np.nan):
    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def format_price(value):
    value = safe_float(value)

    if pd.isna(value):
        return "UNAVAILABLE"

    return f"${value:,.2f}"


def format_number(value):
    value = safe_float(value)

    if pd.isna(value):
        return "UNAVAILABLE"

    return f"{value:,.0f}"


def format_pct(value):
    value = safe_float(value)

    if pd.isna(value):
        return "UNAVAILABLE"

    return f"{value:.2f}%"


def normalize_option_type(value):
    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    if text in ("CALL", "C"):
        return "CALL"

    if text in ("PUT", "P"):
        return "PUT"

    return text


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def prepare_dataframe(df):

    aliases = {
        "ticker": [
            "ticker",
            "Ticker",
            "symbol",
            "Symbol"
        ],
        "option_type": [
            "option_type",
            "OptionType",
            "type",
            "Type",
            "contract_type"
        ],
        "strike": [
            "strike",
            "Strike"
        ],
        "expiration": [
            "expiration",
            "Expiration",
            "expiry",
            "Expiry"
        ],
        "dte": [
            "dte",
            "DTE"
        ],
        "current_price": [
            "current_price",
            "Current Price",
            "underlying_price",
            "Underlying Price",
            "spot",
            "Spot"
        ],
        "volume": [
            "volume",
            "Volume"
        ],
        "open_interest": [
            "open_interest",
            "Open Interest",
            "oi",
            "OI"
        ],
        "premium": [
            "premium",
            "Premium",
            "estimated_traded_premium",
            "Estimated Traded Premium"
        ],
        "gex": [
            "gex",
            "GEX"
        ],
        "iv": [
            "iv",
            "IV",
            "implied_volatility"
        ],
        "delta": [
            "delta",
            "Delta"
        ],
        "gamma": [
            "gamma",
            "Gamma"
        ],
        "flow_score": [
            "flow_score",
            "Flow Score"
        ]
    }

    result = pd.DataFrame(index=df.index)

    for target, candidates in aliases.items():
        result[target] = first_existing(
            df,
            candidates
        )

    result["ticker"] = (
        result["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["option_type"] = (
        result["option_type"]
        .apply(normalize_option_type)
    )

    for col in [
        "strike",
        "dte",
        "current_price",
        "volume",
        "open_interest",
        "premium",
        "gex",
        "iv",
        "delta",
        "gamma",
        "flow_score"
    ]:
        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        )

    return result


# ============================================================
# WALL CALCULATION
# ============================================================

def calculate_wall(group, option_type):

    data = group[
        group["option_type"] == option_type
    ].copy()

    data = data.dropna(
        subset=["strike"]
    )

    if data.empty:
        return np.nan

    # Priority:
    # 1. GEX
    # 2. Open Interest
    # 3. Volume
    #
    # This is a structural estimate,
    # not an official market-maker wall.

    data["wall_score"] = 0.0

    if data["gex"].notna().any():
        data["wall_score"] += (
            data["gex"].abs().fillna(0)
        )

    if data["open_interest"].notna().any():
        oi = data["open_interest"].fillna(0)

        if oi.max() > 0:
            data["wall_score"] += (
                oi / oi.max()
            ) * 0.5

    if data["volume"].notna().any():
        volume = data["volume"].fillna(0)

        if volume.max() > 0:
            data["wall_score"] += (
                volume / volume.max()
            ) * 0.25

    row = data.sort_values(
        "wall_score",
        ascending=False
    ).iloc[0]

    return row["strike"]


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(group, current_price):

    strikes = (
        group["strike"]
        .dropna()
        .unique()
    )

    if len(strikes) == 0 or pd.isna(current_price):
        return np.nan, np.nan

    strikes = sorted(
        float(x) for x in strikes
    )

    below = [
        x for x in strikes
        if x < current_price
    ]

    above = [
        x for x in strikes
        if x > current_price
    ]

    support = max(below) if below else np.nan
    resistance = min(above) if above else np.nan

    return support, resistance


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def classify_structure(
    current_price,
    call_wall,
    put_wall,
    net_gex
):

    if pd.isna(current_price):
        return "UNAVAILABLE"

    bullish = 0
    bearish = 0

    if not pd.isna(call_wall):
        if call_wall > current_price:
            bullish += 1

    if not pd.isna(put_wall):
        if put_wall < current_price:
            bearish += 1

    if not pd.isna(net_gex):

        if net_gex > 0:
            # Positive GEX can indicate
            # more price stabilization.
            pass

        elif net_gex < 0:
            # Negative GEX can indicate
            # greater sensitivity to moves.
            pass

    if bullish > bearish:
        return "BULLISH STRUCTURE"

    if bearish > bullish:
        return "BEARISH STRUCTURE"

    return "MIXED STRUCTURE"


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    if not os.path.exists(GREEKS_FILE):
        raise FileNotFoundError(
            f"Missing file: {GREEKS_FILE}"
        )

    if not os.path.exists(TOP20_FILE):
        raise FileNotFoundError(
            f"Missing file: {TOP20_FILE}"
        )

    log("Loading Greeks data")

    greeks = pd.read_csv(
        GREEKS_FILE
    )

    log(
        f"GREEKS ROWS : {len(greeks):,}"
    )

    log("Loading TOP20")

    top20 = pd.read_csv(
        TOP20_FILE
    )

    if "ticker" in top20.columns:
        top_tickers = (
            top20["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
            .tolist()
        )

    elif "Ticker" in top20.columns:
        top_tickers = (
            top20["Ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
            .tolist()
        )

    else:
        raise ValueError(
            "TOP20 ticker column not found"
        )

    log(
        f"TOP20 TICKERS : {len(top_tickers)}"
    )

    df = prepare_dataframe(
        greeks
    )

    df = df[
        df["ticker"].isin(top_tickers)
    ].copy()

    if df.empty:
        raise ValueError(
            "No matching TOP20 option data found"
        )

    rows = []

    for ticker in top_tickers:

        group = df[
            df["ticker"] == ticker
        ].copy()

        if group.empty:
            continue

        current_prices = (
            group["current_price"]
            .dropna()
        )

        if current_prices.empty:
            current_price = np.nan
        else:
            current_price = (
                current_prices.iloc[0]
            )

        call_wall = calculate_wall(
            group,
            "CALL"
        )

        put_wall = calculate_wall(
            group,
            "PUT"
        )

        support, resistance = (
            calculate_support_resistance(
                group,
                current_price
            )
        )

        call_data = group[
            group["option_type"] == "CALL"
        ].copy()

        put_data = group[
            group["option_type"] == "PUT"
        ].copy()

        call_gex = (
            call_data["gex"]
            .sum(min_count=1)
        )

        put_gex = (
            put_data["gex"]
            .sum(min_count=1)
        )

        if pd.isna(call_gex):
            call_gex = 0.0

        if pd.isna(put_gex):
            put_gex = 0.0

        net_gex = (
            call_gex + put_gex
        )

        structure = classify_structure(
            current_price,
            call_wall,
            put_wall,
            net_gex
        )

        rows.append({

            "ticker": ticker,

            "current_price": current_price,

            "call_wall": call_wall,

            "put_wall": put_wall,

            "support": support,

            "resistance": resistance,

            "call_gex": call_gex,

            "put_gex": put_gex,

            "net_gex": net_gex,

            "structure": structure,

            "data_source": "CALCULATED"

        })

        log(
            f"{ticker} | "
            f"CALL WALL {format_price(call_wall)} | "
            f"PUT WALL {format_price(put_wall)} | "
            f"NET GEX {net_gex:,.2f} | "
            f"{structure}"
        )

    output = pd.DataFrame(rows)

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 72)
    print("🔎 STEP 8 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT GREEKS ROWS : {len(greeks):,}"
    )

    print(
        f"TOP20 TICKERS     : {len(top_tickers)}"
    )

    print(
        f"STRUCTURE ROWS    : {len(output)}"
    )

    print(
        f"STRUCTURE TICKERS : "
        f"{output['ticker'].nunique()}"
    )

    print(
        "CALL WALL          : "
        + str(
            output["call_wall"]
            .notna()
            .sum()
        )
        + " VALID"
    )

    print(
        "PUT WALL           : "
        + str(
            output["put_wall"]
            .notna()
            .sum()
        )
        + " VALID"
    )

    print(
        "NET GEX            : "
        + str(
            output["net_gex"]
            .notna()
            .sum()
        )
        + " VALID"
    )

    print(
        "STRUCTURE          : OK"
    )

    print(
        f"OUTPUT FILE        : "
        f"data/analysis/structure.csv"
    )

    print("=" * 72)

    log(
        "STEP 8 STRUCTURE COMPLETE"
    )


if __name__ == "__main__":
    main()
```

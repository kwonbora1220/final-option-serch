from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

GREEKS_PATH = BASE_DIR / "data" / "analysis" / "options_greeks.csv"
TOP20_PATH = BASE_DIR / "data" / "analysis" / "top20.csv"
OUTPUT_PATH = BASE_DIR / "data" / "analysis" / "structure.csv"


# ============================================================
# HELPERS
# ============================================================

def norm_col(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(df, candidates):
    mapping = {norm_col(c): c for c in df.columns}

    for candidate in candidates:
        key = norm_col(candidate)
        if key in mapping:
            return mapping[key]

    return None


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return np.nan


# ============================================================
# LOAD
# ============================================================

def load_data():

    if not GREEKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing Greeks file: {GREEKS_PATH}"
        )

    if not TOP20_PATH.exists():
        raise FileNotFoundError(
            f"Missing TOP20 file: {TOP20_PATH}"
        )

    greeks = pd.read_csv(GREEKS_PATH)
    top20 = pd.read_csv(TOP20_PATH)

    greeks.columns = [str(c).strip() for c in greeks.columns]
    top20.columns = [str(c).strip() for c in top20.columns]

    return greeks, top20


# ============================================================
# NORMALIZE GREEKS
# ============================================================

def normalize_greeks(greeks):

    cols = {
        "ticker": find_col(
            greeks,
            ["ticker", "symbol", "underlying", "underlying_symbol"],
        ),
        "option_type": find_col(
            greeks,
            ["option_type", "type", "call_put", "cp"],
        ),
        "strike": find_col(
            greeks,
            ["strike", "strike_price"],
        ),
        "current_price": find_col(
            greeks,
            [
                "current_price",
                "underlying_price",
                "stock_price",
                "spot_price",
                "underlying_last",
            ],
        ),
        "gamma": find_col(
            greeks,
            ["gamma"],
        ),
        "open_interest": find_col(
            greeks,
            ["open_interest", "oi"],
        ),
        "volume": find_col(
            greeks,
            ["volume", "option_volume"],
        ),
        "dte": find_col(
            greeks,
            ["dte", "DTE", "days_to_expiration"],
        ),
    }

    required = [
        "ticker",
        "option_type",
        "strike",
    ]

    missing = [
        name
        for name in required
        if cols[name] is None
    ]

    if missing:
        raise ValueError(
            f"STEP 8 missing required columns: {missing}"
        )

    df = pd.DataFrame()

    df["ticker"] = (
        greeks[cols["ticker"]]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["option_type"] = (
        greeks[cols["option_type"]]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["option_type"] = df["option_type"].replace({
        "C": "CALL",
        "CALLS": "CALL",
        "P": "PUT",
        "PUTS": "PUT",
    })

    df["strike"] = numeric(
        greeks[cols["strike"]]
    )

    if cols["current_price"]:
        df["current_price"] = numeric(
            greeks[cols["current_price"]]
        )
    else:
        df["current_price"] = np.nan

    if cols["gamma"]:
        df["gamma"] = numeric(
            greeks[cols["gamma"]]
        )
    else:
        df["gamma"] = np.nan

    if cols["open_interest"]:
        df["open_interest"] = (
            numeric(
                greeks[cols["open_interest"]]
            )
            .fillna(0)
            .clip(lower=0)
        )
    else:
        df["open_interest"] = 0.0

    if cols["volume"]:
        df["volume"] = (
            numeric(
                greeks[cols["volume"]]
            )
            .fillna(0)
            .clip(lower=0)
        )
    else:
        df["volume"] = 0.0

    if cols["dte"]:
        df["dte"] = numeric(
            greeks[cols["dte"]]
        )
    else:
        df["dte"] = np.nan

    return df


# ============================================================
# STRENGTH
# ============================================================

def calculate_strength(df):

    gamma = df["gamma"].abs().fillna(0)
    oi = df["open_interest"].fillna(0)
    volume = df["volume"].fillna(0)

    gamma_oi = gamma * oi

    return (
        np.log1p(gamma_oi)
        + 0.15 * np.log1p(volume)
    )


# ============================================================
# WALL
# ============================================================

def choose_wall(df, option_type, current_price, direction):

    side = df[
        df["option_type"] == option_type
    ].copy()

    if side.empty:
        return np.nan

    if np.isfinite(current_price):

        if direction == "CALL":

            side = side[
                side["strike"] >= current_price
            ]

        else:

            side = side[
                side["strike"] <= current_price
            ]

    if side.empty:
        return np.nan

    # GEX strength
    side["gex_strength"] = (
        side["gamma"].abs().fillna(0)
        *
        side["open_interest"].fillna(0)
    )

    # Structural strength
    side["strength"] = calculate_strength(side)

    side["distance"] = (
        (
            side["strike"]
            - current_price
        ).abs()
        / current_price
        if np.isfinite(current_price)
        else 999
    )

    # 가까운 strike와 실제 포지셔닝을 함께 고려
    side["selection_score"] = (
        np.log1p(side["gex_strength"])
        +
        side["strength"]
        +
        2.0 / (1.0 + side["distance"] * 20.0)
    )

    best = (
        side
        .sort_values(
            [
                "selection_score",
                "gex_strength",
                "strength",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return safe_float(best["strike"])


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def choose_support(df, current_price):

    if not np.isfinite(current_price):
        return np.nan

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["strike"] < current_price)
    ].copy()

    if puts.empty:
        return np.nan

    puts["distance"] = (
        current_price - puts["strike"]
    ) / current_price

    puts = puts[
        puts["distance"] <= 0.20
    ]

    if puts.empty:
        return np.nan

    puts["strength"] = calculate_strength(puts)

    puts["score"] = (
        puts["strength"]
        +
        3.0 / (1.0 + puts["distance"] * 20)
    )

    return safe_float(
        puts.sort_values(
            ["score", "strength"],
            ascending=False,
        ).iloc[0]["strike"]
    )


def choose_resistance(df, current_price):

    if not np.isfinite(current_price):
        return np.nan

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["strike"] > current_price)
    ].copy()

    if calls.empty:
        return np.nan

    calls["distance"] = (
        calls["strike"] - current_price
    ) / current_price

    calls = calls[
        calls["distance"] <= 0.20
    ]

    if calls.empty:
        return np.nan

    calls["strength"] = calculate_strength(calls)

    calls["score"] = (
        calls["strength"]
        +
        3.0 / (1.0 + calls["distance"] * 20)
    )

    return safe_float(
        calls.sort_values(
            ["score", "strength"],
            ascending=False,
        ).iloc[0]["strike"]
    )


# ============================================================
# GEX
# ============================================================

def calculate_gex(df, option_type):

    side = df[
        df["option_type"] == option_type
    ].copy()

    if side.empty:
        return np.nan

    valid = side[
        side["gamma"].notna()
        &
        side["open_interest"].notna()
    ].copy()

    if valid.empty:
        return np.nan

    # 실제 GEX 데이터가 있는 경우에만 계산
    return float(
        (
            valid["gamma"]
            *
            valid["open_interest"]
        ).sum()
    )


def classify_gex(net_gex):

    if not np.isfinite(net_gex):
        return "GEX UNAVAILABLE"

    if net_gex > 0:
        return "POSITIVE GEX"

    if net_gex < 0:
        return "NEGATIVE GEX"

    return "NEUTRAL GEX"


# ============================================================
# WALL STRUCTURE
# ============================================================

def classify_wall_structure(
    current_price,
    call_wall,
    put_wall,
):

    if not np.isfinite(current_price):
        return "WALL UNAVAILABLE"

    if (
        np.isfinite(call_wall)
        and np.isfinite(put_wall)
    ):

        # 정상적인 range
        if put_wall < call_wall:

            if current_price > call_wall:
                return "BULLISH BREAKOUT"

            if current_price < put_wall:
                return "BEARISH BREAKDOWN"

            return "RANGE"

        # CALL/PUT wall이 같은 가격이면
        # RANGE WALL로 취급하되 방향성을 부여하지 않는다.
        if call_wall == put_wall:

            return "SINGLE WALL"

    if np.isfinite(call_wall):

        if current_price > call_wall:
            return "ABOVE CALL WALL"

        return "BELOW CALL WALL"

    if np.isfinite(put_wall):

        if current_price < put_wall:
            return "BELOW PUT WALL"

        return "ABOVE PUT WALL"

    return "WALL UNAVAILABLE"


# ============================================================
# PRICE LOCATION
# ============================================================

def classify_price_location(
    current_price,
    support,
    resistance,
):

    if not np.isfinite(current_price):
        return "PRICE UNAVAILABLE"

    result = []

    if np.isfinite(support):

        distance = (
            current_price - support
        ) / current_price

        if distance <= 0.02:
            result.append("NEAR SUPPORT")
        elif current_price > support:
            result.append("ABOVE SUPPORT")
        else:
            result.append("BELOW SUPPORT")

    else:
        result.append("SUPPORT UNAVAILABLE")

    if np.isfinite(resistance):

        distance = (
            resistance - current_price
        ) / current_price

        if distance <= 0.02:
            result.append("NEAR RESISTANCE")
        elif current_price < resistance:
            result.append("BELOW RESISTANCE")
        else:
            result.append("ABOVE RESISTANCE")

    else:
        result.append("RESISTANCE UNAVAILABLE")

    return " | ".join(result)


# ============================================================
# STRUCTURE
# ============================================================

def classify_structure(
    current_price,
    call_wall,
    put_wall,
    net_gex,
    support,
    resistance,
):

    wall = classify_wall_structure(
        current_price,
        call_wall,
        put_wall,
    )

    gex = classify_gex(
        net_gex
    )

    price = classify_price_location(
        current_price,
        support,
        resistance,
    )

    if wall in {
        "BULLISH BREAKOUT",
        "ABOVE CALL WALL",
    }:
        structure = "BULLISH"

    elif wall in {
        "BEARISH BREAKDOWN",
        "BELOW PUT WALL",
    }:
        structure = "BEARISH"

    else:
        # RANGE / SINGLE WALL / unavailable
        # 절대 자동으로 bullish 처리하지 않는다.
        structure = "NEUTRAL"

    return (
        structure,
        price,
        gex,
        wall,
    )


# ============================================================
# PROCESS
# ============================================================

def process_ticker(ticker, greeks):

    df = greeks[
        greeks["ticker"] == ticker
    ].copy()

    if df.empty:

        return {
            "ticker": ticker,
            "current_price": np.nan,
            "call_wall": np.nan,
            "put_wall": np.nan,
            "call_gex": np.nan,
            "put_gex": np.nan,
            "net_gex": np.nan,
            "support": np.nan,
            "resistance": np.nan,
            "structure": "NO DATA",
            "price_location": "UNAVAILABLE",
            "gex_structure": "GEX UNAVAILABLE",
            "wall_structure": "WALL UNAVAILABLE",
            "gex_valid_rows": 0,
            "data_source": "CALCULATED",
        }

    prices = df["current_price"].dropna()

    current_price = (
        float(prices.iloc[-1])
        if not prices.empty
        else np.nan
    )

    call_wall = choose_wall(
        df,
        "CALL",
        current_price,
        "CALL",
    )

    put_wall = choose_wall(
        df,
        "PUT",
        current_price,
        "PUT",
    )

    support = choose_support(
        df,
        current_price,
    )

    resistance = choose_resistance(
        df,
        current_price,
    )

    call_gex = calculate_gex(
        df,
        "CALL",
    )

    put_gex = calculate_gex(
        df,
        "PUT",
    )

    if (
        np.isfinite(call_gex)
        and np.isfinite(put_gex)
    ):
        net_gex = call_gex + put_gex
    else:
        net_gex = np.nan

    structure, price_location, gex_structure, wall_structure = (
        classify_structure(
            current_price,
            call_wall,
            put_wall,
            net_gex,
            support,
            resistance,
        )
    )

    valid_gex = int(
        df["gamma"].notna().sum()
    )

    return {
        "ticker": ticker,
        "current_price": current_price,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_gex": call_gex,
        "put_gex": put_gex,
        "net_gex": net_gex,
        "support": support,
        "resistance": resistance,
        "structure": structure,
        "price_location": price_location,
        "gex_structure": gex_structure,
        "wall_structure": wall_structure,
        "gex_valid_rows": valid_gex,
        "data_source": "CALCULATED",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("============================================================")
    print("STEP 8 - STRUCTURE ANALYSIS")
    print("============================================================")

    greeks_raw, top20 = load_data()

    greeks = normalize_greeks(
        greeks_raw
    )

    ticker_col = find_col(
        top20,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
        ],
    )

    if ticker_col is None:
        raise ValueError(
            "TOP20 ticker column not found"
        )

    tickers = (
        top20[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    tickers = [
        ticker
        for ticker in tickers
        if ticker
    ][:20]

    rows = []

    for ticker in tickers:

        row = process_ticker(
            ticker,
            greeks,
        )

        rows.append(row)

        print(
            f"[08 STRUCTURE] {ticker} | "
            f"STRUCTURE {row['structure']} | "
            f"WALL {row['wall_structure']} | "
            f"GEX {row['gex_structure']}"
        )

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("STEP 8 OUTPUT CHECK")
    print(f"ROWS              : {len(result)}")
    print(
        f"UNIQUE TICKERS    : "
        f"{result['ticker'].nunique()}"
    )
    print(
        f"STRUCTURES        : "
        f"{result['structure'].value_counts(dropna=False).to_dict()}"
    )
    print(
        f"WALL STRUCTURES   : "
        f"{result['wall_structure'].value_counts(dropna=False).to_dict()}"
    )
    print(
        f"GEX STRUCTURES    : "
        f"{result['gex_structure'].value_counts(dropna=False).to_dict()}"
    )
    print()
    print(f"OUTPUT FILE : {OUTPUT_PATH}")
    print("STEP 8 OUTPUT : OK")


if __name__ == "__main__":
    main()

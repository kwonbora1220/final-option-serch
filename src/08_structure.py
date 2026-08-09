from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

GREEKS_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "options_greeks.csv"
)

TOP20_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "top20.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "structure.csv"
)


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def find_col(df, names):

    normalized = {
        str(c)
        .strip()
        .lower()
        .replace(" ", "_"): c
        for c in df.columns
    }

    for name in names:

        key = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def normalize_option_type(series):

    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
        .replace(
            {
                "C": "CALL",
                "CALLS": "CALL",
                "P": "PUT",
                "PUTS": "PUT",
            }
        )
    )


def safe(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def calculate_strength(df):

    gamma = df["gamma"].abs().fillna(0)
    oi = df["open_interest"].fillna(0)
    volume = df["volume"].fillna(0)

    return (
        np.log1p(gamma * oi)
        +
        0.25 * np.log1p(volume)
    )


def choose_wall(
    df,
    option_type,
    price,
):

    side = df[
        df["option_type"] == option_type
    ].copy()

    if side.empty:
        return np.nan

    if option_type == "CALL":

        side = side[
            side["strike"] >= price
        ]

    else:

        side = side[
            side["strike"] <= price
        ]

    if side.empty:
        return np.nan

    side["distance"] = (
        (side["strike"] - price).abs()
        / price
    )

    # 너무 먼 strike가 wall이 되는 것을 방지
    side = side[
        side["distance"] <= 0.20
    ].copy()

    if side.empty:
        return np.nan

    side["strength"] = (
        calculate_strength(side)
    )

    side["selection"] = (
        side["strength"]
        +
        3.0
        /
        (
            1.0
            +
            side["distance"] * 20
        )
    )

    best = (
        side
        .sort_values(
            [
                "selection",
                "strength",
                "open_interest",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return safe(best["strike"])


def calculate_gex(
    df,
    option_type,
    price,
):

    side = df[
        df["option_type"] == option_type
    ].copy()

    side = side[
        side["gamma"].notna()
        &
        side["open_interest"].notna()
        &
        side["gamma"].ne(0)
        &
        side["open_interest"].gt(0)
    ]

    if side.empty:
        return np.nan

    # 표준적인 단순 GEX proxy
    #
    # CALL : +
    # PUT  : -
    #
    # contract multiplier = 100
    #
    # spot^2 scaling을 사용하되,
    # 너무 큰 숫자가 의사결정에 직접 영향을 주지 않도록
    # 구조에서는 부호/상대값 중심으로 사용한다.

    raw = (
        side["gamma"]
        *
        side["open_interest"]
        *
        100.0
        *
        price
        *
        price
    )

    value = raw.sum()

    if option_type == "PUT":
        value = -value

    if not np.isfinite(value):
        return np.nan

    return float(value)


def choose_support(
    df,
    price,
):

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["strike"] < price)
    ].copy()

    if puts.empty:
        return np.nan

    puts["distance"] = (
        price - puts["strike"]
    ) / price

    puts = puts[
        puts["distance"] <= 0.15
    ].copy()

    if puts.empty:
        return np.nan

    puts["strength"] = (
        calculate_strength(puts)
    )

    puts["score"] = (
        puts["strength"]
        +
        3.0
        /
        (
            1
            +
            puts["distance"] * 20
        )
    )

    return safe(
        puts
        .sort_values(
            ["score", "strength"],
            ascending=False,
        )
        .iloc[0]["strike"]
    )


def choose_resistance(
    df,
    price,
):

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["strike"] > price)
    ].copy()

    if calls.empty:
        return np.nan

    calls["distance"] = (
        calls["strike"] - price
    ) / price

    calls = calls[
        calls["distance"] <= 0.15
    ].copy()

    if calls.empty:
        return np.nan

    calls["strength"] = (
        calculate_strength(calls)
    )

    calls["score"] = (
        calls["strength"]
        +
        3.0
        /
        (
            1
            +
            calls["distance"] * 20
        )
    )

    return safe(
        calls
        .sort_values(
            ["score", "strength"],
            ascending=False,
        )
        .iloc[0]["strike"]
    )


def wall_structure(
    price,
    call_wall,
    put_wall,
):

    if not np.isfinite(price):
        return "WALL UNAVAILABLE"

    if (
        np.isfinite(call_wall)
        and np.isfinite(put_wall)
    ):

        if put_wall < call_wall:

            if price > call_wall:
                return "BULLISH BREAKOUT"

            if price < put_wall:
                return "BEARISH BREAKDOWN"

            return "RANGE"

        if put_wall == call_wall:
            return "SINGLE WALL"

    if np.isfinite(call_wall):

        if price > call_wall:
            return "ABOVE CALL WALL"

        return "BELOW CALL WALL"

    if np.isfinite(put_wall):

        if price < put_wall:
            return "BELOW PUT WALL"

        return "ABOVE PUT WALL"

    return "WALL UNAVAILABLE"


def price_location(
    price,
    support,
    resistance,
):

    result = []

    if np.isfinite(support):

        d = (
            price - support
        ) / price

        if 0 <= d <= 0.02:
            result.append("NEAR SUPPORT")

        elif price > support:
            result.append("ABOVE SUPPORT")

        else:
            result.append("BELOW SUPPORT")

    else:

        result.append(
            "SUPPORT UNAVAILABLE"
        )

    if np.isfinite(resistance):

        d = (
            resistance - price
        ) / price

        if 0 <= d <= 0.02:
            result.append("NEAR RESISTANCE")

        elif price < resistance:
            result.append(
                "BELOW RESISTANCE"
            )

        else:
            result.append(
                "ABOVE RESISTANCE"
            )

    else:

        result.append(
            "RESISTANCE UNAVAILABLE"
        )

    return " | ".join(result)


def classify_structure(
    price,
    call_wall,
    put_wall,
    net_gex,
):

    wall = wall_structure(
        price,
        call_wall,
        put_wall,
    )

    # 벽 자체로 방향성이 확인되는 경우
    if wall in {
        "BULLISH BREAKOUT",
        "ABOVE CALL WALL",
    }:
        return "BULLISH"

    if wall in {
        "BEARISH BREAKDOWN",
        "BELOW PUT WALL",
    }:
        return "BEARISH"

    # GEX가 실제로 계산된 경우만 사용
    if np.isfinite(net_gex):

        if net_gex > 0:
            return "POSITIVE GEX STRUCTURE"

        if net_gex < 0:
            return "NEGATIVE GEX STRUCTURE"

    return "NEUTRAL"


def main():

    if not GREEKS_FILE.exists():
        raise FileNotFoundError(
            GREEKS_FILE
        )

    if not TOP20_FILE.exists():
        raise FileNotFoundError(
            TOP20_FILE
        )

    greeks = pd.read_csv(
        GREEKS_FILE
    )

    top20 = pd.read_csv(
        TOP20_FILE
    )

    ticker_col = find_col(
        top20,
        ["ticker", "symbol"],
    )

    if ticker_col is None:
        raise RuntimeError(
            "TOP20 ticker column missing"
        )

    top_tickers = (
        top20[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    symbol_col = find_col(
        greeks,
        ["symbol", "ticker"],
    )

    type_col = find_col(
        greeks,
        ["option_type", "type"],
    )

    strike_col = find_col(
        greeks,
        ["strike", "strike_price"],
    )

    price_col = find_col(
        greeks,
        [
            "underlying_price",
            "current_price",
            "spot_price",
        ],
    )

    gamma_col = find_col(
        greeks,
        ["gamma"],
    )

    oi_col = find_col(
        greeks,
        ["openInterest", "open_interest", "oi"],
    )

    volume_col = find_col(
        greeks,
        ["volume"],
    )

    required = {
        "symbol": symbol_col,
        "option_type": type_col,
        "strike": strike_col,
        "price": price_col,
        "gamma": gamma_col,
        "open_interest": oi_col,
        "volume": volume_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:
        raise RuntimeError(
            "STEP 8 missing columns: "
            + ", ".join(missing)
        )

    df = pd.DataFrame()

    df["ticker"] = (
        greeks[symbol_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["option_type"] = (
        normalize_option_type(
            greeks[type_col]
        )
    )

    df["strike"] = numeric(
        greeks[strike_col]
    )

    df["current_price"] = numeric(
        greeks[price_col]
    )

    df["gamma"] = numeric(
        greeks[gamma_col]
    )

    df["open_interest"] = (
        numeric(
            greeks[oi_col]
        )
        .fillna(0)
        .clip(lower=0)
    )

    df["volume"] = (
        numeric(
            greeks[volume_col]
        )
        .fillna(0)
        .clip(lower=0)
    )

    df = df[
        df["ticker"].isin(top_tickers)
        &
        df["option_type"].isin(
            ["CALL", "PUT"]
        )
        &
        df["strike"].notna()
        &
        df["current_price"].gt(0)
    ].copy()

    rows = []

    for ticker in top_tickers:

        group = df[
            df["ticker"] == ticker
        ].copy()

        if group.empty:
            raise RuntimeError(
                f"STEP 8 missing ticker: {ticker}"
            )

        price = safe(
            group["current_price"].median()
        )

        call_wall = choose_wall(
            group,
            "CALL",
            price,
        )

        put_wall = choose_wall(
            group,
            "PUT",
            price,
        )

        support = choose_support(
            group,
            price,
        )

        resistance = choose_resistance(
            group,
            price,
        )

        call_gex = calculate_gex(
            group,
            "CALL",
            price,
        )

        put_gex = calculate_gex(
            group,
            "PUT",
            price,
        )

        if (
            np.isfinite(call_gex)
            and np.isfinite(put_gex)
        ):
            net_gex = (
                call_gex
                +
                put_gex
            )
        else:
            net_gex = np.nan

        structure = classify_structure(
            price,
            call_wall,
            put_wall,
            net_gex,
        )

        if np.isfinite(net_gex):

            if net_gex > 0:
                gex_structure = (
                    "POSITIVE GEX"
                )
            elif net_gex < 0:
                gex_structure = (
                    "NEGATIVE GEX"
                )
            else:
                gex_structure = (
                    "NEUTRAL GEX"
                )

        else:

            gex_structure = (
                "GEX UNAVAILABLE"
            )

        rows.append(
            {
                "ticker": ticker,
                "current_price": price,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "support": support,
                "resistance": resistance,
                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex": net_gex,
                "structure": structure,
                "price_location": price_location(
                    price,
                    support,
                    resistance,
                ),
                "gex_structure": gex_structure,
                "wall_structure": wall_structure(
                    price,
                    call_wall,
                    put_wall,
                ),
                "data_source": "CALCULATED",
            }
        )

    output = pd.DataFrame(rows)

    if len(output) != 20:
        raise RuntimeError(
            "STEP 8 must produce exactly 20 rows"
        )

    if output["ticker"].nunique() != 20:
        raise RuntimeError(
            "STEP 8 duplicate ticker"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("STEP 8 COMPLETE")
    print("ROWS       :", len(output))
    print(
        "TICKERS    :",
        output["ticker"].nunique(),
    )
    print(
        "CALL WALL  :",
        output["call_wall"].notna().sum(),
    )
    print(
        "PUT WALL   :",
        output["put_wall"].notna().sum(),
    )
    print(
        "NET GEX    :",
        output["net_gex"].notna().sum(),
    )
    print(
        "STRUCTURE  :",
        output["structure"].notna().sum(),
    )
    print("STEP 8 OUTPUT : OK")


if __name__ == "__main__":
    main()

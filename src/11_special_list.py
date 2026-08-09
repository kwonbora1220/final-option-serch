import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


INPUT_FILE = "data/analysis/unusual_flow.csv"

OUTPUT_DIR = "data/analysis"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "special_list.csv",
)

OUTPUT_MD = os.path.join(
    OUTPUT_DIR,
    "special_list.md",
)

MAX_RESULTS = 20

MAX_DTE_DISTANCE = 7

MIN_RELATIVE_PREMIUM = 0.02


# ============================================================
# LOG
# ============================================================

def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[11 SPECIAL] {now} | {message}"
    )


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(value):

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(df, candidates):

    mapping = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(
            candidate
        )

        if key in mapping:
            return mapping[key]

    return None


def numeric_series(df, column):

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def normalize_side(value):

    if pd.isna(value):
        return ""

    text = (
        str(value)
        .upper()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
        .rstrip(".")
    )

    if text in {
        "BUY",
        "B",
        "BOT",
        "BTO",
        "BUY_EST",
        "BUY_TO_OPEN",
        "BUY_TO_CLOSE",
        "BTC",
        "BTOC",
    }:
        return "BUY"

    if text in {
        "SELL",
        "S",
        "SOLD",
        "STO",
        "SELL_EST",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "STC",
        "STOC",
    }:
        return "SELL"

    return ""


def normalize_option_type(value):

    if pd.isna(value):
        return ""

    text = (
        str(value)
        .upper()
        .strip()
    )

    if text in {
        "C",
        "CALL",
        "CALLS",
    }:
        return "CALL"

    if text in {
        "P",
        "PUT",
        "PUTS",
    }:
        return "PUT"

    return ""


def normalize_expiration(value):

    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce",
    )


def minmax(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    if values.empty:
        return values

    low = values.min()
    high = values.max()

    if high == low:

        if high > 0:
            return pd.Series(
                100.0,
                index=values.index,
            )

        return pd.Series(
            0.0,
            index=values.index,
        )

    return (
        (values - low)
        /
        (high - low)
        * 100
    )


# ============================================================
# PREPARE
# ============================================================

def prepare_data(raw):

    symbol_col = find_column(
        raw,
        [
            "symbol",
            "ticker",
            "underlying",
            "underlying_symbol",
        ],
    )

    option_type_col = find_column(
        raw,
        [
            "option_type",
            "type",
            "contract_type",
            "call_put",
        ],
    )

    side_col = find_column(
        raw,
        [
            "trade_side",
            "trade_side_estimate",
            "trade_direction",
            "side",
        ],
    )

    strike_col = find_column(
        raw,
        [
            "strike",
            "strike_price",
        ],
    )

    expiration_col = find_column(
        raw,
        [
            "expiration",
            "expiration_date",
            "expiry",
            "expiry_date",
        ],
    )

    dte_col = find_column(
        raw,
        [
            "DTE",
            "dte",
            "days_to_expiration",
        ],
    )

    premium_col = find_column(
        raw,
        [
            "estimated_traded_premium",
            "premium",
            "total_premium",
            "premium_flow",
            "trade_value",
        ],
    )

    delta_col = find_column(
        raw,
        [
            "delta",
        ],
    )

    price_col = find_column(
        raw,
        [
            "underlying_price",
            "current_price",
            "stock_price",
            "underlyingPrice",
        ],
    )

    flow_score_col = find_column(
        raw,
        [
            "flow_score",
            "symbol_flow_score",
        ],
    )

    directional_premium_col = find_column(
        raw,
        [
            "estimated_directional_premium",
            "directional_premium",
        ],
    )

    required = {
        "symbol": symbol_col,
        "option_type": option_type_col,
        "side": side_col,
        "strike": strike_col,
        "delta": delta_col,
        "price": price_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:

        raise ValueError(
            f"STEP 11 missing columns: {missing}"
        )

    result = pd.DataFrame(
        index=raw.index
    )

    result["ticker"] = (
        raw[symbol_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["option_type"] = (
        raw[option_type_col]
        .apply(
            normalize_option_type
        )
    )

    result["side"] = (
        raw[side_col]
        .apply(
            normalize_side
        )
    )

    result["strike"] = numeric_series(
        raw,
        strike_col,
    )

    result["DTE"] = numeric_series(
        raw,
        dte_col,
    )

    if expiration_col:

        result["expiration"] = (
            raw[expiration_col]
            .apply(
                normalize_expiration
            )
        )

    else:

        result["expiration"] = pd.NaT

    result["premium"] = numeric_series(
        raw,
        premium_col,
    ).fillna(0)

    result["delta"] = numeric_series(
        raw,
        delta_col,
    )

    result["current_price"] = numeric_series(
        raw,
        price_col,
    )

    result["flow_score"] = numeric_series(
        raw,
        flow_score_col,
    )

    result["directional_premium"] = (
        numeric_series(
            raw,
            directional_premium_col,
        )
    )

    return result


# ============================================================
# BUILD
# ============================================================

def build_candidates(df):

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["side"] == "BUY")
        &
        (df["delta"] > 0)
    ].copy()

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["side"] == "SELL")
        &
        (df["delta"] < 0)
    ].copy()

    log(
        f"CALL BUY ROWS : {len(calls):,}"
    )

    log(
        f"PUT SELL ROWS : {len(puts):,}"
    )

    if calls.empty or puts.empty:
        return pd.DataFrame()

    common = sorted(
        set(calls["ticker"])
        &
        set(puts["ticker"])
    )

    calls = calls[
        calls["ticker"].isin(common)
    ]

    puts = puts[
        puts["ticker"].isin(common)
    ]

    pairs = pd.merge(
        calls,
        puts,
        on="ticker",
        suffixes=(
            "_call",
            "_put",
        ),
    )

    log(
        f"RAW PAIRS : {len(pairs):,}"
    )

    if pairs.empty:
        return pairs

    # ========================================================
    # PRICE RELATION
    # ========================================================

    pairs = pairs[
        pairs["strike_call"]
        >=
        pairs["current_price_call"]
    ].copy()

    pairs = pairs[
        pairs["strike_put"]
        <=
        pairs["current_price_put"]
    ].copy()

    # ========================================================
    # DTE
    # ========================================================

    pairs["dte_distance"] = (
        pairs["DTE_call"]
        -
        pairs["DTE_put"]
    ).abs()

    pairs["dte_distance"] = (
        pairs["dte_distance"]
        .fillna(999)
    )

    pairs = pairs[
        pairs["dte_distance"]
        <= MAX_DTE_DISTANCE
    ].copy()

    log(
        f"AFTER DTE FILTER : {len(pairs):,}"
    )

    if pairs.empty:
        return pairs

    # ========================================================
    # PREMIUM
    # ========================================================

    pairs["call_premium_value"] = (
        pairs["premium_call"]
        .clip(lower=0)
    )

    pairs["put_premium_value"] = (
        pairs["premium_put"]
        .clip(lower=0)
    )

    pairs["combined_premium"] = (
        pairs["call_premium_value"]
        +
        pairs["put_premium_value"]
    )

    pairs = pairs[
        pairs["combined_premium"] > 0
    ].copy()

    if pairs.empty:
        return pairs

    pairs["premium_ratio"] = (
        pairs["call_premium_value"]
        /
        pairs["combined_premium"]
    )

    # 너무 작은 한쪽 premium은
    # 의미있는 RR이라고 보기 어렵다.
    pairs = pairs[
        (
            pairs["call_premium_value"]
            >=
            pairs["combined_premium"]
            * MIN_RELATIVE_PREMIUM
        )
        &
        (
            pairs["put_premium_value"]
            >=
            pairs["combined_premium"]
            * MIN_RELATIVE_PREMIUM
        )
    ].copy()

    if pairs.empty:
        return pairs

    # ========================================================
    # DIRECTION QUALITY
    # ========================================================

    call_direction = (
        pairs["call_premium_value"]
        /
        pairs["combined_premium"]
    )

    put_direction = (
        pairs["put_premium_value"]
        /
        pairs["combined_premium"]
    )

    pairs["direction_quality"] = (
        50
        +
        (
            call_direction
            -
            put_direction
        )
        * 50
    )

    # ========================================================
    # FLOW QUALITY
    # ========================================================

    call_flow = (
        pairs["flow_score_call"]
        .fillna(0)
    )

    put_flow = (
        pairs["flow_score_put"]
        .fillna(0)
    )

    pairs["flow_quality"] = (
        call_flow * 0.55
        +
        put_flow * 0.45
    )

    # ========================================================
    # DISTANCE QUALITY
    # ========================================================

    call_price = (
        pairs["current_price_call"]
    )

    put_distance = (
        call_price
        -
        pairs["strike_put"]
    ).abs() / call_price

    call_distance = (
        pairs["strike_call"]
        -
        call_price
    ).abs() / call_price

    pairs["distance_quality"] = (
        100
        -
        (
            call_distance
            +
            put_distance
        )
        * 100
    ).clip(
        lower=0,
        upper=100,
    )

    # ========================================================
    # RR SCORE
    # ========================================================

    premium_score = minmax(
        pairs["combined_premium"]
    )

    pairs["rr_score"] = (
        premium_score * 0.40
        +
        pairs["flow_quality"] * 0.25
        +
        pairs["direction_quality"] * 0.20
        +
        pairs["distance_quality"] * 0.15
    )

    # 실제 premium이 높은 것 우선
    pairs = pairs.sort_values(
        [
            "rr_score",
            "combined_premium",
        ],
        ascending=False,
    )

    # ========================================================
    # ONE BEST STRUCTURE PER TICKER
    # ========================================================

    pairs = (
        pairs
        .drop_duplicates(
            "ticker",
            keep="first",
        )
    )

    return pairs.head(
        MAX_RESULTS
    )


# ============================================================
# OUTPUT
# ============================================================

def build_output(pairs):

    if pairs.empty:

        return pd.DataFrame(
            columns=[
                "rank",
                "ticker",
                "rr_score",
                "structure",
                "call_buy_estimate",
                "call_strike",
                "call_dte",
                "call_premium",
                "put_sell_estimate",
                "put_strike",
                "put_dte",
                "put_premium",
                "combined_premium",
                "premium_ratio",
                "direction_quality",
                "flow_quality",
            ]
        )

    rows = []

    for rank, (_, row) in enumerate(
        pairs.iterrows(),
        start=1,
    ):

        rows.append({

            "rank": rank,

            "ticker": row["ticker"],

            "rr_score": round(
                float(row["rr_score"]),
                2,
            ),

            "structure":
                "CALL BUY + PUT SELL",

            "call_buy_estimate":
                "CALL BUY EST.",

            "call_strike":
                row["strike_call"],

            "call_dte":
                row["DTE_call"],

            "call_premium":
                row["call_premium_value"],

            "put_sell_estimate":
                "PUT SELL EST.",

            "put_strike":
                row["strike_put"],

            "put_dte":
                row["DTE_put"],

            "put_premium":
                row["put_premium_value"],

            "combined_premium":
                row["combined_premium"],

            "premium_ratio":
                row["premium_ratio"],

            "direction_quality":
                row["direction_quality"],

            "flow_quality":
                row["flow_quality"],
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("============================================================")
    print("STEP 11 SPECIAL LIST")
    print("============================================================")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            INPUT_FILE
        )

    raw = pd.read_csv(
        INPUT_FILE
    )

    log(
        f"INPUT ROWS : {len(raw):,}"
    )

    df = prepare_data(
        raw
    )

    pairs = build_candidates(
        df
    )

    result = build_output(
        pairs
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# STEP 11 SPECIAL LIST\n\n"
        )

        for _, row in result.iterrows():

            f.write(
                f"## {int(row['rank'])}. "
                f"{row['ticker']}\n\n"
            )

            f.write(
                f"- RR Score: "
                f"{row['rr_score']:.2f}\n"
            )

            f.write(
                "- CALL BUY EST.\n"
            )

            f.write(
                f"  - Strike: "
                f"{row['call_strike']}\n"
            )

            f.write(
                f"  - DTE: "
                f"{row['call_dte']}\n"
            )

            f.write(
                f"  - Premium: "
                f"{row['call_premium']}\n"
            )

            f.write(
                "- PUT SELL EST.\n"
            )

            f.write(
                f"  - Strike: "
                f"{row['put_strike']}\n"
            )

            f.write(
                f"  - DTE: "
                f"{row['put_dte']}\n"
            )

            f.write(
                f"  - Premium: "
                f"{row['put_premium']}\n"
            )

            f.write(
                "\n🔥 BULLISH RISK-REVERSAL\n\n"
            )

    print()
    print("STEP 11 OUTPUT CHECK")

    print(
        f"ROWS              : {len(result)}"
    )

    print(
        f"SPECIAL SCORE     : "
        f"{result['rr_score'].notna().sum()}"
    )

    if not result.empty:

        print(
            "CALL BUY + PUT SELL: "
            f"{len(result)}"
        )

    else:

        print(
            "CALL BUY + PUT SELL: 0"
        )

    print(
        "STEP 11 OUTPUT : OK"
    )


if __name__ == "__main__":
    main()

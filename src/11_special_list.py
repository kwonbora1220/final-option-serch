from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


INPUT_FILE = (
    "data/analysis/unusual_flow.csv"
)

OUTPUT_CSV = (
    "data/analysis/special_list.csv"
)

OUTPUT_MD = (
    "data/analysis/special_list.md"
)

MAX_RESULTS = 20

MAX_DTE_DISTANCE = 14

MIN_RELATIVE_PREMIUM = 0.02


def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[11 SPECIAL] {now} | {message}"
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


def numeric(series):

    return pd.to_numeric(
        series,
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
        "BUY_EST_",
        "BUY_TO_OPEN",
        "BUY_TO_CLOSE",
        "BTC",
    }:
        return "BUY"

    if text in {
        "SELL",
        "S",
        "SOLD",
        "STO",
        "SELL_EST",
        "SELL_EST_",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "STC",
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


def minmax(series):

    values = (
        numeric(series)
        .fillna(0)
    )

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


def prepare(raw):

    symbol_col = find_col(
        raw,
        [
            "symbol",
            "ticker",
            "underlying",
        ],
    )

    option_type_col = find_col(
        raw,
        [
            "option_type",
            "type",
            "call_put",
        ],
    )

    side_col = find_col(
        raw,
        [
            "trade_side",
            "trade_side_estimate",
            "side",
            "trade_direction",
        ],
    )

    strike_col = find_col(
        raw,
        [
            "strike",
            "strike_price",
        ],
    )

    dte_col = find_col(
        raw,
        [
            "DTE",
            "dte",
            "days_to_expiration",
        ],
    )

    premium_col = find_col(
        raw,
        [
            "estimated_traded_premium",
            "premium",
            "estimated_premium",
            "premium_flow",
        ],
    )

    delta_col = find_col(
        raw,
        ["delta"],
    )

    price_col = find_col(
        raw,
        [
            "underlying_price",
            "current_price",
            "stock_price",
        ],
    )

    flow_score_col = find_col(
        raw,
        [
            "flow_score",
            "symbol_flow_score",
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

        raise RuntimeError(
            "STEP 11 missing columns: "
            + ", ".join(missing)
        )

    df = pd.DataFrame()

    df["ticker"] = (
        raw[symbol_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["option_type"] = (
        raw[option_type_col]
        .apply(
            normalize_option_type
        )
    )

    df["side"] = (
        raw[side_col]
        .apply(
            normalize_side
        )
    )

    df["strike"] = numeric(
        raw[strike_col]
    )

    if dte_col:
        df["DTE"] = numeric(
            raw[dte_col]
        )
    else:
        df["DTE"] = np.nan

    if premium_col:
        df["premium"] = (
            numeric(
                raw[premium_col]
            )
            .fillna(0)
            .clip(lower=0)
        )
    else:
        df["premium"] = 0.0

    df["delta"] = numeric(
        raw[delta_col]
    )

    df["current_price"] = numeric(
        raw[price_col]
    )

    if flow_score_col:
        df["flow_score"] = numeric(
            raw[flow_score_col]
        )
    else:
        df["flow_score"] = np.nan

    return df


def build_pairs(df):

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["side"] == "BUY")
        &
        (df["delta"] > 0)
        &
        (df["strike"] >= df["current_price"])
    ].copy()

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["side"] == "SELL")
        &
        (df["delta"] < 0)
        &
        (df["strike"] <= df["current_price"])
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

    if pairs.empty:
        return pairs

    pairs["dte_distance"] = (
        pairs["DTE_call"]
        -
        pairs["DTE_put"]
    ).abs()

    pairs = pairs[
        pairs["dte_distance"]
        .fillna(999)
        <= MAX_DTE_DISTANCE
    ].copy()

    if pairs.empty:
        return pairs

    pairs["call_premium"] = (
        pairs["premium_call"]
        .clip(lower=0)
    )

    pairs["put_premium"] = (
        pairs["premium_put"]
        .clip(lower=0)
    )

    pairs["combined_premium"] = (
        pairs["call_premium"]
        +
        pairs["put_premium"]
    )

    pairs = pairs[
        pairs["combined_premium"] > 0
    ].copy()

    if pairs.empty:
        return pairs

    # 양쪽 모두 최소 2%
    pairs = pairs[
        (
            pairs["call_premium"]
            >=
            pairs["combined_premium"]
            * MIN_RELATIVE_PREMIUM
        )
        &
        (
            pairs["put_premium"]
            >=
            pairs["combined_premium"]
            * MIN_RELATIVE_PREMIUM
        )
    ].copy()

    if pairs.empty:
        return pairs

    # ---------------------------------------------------------
    # PREMIUM QUALITY
    # ---------------------------------------------------------

    pairs["premium_balance"] = (
        1
        -
        (
            (
                pairs["call_premium"]
                -
                pairs["put_premium"]
            ).abs()
            /
            pairs["combined_premium"]
        )
    ).clip(0, 1)

    # ---------------------------------------------------------
    # DISTANCE
    # ---------------------------------------------------------

    pairs["call_distance"] = (
        (
            pairs["strike_call"]
            -
            pairs["current_price_call"]
        ).abs()
        /
        pairs["current_price_call"]
    )

    pairs["put_distance"] = (
        (
            pairs["current_price_put"]
            -
            pairs["strike_put"]
        ).abs()
        /
        pairs["current_price_put"]
    )

    pairs["distance_quality"] = (
        100
        -
        (
            pairs["call_distance"]
            +
            pairs["put_distance"]
        )
        * 250
    ).clip(
        0,
        100,
    )

    # ---------------------------------------------------------
    # FLOW QUALITY
    # ---------------------------------------------------------

    call_flow = (
        pairs["flow_score_call"]
        .fillna(50)
    )

    put_flow = (
        pairs["flow_score_put"]
        .fillna(50)
    )

    pairs["flow_quality"] = (
        call_flow * 0.55
        +
        put_flow * 0.45
    ).clip(
        0,
        100,
    )

    # ---------------------------------------------------------
    # DTE QUALITY
    # ---------------------------------------------------------

    pairs["dte_quality"] = (
        100
        -
        pairs["dte_distance"]
        .fillna(MAX_DTE_DISTANCE)
        * 5
    ).clip(
        0,
        100,
    )

    # ---------------------------------------------------------
    # FINAL RR SCORE
    # ---------------------------------------------------------

    premium_score = minmax(
        pairs["combined_premium"]
    )

    pairs["rr_score"] = (
        premium_score * 0.35
        +
        pairs["flow_quality"] * 0.25
        +
        pairs["distance_quality"] * 0.20
        +
        pairs["premium_balance"] * 100 * 0.10
        +
        pairs["dte_quality"] * 0.10
    )

    pairs = pairs.sort_values(
        [
            "rr_score",
            "combined_premium",
        ],
        ascending=False,
    )

    # ticker당 최고 구조 하나
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


def main():

    if not os.path.exists(
        INPUT_FILE
    ):
        raise FileNotFoundError(
            INPUT_FILE
        )

    raw = pd.read_csv(
        INPUT_FILE
    )

    df = prepare(raw)

    log(
        f"INPUT ROWS : {len(df):,}"
    )

    log(
        f"UNKNOWN SIDE : "
        f"{(df['side'] == '').sum():,}"
    )

    pairs = build_pairs(
        df
    )

    if pairs.empty:

        raise RuntimeError(
            "No valid CALL BUY + PUT SELL "
            "structures detected"
        )

    rows = []

    for rank, (_, row) in enumerate(
        pairs.iterrows(),
        start=1,
    ):

        rows.append(
            {
                "rank": rank,
                "ticker": row["ticker"],
                "rr_score": round(
                    float(
                        row["rr_score"]
                    ),
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
                    row["call_premium"],
                "put_sell_estimate":
                    "PUT SELL EST.",
                "put_strike":
                    row["strike_put"],
                "put_dte":
                    row["DTE_put"],
                "put_premium":
                    row["put_premium"],
                "combined_premium":
                    row["combined_premium"],
                "premium_balance":
                    row["premium_balance"],
                "direction_quality":
                    row["premium_balance"] * 100,
                "flow_quality":
                    row["flow_quality"],
            }
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # MARKDOWN
    # ---------------------------------------------------------

    lines = [
        "# CALL BUY + PUT SELL",
        "",
        f"STRUCTURES : {len(output)}",
        "",
    ]

    for _, row in output.iterrows():

        lines.extend(
            [
                f"## {int(row['rank'])}. "
                f"{row['ticker']}",
                "",
                f"- RR Score: "
                f"{row['rr_score']:.2f}",
                f"- CALL BUY EST. "
                f"{row['call_strike']} "
                f"DTE {row['call_dte']}",
                f"- CALL Premium: "
                f"${row['call_premium']:,.0f}",
                f"- PUT SELL EST. "
                f"{row['put_strike']} "
                f"DTE {row['put_dte']}",
                f"- PUT Premium: "
                f"${row['put_premium']:,.0f}",
                "- Structure: "
                "BULLISH RISK-REVERSAL",
                "",
            ]
        )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if output["ticker"].nunique() != len(output):
        raise RuntimeError(
            "Duplicate special ticker"
        )

    if not (
        output["rr_score"]
        .between(0, 100)
        .all()
    ):
        raise RuntimeError(
            "RR score outside 0-100"
        )

    print()
    print("STEP 11 SPECIAL LIST")
    print(
        "INPUT ROWS       :",
        len(df),
    )
    print(
        "CALL BUY ROWS    :",
        (
            (
                df["option_type"]
                == "CALL"
            )
            &
            (
                df["side"]
                == "BUY"
            )
        ).sum(),
    )
    print(
        "PUT SELL ROWS    :",
        (
            (
                df["option_type"]
                == "PUT"
            )
            &
            (
                df["side"]
                == "SELL"
            )
        ).sum(),
    )
    print(
        "UNKNOWN SIDE     :",
        (
            df["side"] == ""
        ).sum(),
    )
    print(
        "SPECIAL ROWS     :",
        len(output),
    )
    print(
        "SPECIAL TICKERS  :",
        output["ticker"].nunique(),
    )
    print(
        "SCORE VALID      :",
        output["rr_score"].notna().sum(),
    )
    print(
        "STRUCTURE        :",
        "CALL BUY + PUT SELL",
    )
    print(
        "FILTER           : STRICT BULLISH"
    )
    print(
        "CALL DELTA       : > 0"
    )
    print(
        "PUT DELTA        : < 0"
    )

    print()
    print("STEP 11 OUTPUT CHECK")
    print(
        "ROWS              :",
        len(output),
    )
    print(
        "SPECIAL SCORE     :",
        output["rr_score"].notna().sum(),
    )
    print(
        "CALL BUY + PUT SELL:",
        (
            output["structure"]
            ==
            "CALL BUY + PUT SELL"
        ).sum(),
    )
    print(
        "STEP 11 OUTPUT : OK"
    )


if __name__ == "__main__":
    main()

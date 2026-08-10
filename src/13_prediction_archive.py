from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis",
)

DECISION_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv",
)

MARKET_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv",
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "prediction_history.csv",
)


# ============================================================
# HELPERS
# ============================================================

def find_col(df, names):

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    return None


def numeric_series(df, names):

    col = find_col(
        df,
        names,
    )

    if col is None:
        return pd.Series(
            [float("nan")] * len(df),
            index=df.index,
        )

    return pd.to_numeric(
        df[col],
        errors="coerce",
    )


def text_series(df, names):

    col = find_col(
        df,
        names,
    )

    if col is None:
        return pd.Series(
            [""] * len(df),
            index=df.index,
        )

    return (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# LOAD
# ============================================================

decision = pd.read_csv(
    DECISION_FILE
)

market = pd.read_csv(
    MARKET_FILE
)

top20 = pd.read_csv(
    TOP20_FILE
)


if decision.empty:
    raise RuntimeError(
        "decision.csv is empty"
    )


# ============================================================
# CURRENT SIGNAL DATE
# ============================================================

signal_timestamp = datetime.now(
    timezone.utc
).isoformat()

signal_date = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%d"
)


# ============================================================
# MARKET
# ============================================================

market_row = market.iloc[-1]

market_score_col = find_col(
    market,
    [
        "market_score",
        "market_regime_score",
        "score",
    ],
)

market_regime_col = find_col(
    market,
    [
        "market_regime",
        "regime",
    ],
)

market_score = (
    pd.to_numeric(
        market_row[market_score_col],
        errors="coerce",
    )
    if market_score_col
    else float("nan")
)

market_regime = (
    str(
        market_row[
            market_regime_col
        ]
    ).strip()
    if market_regime_col
    else ""
)


# ============================================================
# MARKET DIRECTIONS
# ============================================================

def market_value(name):

    col = find_col(
        market,
        [name],
    )

    if col is None:
        return ""

    return str(
        market_row[col]
    ).strip()


ndx_direction = market_value(
    "ndx_direction"
)

spy_direction = market_value(
    "spy_direction"
)

soxx_direction = market_value(
    "soxx_direction"
)

dia_direction = market_value(
    "dia_direction"
)


# ============================================================
# DECISION NORMALIZATION
# ============================================================

ticker_col = find_col(
    decision,
    [
        "ticker",
        "symbol",
    ],
)

if ticker_col is None:
    raise RuntimeError(
        "decision.csv ticker column missing"
    )


result = pd.DataFrame()

result["signal_date"] = [
    signal_date
] * len(decision)

result["signal_timestamp"] = [
    signal_timestamp
] * len(decision)

result["ticker"] = (
    decision[ticker_col]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# DECISION DATA
# ============================================================

for output_name, source_names in [

    (
        "decision_score",
        [
            "decision_score",
            "score",
        ],
    ),

    (
        "market_score",
        [
            "market_score",
        ],
    ),

    (
        "flow_score",
        [
            "flow_score",
        ],
    ),

    (
        "structure_score",
        [
            "structure_score",
        ],
    ),

    (
        "option_score",
        [
            "option_score",
        ],
    ),

]:

    result[output_name] = numeric_series(
        decision,
        source_names,
    )


decision_col = find_col(
    decision,
    [
        "decision",
    ],
)

if decision_col:

    result["decision"] = (
        decision[decision_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

else:

    result["decision"] = ""


# ============================================================
# PRICE
#
# We try several possible names.
# ============================================================

result["signal_price"] = numeric_series(
    decision,
    [
        "current_price",
        "price",
        "stock_price",
        "underlying_price",
    ],
)


# ============================================================
# MARKET SNAPSHOT
# ============================================================

result["market_regime"] = market_regime

result["ndx_direction"] = ndx_direction

result["spy_direction"] = spy_direction

result["soxx_direction"] = soxx_direction

result["dia_direction"] = dia_direction


# ============================================================
# TOP20 RANK
# ============================================================

top_ticker_col = find_col(
    top20,
    [
        "ticker",
        "symbol",
    ],
)

rank_col = find_col(
    top20,
    [
        "rank",
    ],
)

if (
    top_ticker_col
    and rank_col
):

    rank_map = dict(
        zip(
            top20[
                top_ticker_col
            ]
            .astype(str)
            .str.upper()
            .str.strip(),

            pd.to_numeric(
                top20[rank_col],
                errors="coerce",
            ),
        )
    )

    result["top20_rank"] = (
        result["ticker"]
        .map(rank_map)
    )

else:

    result["top20_rank"] = float("nan")


# ============================================================
# INITIAL PERFORMANCE COLUMNS
# ============================================================

for column in [

    "d1_price",
    "d1_return",

    "d3_price",
    "d3_return",

    "d5_price",
    "d5_return",

    "mfe",
    "mae",

    "hit_d1",
    "hit_d3",
    "hit_d5",

]:

    result[column] = pd.NA


# ============================================================
# APPEND HISTORY
# ============================================================

if os.path.exists(
    OUTPUT_FILE
):

    old = pd.read_csv(
        OUTPUT_FILE
    )

    history = pd.concat(
        [
            old,
            result,
        ],
        ignore_index=True,
    )

else:

    history = result


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

history = (
    history
    .drop_duplicates(
        subset=[
            "signal_date",
            "ticker",
        ],
        keep="last",
    )
)


# ============================================================
# SAVE
# ============================================================

history.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "=========================================="
)

print(
    "STEP 13 - PREDICTION ARCHIVE"
)

print(
    "=========================================="
)

print(
    "TODAY SIGNALS :",
    len(result),
)

print(
    "TOTAL HISTORY  :",
    len(history),
)

print(
    "OUTPUT         :",
    OUTPUT_FILE,
)

print()

print(
    result[
        [
            "ticker",
            "decision_score",
            "decision",
            "market_score",
            "market_regime",
        ]
    ]
    .to_string(
        index=False
    )
)

print()

print(
    "STEP 13 OUTPUT : OK"
)

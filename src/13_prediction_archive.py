from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# STEP 13 - PREDICTION ARCHIVE
# ============================================================
#
# 목적:
#
# STEP 9 decision.csv
# STEP 8 structure.csv
# STEP 1 market_regime.csv
# STEP 6 top20.csv
#
# 를 하나의 "오늘의 신호"로 보관한다.
#
# 이후 STEP 14가 이 데이터를 사용해서
#
# D+1
# D+3
# D+5
# MFE
# MAE
#
# 를 계산한다.
#
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

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv",
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
        str(column)
        .strip()
        .lower()
        .replace(" ", "_"): column
        for column in df.columns
    }

    for name in names:

        key = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:

            return normalized[key]

    return None


def numeric(value):

    try:

        value = float(value)

        if pd.notna(value):

            return value

    except Exception:

        pass

    return float("nan")


def normalize_ticker(series):

    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
    )


# ============================================================
# FILE CHECK
# ============================================================


for file in [
    DECISION_FILE,
    STRUCTURE_FILE,
    MARKET_FILE,
    TOP20_FILE,
]:

    if not os.path.exists(file):

        raise RuntimeError(
            f"Required file not found: {file}"
        )


# ============================================================
# LOAD
# ============================================================


decision = pd.read_csv(
    DECISION_FILE
)

structure = pd.read_csv(
    STRUCTURE_FILE
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


if structure.empty:

    raise RuntimeError(
        "structure.csv is empty"
    )


# ============================================================
# TICKER COLUMNS
# ============================================================


decision_ticker_col = find_col(
    decision,
    [
        "ticker",
        "symbol",
    ],
)

structure_ticker_col = find_col(
    structure,
    [
        "ticker",
        "symbol",
    ],
)

top20_ticker_col = find_col(
    top20,
    [
        "ticker",
        "symbol",
    ],
)


if decision_ticker_col is None:

    raise RuntimeError(
        "decision.csv ticker column missing"
    )


if structure_ticker_col is None:

    raise RuntimeError(
        "structure.csv ticker column missing"
    )


# ============================================================
# NORMALIZE TICKERS
# ============================================================


decision["ticker"] = normalize_ticker(
    decision[decision_ticker_col]
)

structure["ticker"] = normalize_ticker(
    structure[structure_ticker_col]
)

if top20_ticker_col:

    top20["ticker"] = normalize_ticker(
        top20[top20_ticker_col]
    )


# ============================================================
# STRUCTURE PRICE
# ============================================================


price_col = find_col(
    structure,
    [
        "current_price",
        "price",
        "stock_price",
        "underlying_price",
    ],
)


if price_col is None:

    raise RuntimeError(
        "No current price column found "
        "in structure.csv"
    )


structure["signal_price"] = pd.to_numeric(
    structure[price_col],
    errors="coerce",
)


# ============================================================
# STRUCTURE FIELDS
# ============================================================


structure_fields = [

    "current_price",

    "call_wall",
    "put_wall",

    "support",
    "resistance",

    "call_gex",
    "put_gex",
    "net_gex",

    "structure",
    "price_location",
    "gex_structure",
    "wall_structure",

]


available_structure_fields = [

    column
    for column in structure_fields
    if column in structure.columns

]


structure_small = structure[
    [
        "ticker",
        "signal_price",
        *available_structure_fields,
    ]
].copy()


structure_small = (
    structure_small
    .drop_duplicates(
        "ticker",
        keep="last",
    )
)


# ============================================================
# MERGE DECISION + STRUCTURE
# ============================================================


result = decision.merge(
    structure_small,
    on="ticker",
    how="left",
    suffixes=("", "_structure"),
)


# ============================================================
# VERIFY PRICE
# ============================================================


missing_price = result[
    "signal_price"
].isna()


if missing_price.any():

    missing_tickers = (
        result.loc[
            missing_price,
            "ticker"
        ]
        .astype(str)
        .tolist()
    )

    raise RuntimeError(
        "Signal price missing for: "
        + ", ".join(
            missing_tickers
        )
    )


# ============================================================
# MARKET SNAPSHOT
# ============================================================


if market.empty:

    raise RuntimeError(
        "market_regime.csv is empty"
    )


market_row = market.iloc[-1]


def market_value(names):

    column = find_col(
        market,
        names,
    )

    if column is None:

        return ""

    value = market_row[column]

    if pd.isna(value):

        return ""

    return str(value).strip()


market_score = numeric(
    market_value(
        [
            "market_score",
            "market_regime_score",
            "score",
        ]
    )
)


market_regime = market_value(
    [
        "market_regime",
        "regime",
    ]
)


ndx_direction = market_value(
    [
        "ndx_direction",
    ]
)


spy_direction = market_value(
    [
        "spy_direction",
    ]
)


soxx_direction = market_value(
    [
        "soxx_direction",
    ]
)


dia_direction = market_value(
    [
        "dia_direction",
    ]
)


# ============================================================
# DATE
# ============================================================


now = datetime.now(
    timezone.utc
)

signal_date = now.strftime(
    "%Y-%m-%d"
)

signal_timestamp = now.isoformat()


# ============================================================
# CREATE ARCHIVE
# ============================================================


archive = pd.DataFrame()

archive["signal_date"] = [
    signal_date
] * len(result)

archive["signal_timestamp"] = [
    signal_timestamp
] * len(result)


# ============================================================
# BASIC
# ============================================================


archive["ticker"] = result[
    "ticker"
]


# ============================================================
# DECISION DATA
# ============================================================


for column in [

    "decision_score",
    "market_score",
    "flow_score",
    "direction_score",
    "structure_score",
    "price_score",
    "index_score",

]:

    if column in result.columns:

        archive[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    else:

        archive[column] = float("nan")


decision_col = find_col(
    result,
    [
        "decision",
        "final_decision",
    ],
)


if decision_col:

    archive["decision"] = (
        result[decision_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

else:

    archive["decision"] = ""


# ============================================================
# SIGNAL PRICE
# ============================================================


archive["signal_price"] = pd.to_numeric(
    result["signal_price"],
    errors="coerce",
)


# ============================================================
# MARKET
# ============================================================


archive["market_regime"] = market_regime

archive["market_score_snapshot"] = (
    market_score
)

archive["ndx_direction"] = (
    ndx_direction
)

archive["spy_direction"] = (
    spy_direction
)

archive["soxx_direction"] = (
    soxx_direction
)

archive["dia_direction"] = (
    dia_direction
)


# ============================================================
# TOP20 RANK
# ============================================================


archive["top20_rank"] = float("nan")


if (
    top20_ticker_col
    and "rank" in top20.columns
):

    rank_map = dict(
        zip(
            top20["ticker"],
            pd.to_numeric(
                top20["rank"],
                errors="coerce",
            ),
        )
    )

    archive["top20_rank"] = (
        archive["ticker"]
        .map(rank_map)
    )


# ============================================================
# STRUCTURE SNAPSHOT
# ============================================================


for column in available_structure_fields:

    output_column = (
        f"structure_{column}"
    )

    if output_column in result.columns:

        archive[output_column] = (
            result[output_column]
        )

    elif column in result.columns:

        archive[output_column] = (
            result[column]
        )


# ============================================================
# FUTURE PERFORMANCE COLUMNS
# ============================================================


for column in [

    "d1_price",
    "d1_return",
    "d1_mfe",
    "d1_mae",
    "hit_d1",

    "d3_price",
    "d3_return",
    "d3_mfe",
    "d3_mae",
    "hit_d3",

    "d5_price",
    "d5_return",
    "d5_mfe",
    "d5_mae",
    "hit_d5",

    "mfe",
    "mae",

]:

    archive[column] = pd.NA


# ============================================================
# APPEND EXISTING HISTORY
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
            archive,
        ],
        ignore_index=True,
        sort=False,
    )

else:

    history = archive


# ============================================================
# DUPLICATE PROTECTION
#
# 같은 날짜 + 같은 종목은
# 한 번만 저장한다.
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
    .reset_index(
        drop=True
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
# OUTPUT
# ============================================================


print()
print("=" * 70)
print("🔥 STEP 13 - PREDICTION ARCHIVE")
print("=" * 70)

print(
    "TODAY SIGNALS :",
    len(archive)
)

print(
    "TOTAL HISTORY :",
    len(history)
)

print(
    "SIGNAL DATE   :",
    signal_date
)

print(
    "PRICE VALID   :",
    archive["signal_price"]
    .notna()
    .sum()
)

print(
    "OUTPUT        :",
    OUTPUT_FILE
)

print()

print(
    archive[
        [
            "ticker",
            "signal_price",
            "decision_score",
            "decision",
            "market_regime",
        ]
    ].to_string(
        index=False
    )
)

print()
print("STEP 13 OUTPUT : OK")

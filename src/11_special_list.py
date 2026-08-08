
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/analysis/options_greeks.csv"

OUTPUT_DIR = "data/analysis"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "special_list.csv"
)

OUTPUT_MD = os.path.join(
    OUTPUT_DIR,
    "special_list.md"
)

# 최대 SPECIAL LIST
MAX_RESULTS = 20

# CALL BUY / PUT SELL 판정에 사용할 최소 점수
MIN_SPECIAL_SCORE = 50.0


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
# COLUMN HELPERS
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

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(
            candidate
        )

        if key in normalized:
            return normalized[key]

    return None


def numeric_series(df, column):

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# SIDE NORMALIZATION
# ============================================================

def normalize_side(value):

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    text = (
        text
        .replace("-", "_")
        .replace(" ", "_")
    )

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if text in {
        "BUY",
        "B",
        "BOT",
        "BTO",
        "BUY_TO_OPEN",
        "BUY_TO_CLOSE",
        "BTC",
        "BTOC",
        "BUYTOOPEN",
        "BUYTOCLOSE",
    }:
        return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if text in {
        "SELL",
        "S",
        "SOLD",
        "STO",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "STC",
        "STOC",
        "SELLTOOPEN",
        "SELLTOCLOSE",
    }:
        return "SELL"

    # --------------------------------------------------------
    # TEXT CONTAINS
    # --------------------------------------------------------

    if "BUY" in text:
        return "BUY"

    if "SELL" in text:
        return "SELL"

    if text.startswith("BTO"):
        return "BUY"

    if text.startswith("BTC"):
        return "BUY"

    if text.startswith("STO"):
        return "SELL"

    if text.startswith("STC"):
        return "SELL"

    return ""


# ============================================================
# OPTION TYPE NORMALIZATION
# ============================================================

def normalize_option_type(value):

    if pd.isna(value):
        return ""

    text = str(value).upper().strip()

    if text in {
        "C",
        "CALL",
        "CALLS"
    }:
        return "CALL"

    if text in {
        "P",
        "PUT",
        "PUTS"
    }:
        return "PUT"

    return text


# ============================================================
# SYMBOL EXTRACTION
# ============================================================

def extract_symbol(df):

    column = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
            "stock",
            "stock_symbol"
        ]
    )

    if column is None:
        raise ValueError(
            "Ticker/symbol column not found"
        )

    return (
        df[column]
        .astype(str)
        .str.upper()
        .str.strip()
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    symbol_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
            "stock",
            "stock_symbol"
        ]
    )

    type_col = find_column(
        df,
        [
            "option_type",
            "type",
            "contract_type",
            "call_put"
        ]
    )

    side_col = find_column(
        df,
        [
            "side",
            "trade_side",
            "transaction",
            "transaction_type",
            "order_side",
            "sentiment",
            "flow_type",
            "trade_type",
            "direction"
        ]
    )

    strike_col = find_column(
        df,
        [
            "strike",
            "strike_price"
        ]
    )

    dte_col = find_column(
        df,
        [
            "DTE",
            "dte",
            "days_to_expiration"
        ]
    )

    expiration_col = find_column(
        df,
        [
            "expiration",
            "expiration_date",
            "expiry",
            "expiry_date"
        ]
    )

    volume_col = find_column(
        df,
        [
            "volume",
            "option_volume"
        ]
    )

    oi_col = find_column(
        df,
        [
            "open_interest",
            "oi"
        ]
    )

    premium_col = find_column(
        df,
        [
            "premium",
            "total_premium",
            "premium_flow",
            "notional",
            "trade_value"
        ]
    )

    delta_col = find_column(
        df,
        [
            "delta"
        ]
    )

    iv_col = find_column(
        df,
        [
            "impliedVolatility",
            "implied_volatility",
            "iv"
        ]
    )

    price_col = find_column(
        df,
        [
            "underlying_price",
            "current_price",
            "stock_price",
            "underlyingPrice"
        ]
    )

    if symbol_col is None:
        raise ValueError(
            "Symbol column not found"
        )

    if type_col is None:
        raise ValueError(
            "Option type column not found"
        )

    if side_col is None:

        log(
            "WARNING: No trade side column found"
        )

        log(
            "SPECIAL LIST requires actual BUY/SELL information"
        )

        side_series = pd.Series(
            "",
            index=df.index
        )

    else:

        side_series = (
            df[side_col]
            .apply(normalize_side)
        )

        log(
            f"SIDE COLUMN : {side_col}"
        )

    result = pd.DataFrame(
        index=df.index
    )

    result["ticker"] = (
        df[symbol_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["option_type"] = (
        df[type_col]
        .apply(normalize_option_type)
    )

    result["side"] = side_series

    result["strike"] = numeric_series(
        df,
        strike_col
    )

    result["DTE"] = numeric_series(
        df,
        dte_col
    )

    result["volume"] = numeric_series(
        df,
        volume_col
    )

    result["open_interest"] = numeric_series(
        df,
        oi_col
    )

    result["premium"] = numeric_series(
        df,
        premium_col
    )

    result["delta"] = numeric_series(
        df,
        delta_col
    )

    result["iv"] = numeric_series(
        df,
        iv_col
    )

    result["current_price"] = numeric_series(
        df,
        price_col
    )

    if expiration_col is not None:

        result["expiration"] = (
            df[expiration_col]
            .astype(str)
            .replace(
                {
                    "nan": ""
                }
            )
            .str.strip()
        )

    else:

        result["expiration"] = ""

    return result


# ============================================================
# PREMIUM SCORE
# ============================================================

def normalize_score(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if values.empty:
        return values

    maximum = values.max()

    if not np.isfinite(maximum):
        return pd.Series(
            0.0,
            index=values.index
        )

    if maximum <= 0:
        return pd.Series(
            0.0,
            index=values.index
        )

    return (
        values / maximum * 100.0
    )


# ============================================================
# BUILD SPECIAL CANDIDATES
# ============================================================

def build_special_candidates(df):

    calls = df[
        (df["option_type"] == "CALL")
        & (df["side"] == "BUY")
    ].copy()

    puts = df[
        (df["option_type"] == "PUT")
        & (df["side"] == "SELL")
    ].copy()

    log(
        f"CALL BUY ROWS : {len(calls):,}"
    )

    log(
        f"PUT SELL ROWS : {len(puts):,}"
    )

    if calls.empty or puts.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Aggregate each leg by ticker
    # --------------------------------------------------------

    call_group = (
        calls
        .groupby("ticker", as_index=False)
        .agg(
            call_buy_count=(
                "ticker",
                "size"
            ),
            call_buy_volume=(
                "volume",
                "sum"
            ),
            call_buy_premium=(
                "premium",
                "sum"
            ),
            call_buy_oi=(
                "open_interest",
                "sum"
            ),
            call_buy_delta=(
                "delta",
                "mean"
            ),
            call_buy_iv=(
                "iv",
                "mean"
            ),
            call_strike=(
                "strike",
                "first"
            ),
            call_dte=(
                "DTE",
                "first"
            ),
            call_expiration=(
                "expiration",
                "first"
            ),
            current_price=(
                "current_price",
                "median"
            ),
        )
    )

    put_group = (
        puts
        .groupby("ticker", as_index=False)
        .agg(
            put_sell_count=(
                "ticker",
                "size"
            ),
            put_sell_volume=(
                "volume",
                "sum"
            ),
            put_sell_premium=(
                "premium",
                "sum"
            ),
            put_sell_oi=(
                "open_interest",
                "sum"
            ),
            put_sell_delta=(
                "delta",
                "mean"
            ),
            put_sell_iv=(
                "iv",
                "mean"
            ),
            put_strike=(
                "strike",
                "first"
            ),
            put_dte=(
                "DTE",
                "first"
            ),
            put_expiration=(
                "expiration",
                "first"
            ),
        )
    )

    merged = pd.merge(
        call_group,
        put_group,
        on="ticker",
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Structure validation
    # --------------------------------------------------------

    merged["strike_distance"] = (
        merged["call_strike"]
        - merged["put_strike"]
    ).abs()

    merged["dte_distance"] = (
        merged["call_dte"]
        - merged["put_dte"]
    ).abs()
    
    # --------------------------------------------------------
    # Directional confirmation
    #
    # CALL BUY:
    # Positive delta is bullish
    #
    # PUT SELL:
    # Selling a put is also bullish
    # --------------------------------------------------------

    merged["call_bullish"] = (
        merged["call_buy_delta"]
        .fillna(0)
        > 0
    )

    merged["put_bullish"] = (
        merged["put_sell_delta"]
        .fillna(0)
        < 0
    )

    # --------------------------------------------------------
    # Premium / volume scores
    # --------------------------------------------------------

    merged["call_premium_score"] = (
        normalize_score(
            merged["call_buy_premium"]
        )
    )

    merged["put_premium_score"] = (
        normalize_score(
            merged["put_sell_premium"]
        )
    )

    merged["call_volume_score"] = (
        normalize_score(
            merged["call_buy_volume"]
        )
    )

    merged["put_volume_score"] = (
        normalize_score(
            merged["put_sell_volume"]
        )
    )

    # --------------------------------------------------------
    # Special score
    #
    # 30% CALL BUY strength
    # 30% PUT SELL strength
    # 20% directional confirmation
    # 10% volume
    # 10% expiry alignment
    # --------------------------------------------------------

    merged["flow_strength"] = (
        merged["call_premium_score"] * 0.15
        + merged["put_premium_score"] * 0.15
        + merged["call_volume_score"] * 0.10
        + merged["put_volume_score"] * 0.10
    )

    merged["direction_score"] = (
        merged["call_bullish"].astype(int)
        + merged["put_bullish"].astype(int)
    ) * 10.0

    merged["expiry_score"] = np.where(
        merged["dte_distance"].fillna(999) <= 7,
        10.0,
        0.0
    )

    merged["special_score"] = (
        merged["flow_strength"]
        + merged["direction_score"]
        + merged["expiry_score"]
    )

    # --------------------------------------------------------
    # Structure label
    # --------------------------------------------------------

    merged["structure"] = (
        "CALL BUY + PUT SELL"
    )

    merged["special_reason"] = (
        "CALL BUY confirmed | "
        "PUT SELL confirmed | "
        "Bullish synthetic structure"
    )

    # --------------------------------------------------------
    # Only valid structures
    # --------------------------------------------------------

    merged = merged[
        (merged["call_buy_count"] > 0)
        & (merged["put_sell_count"] > 0)
    ].copy()

    if merged.empty:
        return merged

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    merged = merged.sort_values(
        [
            "special_score",
            "call_buy_premium",
            "put_sell_premium"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    merged["special_rank"] = (
        np.arange(len(merged))
        + 1
    )

    # --------------------------------------------------------
    # Limit output
    # --------------------------------------------------------

    merged = merged.head(
        MAX_RESULTS
    ).copy()

    return merged


# ============================================================
# EMPTY OUTPUT
# ============================================================

def create_empty_output():

    columns = [
        "special_rank",
        "ticker",
        "structure",
        "special_score",
        "current_price",

        "call_buy_count",
        "call_buy_volume",
        "call_buy_premium",
        "call_buy_oi",
        "call_buy_delta",
        "call_buy_iv",
        "call_strike",
        "call_dte",
        "call_expiration",

        "put_sell_count",
        "put_sell_volume",
        "put_sell_premium",
        "put_sell_oi",
        "put_sell_delta",
        "put_sell_iv",
        "put_strike",
        "put_dte",
        "put_expiration",

        "strike_distance",
        "dte_distance",

        "special_reason",
        "data_source",
    ]

    return pd.DataFrame(
        columns=columns
    )


# ============================================================
# MARKDOWN
# ============================================================

def create_markdown(df):

    lines = []

    lines.append(
        "# STEP 11 - SPECIAL LIST"
    )

    lines.append("")

    lines.append(
        "## CALL BUY + PUT SELL"
    )

    lines.append("")

    if df.empty:

        lines.append(
            "오늘 확인된 CALL BUY + PUT SELL "
            "특수 구조가 없습니다."
        )

        lines.append("")

        lines.append(
            "※ 0건도 정상 결과입니다."
        )

        lines.append("")

        lines.append(
            "SPECIAL LIST : 0"
        )

        return "\n".join(lines)

    lines.append(
        f"총 SPECIAL : {len(df)}"
    )

    lines.append("")

    for _, row in df.iterrows():

        ticker = row["ticker"]

        score = float(
            row["special_score"]
        )

        call_strike = row["call_strike"]

        put_strike = row["put_strike"]

        call_dte = row["call_dte"]

        put_dte = row["put_dte"]

        lines.append(
            f"## #{int(row['special_rank'])} {ticker}"
        )

        lines.append("")

        lines.append(
            f"- 구조: **CALL BUY + PUT SELL**"
        )

        lines.append(
            f"- Special Score: **{score:.2f}**"
        )

        lines.append(
            f"- 현재가: {row['current_price']}"
        )

        lines.append(
            f"- CALL BUY: strike={call_strike}, "
            f"DTE={call_dte}"
        )

        lines.append(
            f"- PUT SELL: strike={put_strike}, "
            f"DTE={put_dte}"
        )

        lines.append(
            f"- CALL BUY premium: "
            f"{row['call_buy_premium']}"
        )

        lines.append(
            f"- PUT SELL premium: "
            f"{row['put_sell_premium']}"
        )

        lines.append(
            f"- 이유: {row['special_reason']}"
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    log(
        f"INPUT : {INPUT_FILE}"
    )

    raw = pd.read_csv(
        INPUT_FILE
    )

    log(
        f"INPUT ROWS : {len(raw):,}"
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(
        raw
    )

    log(
        f"PREPARED ROWS : {len(df):,}"
    )

    # --------------------------------------------------------
    # SPECIAL SEARCH
    # --------------------------------------------------------

    special = build_special_candidates(
        df
    )

    if special is None or special.empty:

        log(
            "NO CALL BUY + PUT SELL STRUCTURE FOUND"
        )

        output = create_empty_output()

    else:

        output = special.copy()

        output["data_source"] = (
            "CALCULATED"
        )

    # --------------------------------------------------------
    # Ensure stable columns
    # --------------------------------------------------------

    required_output_columns = [
        "special_rank",
        "ticker",
        "structure",
        "special_score",
        "current_price",
        "call_buy_count",
        "call_buy_volume",
        "call_buy_premium",
        "call_buy_oi",
        "call_buy_delta",
        "call_buy_iv",
        "call_strike",
        "call_dte",
        "call_expiration",
        "put_sell_count",
        "put_sell_volume",
        "put_sell_premium",
        "put_sell_oi",
        "put_sell_delta",
        "put_sell_iv",
        "put_strike",
        "put_dte",
        "put_expiration",
        "strike_distance",
        "dte_distance",
        "special_reason",
        "data_source",
    ]

    for column in required_output_columns:

        if column not in output.columns:

            output[column] = np.nan

    output = output[
        required_output_columns
    ]

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SAVE MARKDOWN
    # --------------------------------------------------------

    markdown = create_markdown(
        output
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            markdown
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("🔎 STEP 11 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT ROWS       : {len(raw):,}"
    )

    print(
        f"CALL BUY ROWS    : "
        f"{(df['option_type'].eq('CALL') & df['side'].eq('BUY')).sum():,}"
    )

    print(
        f"PUT SELL ROWS    : "
        f"{(df['option_type'].eq('PUT') & df['side'].eq('SELL')).sum():,}"
    )

    print(
        f"SPECIAL ROWS     : {len(output)}"
    )

    print(
        f"SPECIAL TICKERS  : "
        f"{output['ticker'].nunique()}"
    )

    print(
        f"SCORE VALID      : "
        f"{output['special_score'].notna().sum()}"
    )

    print()
    print(
        "STRUCTURE        : CALL BUY + PUT SELL"
    )

    print(
        f"CSV              : {OUTPUT_CSV}"
    )

    print(
        f"MARKDOWN         : {OUTPUT_MD}"
    )

    print("=" * 72)

    log(
        "STEP 11 SPECIAL LIST COMPLETE"
    )


if __name__ == "__main__":
    main()


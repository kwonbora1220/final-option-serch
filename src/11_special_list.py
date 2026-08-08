import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# STEP 11 - SPECIAL LIST
#
# CALL BUY + PUT SELL
#
# INPUT:
# data/analysis/unusual_flow.csv
#
# ============================================================

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

# 같은 구조로 인정할 최대 DTE 차이
MAX_DTE_DISTANCE = 7

# 같은 구조로 인정할 최대 만기 차이
MAX_EXPIRATION_DISTANCE = 7


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
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# SIDE NORMALIZATION
#
# IMPORTANT:
#
# STEP 5 produces:
#
# BUY EST.
# SELL EST.
# UNKNOWN
#
# STEP 11 converts them to:
#
# BUY
# SELL
# ""
#
# ============================================================

def normalize_side(value):

    if pd.isna(value):
        return ""

    text = (
        str(value)
        .upper()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )

    # Remove trailing punctuation
    text = text.rstrip(".")

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

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
        "SELL_EST",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "STC",
        "STOC",
        "SELLTOOPEN",
        "SELLTOCLOSE",
    }:

        return "SELL"

    # --------------------------------------------------------
    # UNKNOWN
    #
    # NEVER GUESS
    # --------------------------------------------------------

    return ""


# ============================================================
# OPTION TYPE
# ============================================================

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


# ============================================================
# EXPIRATION
# ============================================================

def normalize_expiration(value):

    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce",
    )


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def minmax_score(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:
        return values

    maximum = values.max()
    minimum = values.min()

    if not np.isfinite(maximum):

        return pd.Series(
            0.0,
            index=values.index,
        )

    if maximum == minimum:

        if maximum > 0:

            return pd.Series(
                100.0,
                index=values.index,
            )

        return pd.Series(
            0.0,
            index=values.index,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
        * 100.0
    )


# ============================================================
# PREPARE INPUT
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

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Prefer normalized trade_side from STEP 5.
    #
    # If it doesn't exist, fall back to
    # trade_side_estimate.
    # --------------------------------------------------------

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

    volume_col = find_column(
        raw,
        [
            "volume",
            "option_volume",
        ],
    )

    oi_col = find_column(
        raw,
        [
            "openInterest",
            "open_interest",
            "oi",
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

    gamma_col = find_column(
        raw,
        [
            "gamma",
        ],
    )

    vega_col = find_column(
        raw,
        [
            "vega",
        ],
    )

    iv_col = find_column(
        raw,
        [
            "impliedVolatility",
            "implied_volatility",
            "iv",
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

    if symbol_col is None:

        raise ValueError(
            "Symbol column not found"
        )

    if option_type_col is None:

        raise ValueError(
            "Option type column not found"
        )

    if side_col is None:

        raise ValueError(
            "Trade side column not found. "
            "STEP 11 requires STEP 5 trade direction."
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

    if expiration_col is not None:

        result["expiration"] = (
            raw[expiration_col]
            .apply(
                normalize_expiration
            )
        )

    else:

        result["expiration"] = pd.NaT

    result["volume"] = numeric_series(
        raw,
        volume_col,
    )

    result["open_interest"] = numeric_series(
        raw,
        oi_col,
    )

    result["premium"] = numeric_series(
        raw,
        premium_col,
    )

    result["delta"] = numeric_series(
        raw,
        delta_col,
    )

    result["gamma"] = numeric_series(
        raw,
        gamma_col,
    )

    result["vega"] = numeric_series(
        raw,
        vega_col,
    )

    result["iv"] = numeric_series(
        raw,
        iv_col,
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
# BUILD SPECIAL STRUCTURES
# ============================================================

def build_special_candidates(df):

    # --------------------------------------------------------
    # CALL BUY
    # --------------------------------------------------------

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["side"] == "BUY")
    ].copy()

    # --------------------------------------------------------
    # PUT SELL
    # --------------------------------------------------------

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["side"] == "SELL")
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
    # Keep valid tickers
    # --------------------------------------------------------

    calls = calls[
        calls["ticker"].ne("")
    ].copy()

    puts = puts[
        puts["ticker"].ne("")
    ].copy()

    if calls.empty or puts.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Pair by ticker
    #
    # We still evaluate individual option pairs.
    # --------------------------------------------------------

    calls["_pair_key"] = 1
    puts["_pair_key"] = 1

    pairs = pd.merge(
        calls,
        puts,
        on=[
            "ticker",
            "_pair_key",
        ],
        suffixes=(
            "_call",
            "_put",
        ),
    )

    if pairs.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # DTE DISTANCE
    # --------------------------------------------------------

    pairs["dte_distance"] = (
        pairs["DTE_call"]
        - pairs["DTE_put"]
    ).abs()

    pairs["dte_distance"] = (
        pairs["dte_distance"]
        .fillna(999)
    )

    # --------------------------------------------------------
    # EXPIRATION DISTANCE
    # --------------------------------------------------------

    pairs["expiration_distance"] = (
        pairs["expiration_call"]
        - pairs["expiration_put"]
    ).abs().dt.days

    pairs["expiration_distance"] = (
        pairs["expiration_distance"]
        .fillna(999)
    )

    # --------------------------------------------------------
    # REQUIRE ALIGNED EXPIRATION
    # --------------------------------------------------------

    pairs = pairs[
        (
            pairs["dte_distance"]
            <= MAX_DTE_DISTANCE
        )
        |
        (
            pairs["expiration_distance"]
            <= MAX_EXPIRATION_DISTANCE
        )
    ].copy()

    if pairs.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    pairs["current_price"] = (
        pairs["current_price_call"]
        .combine_first(
            pairs["current_price_put"]
        )
    )

    # --------------------------------------------------------
    # STRIKE RELATIONSHIP
    #
    # Bullish structure:
    #
    # CALL strike >= current price
    # PUT strike <= current price
    #
    # --------------------------------------------------------

    pairs["call_bullish_strike"] = (
        pairs["strike_call"]
        >= pairs["current_price"]
    )

    pairs["put_bullish_strike"] = (
        pairs["strike_put"]
        <= pairs["current_price"]
    )

    # --------------------------------------------------------
    # DELTA CONFIRMATION
    #
    # CALL BUY -> positive delta
    # PUT SELL -> negative delta
    # --------------------------------------------------------

    pairs["call_delta_bullish"] = (
        pairs["delta_call"]
        > 0
    )

    pairs["put_delta_bullish"] = (
        pairs["delta_put"]
        < 0
    )

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    pairs["call_premium_abs"] = (
        pairs["premium_call"]
        .abs()
        .fillna(0)
    )

    pairs["put_premium_abs"] = (
        pairs["premium_put"]
        .abs()
        .fillna(0)
    )

    pairs["combined_premium"] = (
        pairs["call_premium_abs"]
        + pairs["put_premium_abs"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    pairs["call_volume"] = (
        pairs["volume_call"]
        .fillna(0)
    )

    pairs["put_volume"] = (
        pairs["volume_put"]
        .fillna(0)
    )

    pairs["combined_volume"] = (
        pairs["call_volume"]
        + pairs["put_volume"]
    )

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

    pairs["call_oi"] = (
        pairs["open_interest_call"]
        .fillna(0)
    )

    pairs["put_oi"] = (
        pairs["open_interest_put"]
        .fillna(0)
    )

    pairs["combined_oi"] = (
        pairs["call_oi"]
        + pairs["put_oi"]
    )

    # ========================================================
    # COMPONENT SCORES
    # ========================================================

    pairs["premium_score"] = minmax_score(
        pairs["combined_premium"]
    )

    pairs["volume_score"] = minmax_score(
        pairs["combined_volume"]
    )

    pairs["flow_score_component"] = minmax_score(
        (
            pairs["flow_score_call"]
            .fillna(0)
            +
            pairs["flow_score_put"]
            .fillna(0)
        )
        / 2.0
    )

    # --------------------------------------------------------
    # EXPIRATION ALIGNMENT
    # --------------------------------------------------------

    pairs["expiry_alignment_score"] = np.where(
        pairs["expiration_distance"] <= 1,
        100.0,
        np.where(
            pairs["expiration_distance"] <= 3,
            80.0,
            np.where(
                pairs["expiration_distance"] <= 7,
                60.0,
                0.0,
            ),
        ),
    )

    # --------------------------------------------------------
    # DIRECTION CONFIRMATION
    # --------------------------------------------------------

    pairs["direction_score"] = (
        pairs["call_delta_bullish"]
        .astype(int)
        +
        pairs["put_delta_bullish"]
        .astype(int)
    ) * 50.0

    # --------------------------------------------------------
    # STRIKE CONFIRMATION
    # --------------------------------------------------------

    pairs["strike_score"] = (
        pairs["call_bullish_strike"]
        .astype(int)
        +
        pairs["put_bullish_strike"]
        .astype(int)
    ) * 50.0

    # ========================================================
    # SPECIAL SCORE
    #
    # 30% premium
    # 20% volume
    # 20% flow
    # 15% direction
    # 10% strike
    # 5% expiry
    # ========================================================

    pairs["special_score"] = (

        pairs["premium_score"]
        * 0.30

        +

        pairs["volume_score"]
        * 0.20

        +

        pairs["flow_score_component"]
        * 0.20

        +

        pairs["direction_score"]
        * 0.15

        +

        pairs["strike_score"]
        * 0.10

        +

        pairs["expiry_alignment_score"]
        * 0.05
    )

    # --------------------------------------------------------
    # STRUCTURE CONFIRMATION
    # --------------------------------------------------------

    pairs["structure_confirmation"] = (
        pairs["call_delta_bullish"]
        &
        pairs["put_delta_bullish"]
    )

    pairs["structure"] = (
        "CALL BUY + PUT SELL"
    )

    pairs["special_reason"] = np.where(
        pairs["structure_confirmation"],

        "CALL BUY confirmed | "
        "PUT SELL confirmed | "
        "Bullish delta confirmation",

        "CALL BUY confirmed | "
        "PUT SELL confirmed",
    )

    # --------------------------------------------------------
    # REQUIRE DELTA CONFIRMATION
    # --------------------------------------------------------

    pairs = pairs[
        pairs["structure_confirmation"]
    ].copy()

    if pairs.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    pairs = pairs.sort_values(
        [
            "ticker",
            "special_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    pairs = pairs.drop_duplicates(
        subset=[
            "ticker",
            "strike_call",
            "expiration_call",
            "strike_put",
            "expiration_put",
        ],
        keep="first",
    )

    # --------------------------------------------------------
    # GLOBAL RANK
    # --------------------------------------------------------

    pairs = pairs.sort_values(
        [
            "special_score",
            "combined_premium",
            "combined_volume",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # MAX ONE STRUCTURE PER TICKER
    # --------------------------------------------------------

    pairs = pairs.drop_duplicates(
        subset=["ticker"],
        keep="first",
    )

    pairs = pairs.head(
        MAX_RESULTS
    ).copy()

    pairs["special_rank"] = (
        np.arange(
            len(pairs)
        )
        + 1
    )

    return pairs


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [

    "special_rank",
    "ticker",
    "structure",
    "special_score",
    "current_price",

    "call_strike",
    "call_expiration",
    "call_DTE",
    "call_volume",
    "call_open_interest",
    "call_premium",
    "call_delta",
    "call_gamma",
    "call_vega",
    "call_iv",
    "call_flow_score",

    "put_strike",
    "put_expiration",
    "put_DTE",
    "put_volume",
    "put_open_interest",
    "put_premium",
    "put_delta",
    "put_gamma",
    "put_vega",
    "put_iv",
    "put_flow_score",

    "combined_premium",
    "combined_volume",
    "combined_oi",

    "dte_distance",
    "expiration_distance",

    "special_reason",
    "data_source",
]


# ============================================================
# EMPTY OUTPUT
# ============================================================

def create_empty_output():

    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


# ============================================================
# FORMAT OUTPUT
# ============================================================

def format_output(pairs):

    if pairs is None or pairs.empty:

        return create_empty_output()

    output = pd.DataFrame()

    output["special_rank"] = (
        pairs["special_rank"]
    )

    output["ticker"] = (
        pairs["ticker"]
    )

    output["structure"] = (
        pairs["structure"]
    )

    output["special_score"] = (
        pairs["special_score"]
        .round(2)
    )

    output["current_price"] = (
        pairs["current_price"]
    )

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    output["call_strike"] = (
        pairs["strike_call"]
    )

    output["call_expiration"] = (
        pairs["expiration_call"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output["call_DTE"] = (
        pairs["DTE_call"]
    )

    output["call_volume"] = (
        pairs["volume_call"]
    )

    output["call_open_interest"] = (
        pairs["open_interest_call"]
    )

    output["call_premium"] = (
        pairs["premium_call"]
    )

    output["call_delta"] = (
        pairs["delta_call"]
    )

    output["call_gamma"] = (
        pairs["gamma_call"]
    )

    output["call_vega"] = (
        pairs["vega_call"]
    )

    output["call_iv"] = (
        pairs["iv_call"]
    )

    output["call_flow_score"] = (
        pairs["flow_score_call"]
    )

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    output["put_strike"] = (
        pairs["strike_put"]
    )

    output["put_expiration"] = (
        pairs["expiration_put"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output["put_DTE"] = (
        pairs["DTE_put"]
    )

    output["put_volume"] = (
        pairs["volume_put"]
    )

    output["put_open_interest"] = (
        pairs["open_interest_put"]
    )

    output["put_premium"] = (
        pairs["premium_put"]
    )

    output["put_delta"] = (
        pairs["delta_put"]
    )

    output["put_gamma"] = (
        pairs["gamma_put"]
    )

    output["put_vega"] = (
        pairs["vega_put"]
    )

    output["put_iv"] = (
        pairs["iv_put"]
    )

    output["put_flow_score"] = (
        pairs["flow_score_put"]
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    output["combined_premium"] = (
        pairs["combined_premium"]
    )

    output["combined_volume"] = (
        pairs["combined_volume"]
    )

    output["combined_oi"] = (
        pairs["combined_oi"]
    )

    output["dte_distance"] = (
        pairs["dte_distance"]
    )

    output["expiration_distance"] = (
        pairs["expiration_distance"]
    )

    output["special_reason"] = (
        pairs["special_reason"]
    )

    output["data_source"] = (
        "CALCULATED_FROM_UNUSUAL_FLOW"
    )

    return output[
        OUTPUT_COLUMNS
    ]


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
            "오늘 확인된 "
            "CALL BUY + PUT SELL "
            "특수 구조가 없습니다."
        )

        lines.append("")

        lines.append(
            "※ 0건도 정상 결과입니다."
        )

        lines.append("")

        lines.append(
            "**SPECIAL LIST : 0**"
        )

        lines.append("")

        return "\n".join(lines)

    lines.append(
        f"**SPECIAL LIST : {len(df)}**"
    )

    lines.append("")

    for _, row in df.iterrows():

        rank = int(
            row["special_rank"]
        )

        ticker = row["ticker"]

        score = float(
            row["special_score"]
        )

        current_price = row[
            "current_price"
        ]

        lines.append(
            f"## #{rank} {ticker}"
        )

        lines.append("")

        lines.append(
            "**구조: CALL BUY + PUT SELL**"
        )

        lines.append(
            f"- Special Score: **{score:.2f}**"
        )

        if pd.notna(current_price):

            lines.append(
                f"- 현재가: **{current_price:.2f}**"
            )

        lines.append("")

        lines.append(
            "### CALL BUY"
        )

        lines.append(
            f"- Strike: "
            f"{row['call_strike']}"
        )

        lines.append(
            f"- Expiration: "
            f"{row['call_expiration']}"
        )

        lines.append(
            f"- DTE: "
            f"{row['call_DTE']}"
        )

        lines.append(
            f"- Volume: "
            f"{row['call_volume']}"
        )

        lines.append(
            f"- OI: "
            f"{row['call_open_interest']}"
        )

        lines.append(
            f"- Premium: "
            f"{row['call_premium']}"
        )

        lines.append(
            f"- Delta: "
            f"{row['call_delta']}"
        )

        lines.append("")

        lines.append(
            "### PUT SELL"
        )

        lines.append(
            f"- Strike: "
            f"{row['put_strike']}"
        )

        lines.append(
            f"- Expiration: "
            f"{row['put_expiration']}"
        )

        lines.append(
            f"- DTE: "
            f"{row['put_DTE']}"
        )

        lines.append(
            f"- Volume: "
            f"{row['put_volume']}"
        )

        lines.append(
            f"- OI: "
            f"{row['put_open_interest']}"
        )

        lines.append(
            f"- Premium: "
            f"{row['put_premium']}"
        )

        lines.append(
            f"- Delta: "
            f"{row['put_delta']}"
        )

        lines.append("")

        lines.append(
            "### Structure"
        )

        lines.append(
            f"- Combined Premium: "
            f"{row['combined_premium']}"
        )

        lines.append(
            f"- Combined Volume: "
            f"{row['combined_volume']}"
        )

        lines.append(
            f"- DTE Distance: "
            f"{row['dte_distance']}"
        )

        lines.append(
            f"- Expiration Distance: "
            f"{row['expiration_distance']}"
        )

        lines.append(
            f"- Reason: "
            f"{row['special_reason']}"
        )

        lines.append("")

        lines.append("---")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # INPUT CHECK
    # ========================================================

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    log(
        f"INPUT : {INPUT_FILE}"
    )

    # ========================================================
    # READ
    # ========================================================

    raw = pd.read_csv(
        INPUT_FILE
    )

    input_rows = len(raw)

    log(
        f"INPUT ROWS : {input_rows:,}"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    df = prepare_data(
        raw
    )

    log(
        f"PREPARED ROWS : {len(df):,}"
    )

    # ========================================================
    # VALIDATION OF SIDE DATA
    # ========================================================

    call_buy_rows = (
        (
            df["option_type"]
            == "CALL"
        )
        &
        (
            df["side"]
            == "BUY"
        )
    ).sum()

    put_sell_rows = (
        (
            df["option_type"]
            == "PUT"
        )
        &
        (
            df["side"]
            == "SELL"
        )
    ).sum()

    unknown_side_rows = (
        df["side"]
        == ""
    ).sum()

    log(
        f"CALL BUY ROWS : {call_buy_rows:,}"
    )

    log(
        f"PUT SELL ROWS : {put_sell_rows:,}"
    )

    log(
        f"UNKNOWN SIDE ROWS : "
        f"{unknown_side_rows:,}"
    )

    # ========================================================
    # SEARCH
    # ========================================================

    candidates = build_special_candidates(
        df
    )

    # ========================================================
    # FORMAT
    # ========================================================

    output = format_output(
        candidates
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    output.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ========================================================
    # SAVE MARKDOWN
    # ========================================================

    markdown = create_markdown(
        output
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            markdown
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "🔎 STEP 11 VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"INPUT ROWS       : "
        f"{input_rows:,}"
    )

    print(
        f"CALL BUY ROWS    : "
        f"{call_buy_rows:,}"
    )

    print(
        f"PUT SELL ROWS    : "
        f"{put_sell_rows:,}"
    )

    print(
        f"UNKNOWN SIDE     : "
        f"{unknown_side_rows:,}"
    )

    print(
        f"SPECIAL ROWS     : "
        f"{len(output):,}"
    )

    print(
        f"SPECIAL TICKERS  : "
        f"{output['ticker'].nunique():,}"
    )

    print(
        f"SCORE VALID      : "
        f"{output['special_score'].notna().sum():,}"
    )

    print()

    print(
        "STRUCTURE        : "
        "CALL BUY + PUT SELL"
    )

    print(
        "SIDE SOURCE      : "
        "unusual_flow.csv / "
        "trade_side"
    )

    print(
        "SIDE TYPE        : "
        "ESTIMATED_BID_ASK_LAST"
    )

    print()

    if not output.empty:

        print(
            "## SPECIAL PREVIEW"
        )

        print()

        preview_columns = [
            "special_rank",
            "ticker",
            "special_score",
            "current_price",
            "call_strike",
            "call_DTE",
            "put_strike",
            "put_DTE",
            "combined_premium",
        ]

        print(
            output[
                preview_columns
            ]
            .to_string(
                index=False
            )
        )

        print()

    print(
        f"CSV              : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"MARKDOWN         : "
        f"{OUTPUT_MD}"
    )

    print(
        "=" * 72
    )

    log(
        "STEP 11 SPECIAL LIST COMPLETE"
    )


if __name__ == "__main__":
    main()

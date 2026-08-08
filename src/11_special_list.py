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
#   data/analysis/unusual_flow.csv
#
# OUTPUT:
#   data/analysis/special_list.csv
#   data/analysis/special_list.md
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

MAX_DTE_DISTANCE = 7

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

    text = text.rstrip(".")

    buy_values = {
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
    }

    sell_values = {
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
    }

    if text in buy_values:
        return "BUY"

    if text in sell_values:
        return "SELL"

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
# SCORE
# ============================================================

def minmax_score(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    if len(values) == 0:

        return values

    minimum = values.min()
    maximum = values.max()

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
        /
        (maximum - minimum)
        * 100.0
    )


# ============================================================
# PREPARE DATA
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

    # IMPORTANT:
    #
    # STEP 5 currently generates:
    #
    # trade_side_estimate
    #
    # Therefore prefer it.
    #
    side_col = find_column(
        raw,
        [
            "trade_side_estimate",
            "trade_side",
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
            "Symbol column not found in unusual_flow.csv"
        )

    if option_type_col is None:
        raise ValueError(
            "Option type column not found in unusual_flow.csv"
        )

    if side_col is None:
        raise ValueError(
            "Trade side column not found in unusual_flow.csv"
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
# BUILD SPECIAL CANDIDATES
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
    # Valid ticker
    # --------------------------------------------------------

    calls = calls[
        calls["ticker"].ne("")
    ].copy()

    puts = puts[
        puts["ticker"].ne("")
    ].copy()

    # --------------------------------------------------------
    # Common tickers
    # --------------------------------------------------------

    common_tickers = sorted(
        set(calls["ticker"].unique())
        &
        set(puts["ticker"].unique())
    )

    log(
        f"COMMON TICKERS : "
        f"{len(common_tickers):,}"
    )

    if not common_tickers:

        return pd.DataFrame()

    calls = calls[
        calls["ticker"].isin(
            common_tickers
        )
    ].copy()

    puts = puts[
        puts["ticker"].isin(
            common_tickers
        )
    ].copy()

    # ========================================================
    # IMPORTANT FIX
    #
    # DO NOT use a merge where ticker becomes an ambiguous
    # column.
    #
    # Instead build candidates ticker-by-ticker.
    # ========================================================

    candidate_frames = []

    for ticker in common_tickers:

        call_group = calls[
            calls["ticker"] == ticker
        ].copy()

        put_group = puts[
            puts["ticker"] == ticker
        ].copy()

        if call_group.empty or put_group.empty:
            continue

        # ----------------------------------------------------
        # Rename every field BEFORE combining.
        #
        # This prevents the KeyError: 'ticker'
        # that occurred in the previous implementation.
        # ----------------------------------------------------

        call_group = call_group.rename(
            columns={
                "ticker": "call_ticker",
                "option_type": "call_option_type",
                "side": "call_side",
                "strike": "call_strike",
                "DTE": "call_DTE",
                "expiration": "call_expiration",
                "volume": "call_volume",
                "open_interest": "call_open_interest",
                "premium": "call_premium",
                "delta": "call_delta",
                "gamma": "call_gamma",
                "vega": "call_vega",
                "iv": "call_iv",
                "current_price": "call_current_price",
                "flow_score": "call_flow_score",
                "directional_premium": "call_directional_premium",
            }
        )

        put_group = put_group.rename(
            columns={
                "ticker": "put_ticker",
                "option_type": "put_option_type",
                "side": "put_side",
                "strike": "put_strike",
                "DTE": "put_DTE",
                "expiration": "put_expiration",
                "volume": "put_volume",
                "open_interest": "put_open_interest",
                "premium": "put_premium",
                "delta": "put_delta",
                "gamma": "put_gamma",
                "vega": "put_vega",
                "iv": "put_iv",
                "current_price": "put_current_price",
                "flow_score": "put_flow_score",
                "directional_premium": "put_directional_premium",
            }
        )

        # ----------------------------------------------------
        # Pair inside ticker
        # ----------------------------------------------------

        call_group["_join_key"] = 1
        put_group["_join_key"] = 1

        ticker_pairs = pd.merge(
            call_group,
            put_group,
            on="_join_key",
            how="inner",
            suffixes=(
                "_call",
                "_put",
            ),
        )

        if ticker_pairs.empty:
            continue

        ticker_pairs["ticker"] = ticker

        candidate_frames.append(
            ticker_pairs
        )

    if not candidate_frames:

        return pd.DataFrame()

    pairs = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    # ========================================================
    # DTE DISTANCE
    # ========================================================

    pairs["dte_distance"] = (
        pairs["call_DTE"]
        -
        pairs["put_DTE"]
    ).abs()

    pairs["dte_distance"] = (
        pairs["dte_distance"]
        .fillna(999)
    )

    # ========================================================
    # EXPIRATION DISTANCE
    # ========================================================

    pairs["expiration_distance"] = (
        pairs["call_expiration"]
        -
        pairs["put_expiration"]
    ).abs().dt.days

    pairs["expiration_distance"] = (
        pairs["expiration_distance"]
        .fillna(999)
    )

    # ========================================================
    # EXPIRATION FILTER
    # ========================================================

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

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    pairs["current_price"] = (
        pairs["call_current_price"]
        .combine_first(
            pairs["put_current_price"]
        )
    )

    # ========================================================
    # STRIKE RELATIONSHIP
    #
    # CALL BUY:
    #   strike >= current price
    #
    # PUT SELL:
    #   strike <= current price
    #
    # This describes a bullish directional structure.
    # ========================================================

    pairs["call_bullish_strike"] = (
        pairs["call_strike"]
        >=
        pairs["current_price"]
    )

    pairs["put_bullish_strike"] = (
        pairs["put_strike"]
        <=
        pairs["current_price"]
    )

    # ========================================================
    # DELTA CONFIRMATION
    # ========================================================

    pairs["call_delta_bullish"] = (
        pairs["call_delta"]
        > 0
    )

    pairs["put_delta_bullish"] = (
        pairs["put_delta"]
        < 0
    )

    # ========================================================
    # PREMIUM
    # ========================================================

    pairs["call_premium_abs"] = (
        pairs["call_premium"]
        .abs()
        .fillna(0)
    )

    pairs["put_premium_abs"] = (
        pairs["put_premium"]
        .abs()
        .fillna(0)
    )

    pairs["combined_premium"] = (
        pairs["call_premium_abs"]
        +
        pairs["put_premium_abs"]
    )

    # ========================================================
    # VOLUME
    # ========================================================

    pairs["call_volume"] = (
        pairs["call_volume"]
        .fillna(0)
    )

    pairs["put_volume"] = (
        pairs["put_volume"]
        .fillna(0)
    )

    pairs["combined_volume"] = (
        pairs["call_volume"]
        +
        pairs["put_volume"]
    )

    # ========================================================
    # OI
    # ========================================================

    pairs["call_oi"] = (
        pairs["call_open_interest"]
        .fillna(0)
    )

    pairs["put_oi"] = (
        pairs["put_open_interest"]
        .fillna(0)
    )

    pairs["combined_oi"] = (
        pairs["call_oi"]
        +
        pairs["put_oi"]
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
            pairs["call_flow_score"]
            .fillna(0)
            +
            pairs["put_flow_score"]
            .fillna(0)
        )
        / 2.0
    )

    # ========================================================
    # EXPIRATION SCORE
    # ========================================================

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

    # ========================================================
    # DIRECTION SCORE
    # ========================================================

    pairs["direction_score"] = (
        pairs["call_delta_bullish"]
        .astype(int)
        +
        pairs["put_delta_bullish"]
        .astype(int)
    ) * 50.0

    # ========================================================
    # STRIKE SCORE
    # ========================================================

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
    # Premium       30%
    # Volume        20%
    # Flow          20%
    # Direction     15%
    # Strike        10%
    # Expiration     5%
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

    # ========================================================
    # STRUCTURE CONFIRMATION
    # ========================================================

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

    # ========================================================
    # REQUIRE DELTA CONFIRMATION
    # ========================================================

    pairs = pairs[
        pairs["structure_confirmation"]
    ].copy()

    if pairs.empty:

        return pd.DataFrame()

    # ========================================================
    # REMOVE INVALID SCORE
    # ========================================================

    pairs = pairs[
        pairs["special_score"].notna()
    ].copy()

    if pairs.empty:

        return pd.DataFrame()

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

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
            "call_strike",
            "call_expiration",
            "put_strike",
            "put_expiration",
        ],
        keep="first",
    )

    # ========================================================
    # GLOBAL RANK
    # ========================================================

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

    # ========================================================
    # MAX ONE STRUCTURE PER TICKER
    # ========================================================

    pairs = pairs.drop_duplicates(
        subset=[
            "ticker"
        ],
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
        pairs["call_strike"]
    )

    output["call_expiration"] = (
        pairs["call_expiration"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output["call_DTE"] = (
        pairs["call_DTE"]
    )

    output["call_volume"] = (
        pairs["call_volume"]
    )

    output["call_open_interest"] = (
        pairs["call_open_interest"]
    )

    output["call_premium"] = (
        pairs["call_premium"]
    )

    output["call_delta"] = (
        pairs["call_delta"]
    )

    output["call_gamma"] = (
        pairs["call_gamma"]
    )

    output["call_vega"] = (
        pairs["call_vega"]
    )

    output["call_iv"] = (
        pairs["call_iv"]
    )

    output["call_flow_score"] = (
        pairs["call_flow_score"]
    )

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    output["put_strike"] = (
        pairs["put_strike"]
    )

    output["put_expiration"] = (
        pairs["put_expiration"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output["put_DTE"] = (
        pairs["put_DTE"]
    )

    output["put_volume"] = (
        pairs["put_volume"]
    )

    output["put_open_interest"] = (
        pairs["put_open_interest"]
    )

    output["put_premium"] = (
        pairs["put_premium"]
    )

    output["put_delta"] = (
        pairs["put_delta"]
    )

    output["put_gamma"] = (
        pairs["put_gamma"]
    )

    output["put_vega"] = (
        pairs["put_vega"]
    )

    output["put_iv"] = (
        pairs["put_iv"]
    )

    output["put_flow_score"] = (
        pairs["put_flow_score"]
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

        current_price = (
            row["current_price"]
        )

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
            f"- Strike: {row['call_strike']}"
        )

        lines.append(
            f"- Expiration: "
            f"{row['call_expiration']}"
        )

        lines.append(
            f"- DTE: {row['call_DTE']}"
        )

        lines.append(
            f"- Volume: {row['call_volume']}"
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
            f"- Strike: {row['put_strike']}"
        )

        lines.append(
            f"- Expiration: "
            f"{row['put_expiration']}"
        )

        lines.append(
            f"- DTE: {row['put_DTE']}"
        )

        lines.append(
            f"- Volume: {row['put_volume']}"
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
            f"- Combined OI: "
            f"{row['combined_oi']}"
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

    log(
        f"INPUT COLUMNS : {len(raw.columns):,}"
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
    # VALIDATION
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
        f"CALL BUY ROWS : "
        f"{call_buy_rows:,}"
    )

    log(
        f"PUT SELL ROWS : "
        f"{put_sell_rows:,}"
    )

    log(
        f"UNKNOWN SIDE ROWS : "
        f"{unknown_side_rows:,}"
    )

    # ========================================================
    # SEARCH
    # ========================================================

    candidates = (
        build_special_candidates(
            df
        )
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
    # FINAL VALIDATION
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
        "trade_side_estimate"
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
            ].to_string(
                index=False
            )
        )

        print()

    else:

        print(
            "SPECIAL PREVIEW : 0 ROWS"
        )

        print(
            "NO VALID CALL BUY + PUT SELL STRUCTURE"
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

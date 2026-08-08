import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# STEP 11 - CALL BUY + PUT SELL STRUCTURE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

DECISION_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "call_buy_put_sell.csv"
)


# ============================================================
# CONFIG
# ============================================================

MIN_VOLUME = 1
MIN_OI = 1

MIN_STRUCTURE_SCORE = 45.0

# Same expiration / strike proximity rules
MAX_STRIKE_DISTANCE_PCT = 0.10

# Premium approximation
CONTRACT_MULTIPLIER = 100


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
        f"[11 CALL+PUT] {now} | {message}"
    )


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(df, candidates):

    normalized = {
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_"): col
        for col in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# NUMERIC
# ============================================================

def numeric(df, column):

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
# TEXT NORMALIZATION
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
    )


# ============================================================
# SIDE NORMALIZATION
# ============================================================

def normalize_side(value):

    text = clean_text(value)

    if not text:
        return "UNKNOWN"

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    buy_patterns = [
        "BUY",
        "BTO",
        "BUY TO OPEN",
        "BOT",
        "BOUGHT",
        "LONG",
        "ASK",
        "AT ASK"
    ]

    for pattern in buy_patterns:

        if pattern in text:
            return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    sell_patterns = [
        "SELL",
        "STO",
        "SELL TO OPEN",
        "SOLD",
        "SHORT",
        "BID",
        "AT BID"
    ]

    for pattern in sell_patterns:

        if pattern in text:
            return "SELL"

    return "UNKNOWN"


# ============================================================
# OPTION TYPE NORMALIZATION
# ============================================================

def normalize_option_type(value):

    text = clean_text(value)

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

    return "UNKNOWN"


# ============================================================
# TICKER
# ============================================================

def ticker_column(df):

    return find_column(
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


# ============================================================
# PREPARE GREEKS
# ============================================================

def prepare_greeks(df):

    ticker_col = ticker_column(df)

    type_col = find_column(
        df,
        [
            "option_type",
            "type",
            "contract_type",
            "call_put"
        ]
    )

    strike_col = find_column(
        df,
        [
            "strike",
            "strike_price"
        ]
    )

    expiration_col = find_column(
        df,
        [
            "expiration",
            "expiration_date",
            "expiry",
            "expirationdate",
            "exp_date"
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

    price_col = find_column(
        df,
        [
            "underlying_price",
            "current_price",
            "stock_price",
            "underlyingPrice"
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

    bid_col = find_column(
        df,
        [
            "bid"
        ]
    )

    ask_col = find_column(
        df,
        [
            "ask"
        ]
    )

    last_col = find_column(
        df,
        [
            "last",
            "last_price",
            "lastPrice"
        ]
    )

    if ticker_col is None:
        raise ValueError(
            "Ticker column not found in options_greeks.csv"
        )

    if type_col is None:
        raise ValueError(
            "Option type column not found"
        )

    if strike_col is None:
        raise ValueError(
            "Strike column not found"
        )

    result = pd.DataFrame(index=df.index)

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["option_type"] = (
        df[type_col]
        .apply(normalize_option_type)
    )

    result["strike"] = numeric(
        df,
        strike_col
    )

    result["expiration"] = (
        df[expiration_col]
        .astype(str).str.strip()
        if expiration_col is not None
        else ""
    )

    result["DTE"] = numeric(
        df,
        dte_col
    )

    result["underlying_price"] = numeric(
        df,
        price_col
    )

    result["volume"] = numeric(
        df,
        volume_col
    ).fillna(0)

    result["open_interest"] = numeric(
        df,
        oi_col
    ).fillna(0)

    result["delta"] = numeric(
        df,
        delta_col
    )

    result["iv"] = numeric(
        df,
        iv_col
    )

    result["bid"] = numeric(
        df,
        bid_col
    )

    result["ask"] = numeric(
        df,
        ask_col
    )

    result["last"] = numeric(
        df,
        last_col
    )

    # --------------------------------------------------------
    # Premium approximation
    # --------------------------------------------------------

    price = result["last"]

    price = price.fillna(
        (
            result["bid"]
            + result["ask"]
        ) / 2.0
    )

    result["premium"] = (
        price.fillna(0)
        * result["volume"]
        * CONTRACT_MULTIPLIER
    )

    return result


# ============================================================
# PREPARE FLOW
# ============================================================

def prepare_flow(df):

    ticker_col = ticker_column(df)

    if ticker_col is None:
        raise ValueError(
            "Ticker column not found in unusual_flow.csv"
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
            "direction",
            "transaction_type",
            "trade_type",
            "order_type",
            "flow_type",
            "sentiment"
        ]
    )

    strike_col = find_column(
        df,
        [
            "strike",
            "strike_price"
        ]
    )

    expiration_col = find_column(
        df,
        [
            "expiration",
            "expiration_date",
            "expiry",
            "expirationdate",
            "exp_date"
        ]
    )

    volume_col = find_column(
        df,
        [
            "volume",
            "option_volume",
            "trade_volume"
        ]
    )

    premium_col = find_column(
        df,
        [
            "premium",
            "estimated_premium",
            "estimated_traded_premium",
            "premium_flow",
            "total_premium"
        ]
    )

    result = pd.DataFrame(index=df.index)

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if type_col is not None:

        result["option_type"] = (
            df[type_col]
            .apply(normalize_option_type)
        )

    else:

        result["option_type"] = "UNKNOWN"

    if side_col is not None:

        result["side"] = (
            df[side_col]
            .apply(normalize_side)
        )

    else:

        result["side"] = "UNKNOWN"

    result["strike"] = numeric(
        df,
        strike_col
    )

    result["expiration"] = (
        df[expiration_col]
        .astype(str).str.strip()
        if expiration_col is not None
        else ""
    )

    result["volume"] = numeric(
        df,
        volume_col
    ).fillna(0)

    result["premium"] = numeric(
        df,
        premium_col
    ).fillna(0)

    return result


# ============================================================
# TOP20
# ============================================================

def load_top20():

    df = pd.read_csv(
        TOP20_FILE
    )

    column = ticker_column(df)

    if column is None:
        raise ValueError(
            "Unable to identify TOP20 ticker column"
        )

    return (
        df[column]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# DECISION
# ============================================================

def load_decisions():

    if not os.path.exists(
        DECISION_FILE
    ):
        return {}

    df = pd.read_csv(
        DECISION_FILE
    )

    ticker_col = ticker_column(df)

    decision_col = find_column(
        df,
        [
            "decision",
            "final_decision"
        ]
    )

    score_col = find_column(
        df,
        [
            "decision_score",
            "score"
        ]
    )

    if ticker_col is None:
        return {}

    result = {}

    for _, row in df.iterrows():

        ticker = clean_text(
            row[ticker_col]
        )

        result[ticker] = {
            "decision": (
                clean_text(row[decision_col])
                if decision_col is not None
                else ""
            ),
            "decision_score": (
                float(row[score_col])
                if score_col is not None
                and pd.notna(row[score_col])
                else np.nan
            )
        }

    return result


# ============================================================
# STRUCTURE SCORE
# ============================================================

def calculate_structure_score(
    call_volume,
    put_volume,
    call_premium,
    put_premium,
    call_delta,
    put_delta,
    same_expiration
):

    score = 0.0

    # --------------------------------------------------------
    # Volume balance
    # --------------------------------------------------------

    total_volume = (
        call_volume
        + put_volume
    )

    if total_volume > 0:

        buy_share = (
            call_volume
            / total_volume
        )

        sell_share = (
            put_volume
            / total_volume
        )

        score += (
            min(
                buy_share,
                1.0
            ) * 20
        )

        score += (
            min(
                sell_share,
                1.0
            ) * 20
        )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    if call_premium > 0:
        score += 15

    if put_premium > 0:
        score += 15

    # --------------------------------------------------------
    # Delta confirmation
    # --------------------------------------------------------

    if not pd.isna(call_delta):

        if call_delta > 0:
            score += 5

    if not pd.isna(put_delta):

        if put_delta < 0:
            score += 5

    # --------------------------------------------------------
    # Same expiration
    # --------------------------------------------------------

    if same_expiration:
        score += 5

    return min(
        100.0,
        score
    )


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(
    greeks,
    flow,
    top20,
    decisions
):

    rows = []

    # --------------------------------------------------------
    # Flow must contain actual direction information
    # --------------------------------------------------------

    flow_direction_available = (
        (flow["side"] == "BUY").any()
        or
        (flow["side"] == "SELL").any()
    )

    print()
    print(
        "FLOW DIRECTION CHECK"
    )
    print("----------------------------------------")

    print(
        f"BUY ROWS  : "
        f"{(flow['side'] == 'BUY').sum()}"
    )

    print(
        f"SELL ROWS : "
        f"{(flow['side'] == 'SELL').sum()}"
    )

    print(
        f"UNKNOWN   : "
        f"{(flow['side'] == 'UNKNOWN').sum()}"
    )

    if not flow_direction_available:

        log(
            "WARNING: No explicit BUY/SELL "
            "direction detected in unusual_flow.csv"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Process TOP20
    # --------------------------------------------------------

    for rank, ticker in enumerate(
        top20,
        start=1
    ):

        ticker_flow = flow[
            flow["ticker"] == ticker
        ].copy()

        if ticker_flow.empty:
            continue

        call_buy = ticker_flow[
            (
                ticker_flow["option_type"]
                == "CALL"
            )
            &
            (
                ticker_flow["side"]
                == "BUY"
            )
        ].copy()

        put_sell = ticker_flow[
            (
                ticker_flow["option_type"]
                == "PUT"
            )
            &
            (
                ticker_flow["side"]
                == "SELL"
            )
        ].copy()

        if call_buy.empty:
            continue

        if put_sell.empty:
            continue

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        call_volume = float(
            call_buy["volume"].sum()
        )

        put_volume = float(
            put_sell["volume"].sum()
        )

        call_premium = float(
            call_buy["premium"].sum()
        )

        put_premium = float(
            put_sell["premium"].sum()
        )

        # ----------------------------------------------------
        # Greeks lookup
        # ----------------------------------------------------

        ticker_greeks = greeks[
            greeks["ticker"] == ticker
        ].copy()

        call_greeks = ticker_greeks[
            ticker_greeks["option_type"]
            == "CALL"
        ]

        put_greeks = ticker_greeks[
            ticker_greeks["option_type"]
            == "PUT"
        ]

        call_delta = (
            call_greeks["delta"].mean()
            if not call_greeks.empty
            else np.nan
        )

        put_delta = (
            put_greeks["delta"].mean()
            if not put_greeks.empty
            else np.nan
        )

        # ----------------------------------------------------
        # Expiration overlap
        # ----------------------------------------------------

        call_exp = set(
            call_buy[
                "expiration"
            ]
            .replace("", np.nan)
            .dropna()
            .astype(str)
        )

        put_exp = set(
            put_sell[
                "expiration"
            ]
            .replace("", np.nan)
            .dropna()
            .astype(str)
        )

        common_exp = (
            call_exp & put_exp
        )

        same_expiration = (
            len(common_exp) > 0
        )

        # ----------------------------------------------------
        # Strike ranges
        # ----------------------------------------------------

        call_strikes = (
            call_buy["strike"]
            .dropna()
        )

        put_strikes = (
            put_sell["strike"]
            .dropna()
        )

        call_strike = (
            float(call_strikes.median())
            if not call_strikes.empty
            else np.nan
        )

        put_strike = (
            float(put_strikes.median())
            if not put_strikes.empty
            else np.nan
        )

        # ----------------------------------------------------
        # Structure score
        # ----------------------------------------------------

        structure_score = (
            calculate_structure_score(
                call_volume,
                put_volume,
                call_premium,
                put_premium,
                call_delta,
                put_delta,
                same_expiration
            )
        )

        if (
            structure_score
            < MIN_STRUCTURE_SCORE
        ):
            continue

        decision = decisions.get(
            ticker,
            {}
        )

        rows.append({

            "rank": rank,

            "ticker": ticker,

            "decision": decision.get(
                "decision",
                ""
            ),

            "decision_score": decision.get(
                "decision_score",
                np.nan
            ),

            "call_buy_volume":
                call_volume,

            "put_sell_volume":
                put_volume,

            "call_buy_premium":
                call_premium,

            "put_sell_premium":
                put_premium,

            "call_buy_strike":
                call_strike,

            "put_sell_strike":
                put_strike,

            "call_delta":
                call_delta,

            "put_delta":
                put_delta,

            "same_expiration":
                same_expiration,

            "common_expiration":
                "|".join(
                    sorted(common_exp)
                ),

            "structure_score":
                structure_score,

            "structure":
                "CALL BUY + PUT SELL",

            "data_source":
                "CALCULATED + FLOW",

        })

        log(
            f"{ticker} | "
            f"CALL BUY {call_volume:.0f} | "
            f"PUT SELL {put_volume:.0f} | "
            f"SCORE {structure_score:.1f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    required_files = {
        "GREEKS": GREEKS_FILE,
        "FLOW": FLOW_FILE,
        "TOP20": TOP20_FILE,
        "DECISION": DECISION_FILE,
    }

    print()
    print("=" * 72)
    print("STEP 11 INPUT CHECK")
    print("=" * 72)

    for name, path in required_files.items():

        exists = os.path.exists(path)

        print(
            f"{name:<12} : "
            f"{'OK' if exists else 'MISSING'}"
        )

        if not exists:

            raise FileNotFoundError(
                f"{name} file not found: {path}"
            )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    greeks_raw = pd.read_csv(
        GREEKS_FILE
    )

    flow_raw = pd.read_csv(
        FLOW_FILE
    )

    top20 = load_top20()

    decisions = load_decisions()

    log(
        f"GREEKS ROWS : "
        f"{len(greeks_raw):,}"
    )

    log(
        f"FLOW ROWS   : "
        f"{len(flow_raw):,}"
    )

    log(
        f"TOP20       : "
        f"{len(top20)}"
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    greeks = prepare_greeks(
        greeks_raw
    )

    flow = prepare_flow(
        flow_raw
    )

    # --------------------------------------------------------
    # FILTER TOP20
    # --------------------------------------------------------

    greeks = greeks[
        greeks["ticker"].isin(
            top20
        )
    ].copy()

    flow = flow[
        flow["ticker"].isin(
            top20
        )
    ].copy()

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    output = build_candidates(
        greeks,
        flow,
        top20,
        decisions
    )

    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if output.empty:

        print()
        print("=" * 72)
        print(
            "STEP 11 RESULT"
        )
        print("=" * 72)

        print(
            "CALL BUY + PUT SELL "
            "STRUCTURE : NONE"
        )

        print(
            "This is a valid result."
        )

        # Empty CSV with schema
        output = pd.DataFrame(
            columns=[
                "rank",
                "ticker",
                "decision",
                "decision_score",
                "call_buy_volume",
                "put_sell_volume",
                "call_buy_premium",
                "put_sell_premium",
                "call_buy_strike",
                "put_sell_strike",
                "call_delta",
                "put_delta",
                "same_expiration",
                "common_expiration",
                "structure_score",
                "structure",
                "data_source",
            ]
        )

    else:

        output = (
            output
            .sort_values(
                [
                    "structure_score",
                    "decision_score"
                ],
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        output["final_rank"] = (
            output.index + 1
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("🔎 STEP 11 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT GREEKS ROWS : "
        f"{len(greeks_raw):,}"
    )

    print(
        f"INPUT FLOW ROWS   : "
        f"{len(flow_raw):,}"
    )

    print(
        f"TOP20 TICKERS     : "
        f"{len(top20)}"
    )

    print(
        f"STRUCTURE ROWS    : "
        f"{len(output)}"
    )

    if not output.empty:

        print(
            f"UNIQUE TICKERS    : "
            f"{output['ticker'].nunique()}"
        )

        print(
            f"STRUCTURE VALID   : "
            f"{output['structure'].notna().sum()}"
        )

        print(
            f"SCORE VALID       : "
            f"{output['structure_score'].notna().sum()}"
        )

        print()
        print(
            "CALL BUY + PUT SELL PREVIEW"
        )

        print(
            "----------------------------------------"
        )

        print(
            output[
                [
                    "final_rank",
                    "ticker",
                    "decision",
                    "call_buy_volume",
                    "put_sell_volume",
                    "call_buy_premium",
                    "put_sell_premium",
                    "structure_score",
                    "structure"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "NO MATCHING STRUCTURE"
        )

    print()
    print(
        "OUTPUT FILE : "
        "data/analysis/call_buy_put_sell.csv"
    )

    print("=" * 72)

    log(
        "STEP 11 CALL BUY + PUT SELL COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

import os
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/analysis/options_greeks.csv"

OUTPUT_DIR = "data/analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "unusual_flow.csv"
)

# Minimum values used to reduce noise.
MIN_VOLUME = 1
MIN_PREMIUM = 0.0


# ============================================================
# SCORE WEIGHTS
# ============================================================

WEIGHT_VOLUME = 20
WEIGHT_VOLUME_OI = 20
WEIGHT_PREMIUM = 25
WEIGHT_DELTA = 10
WEIGHT_GAMMA = 10
WEIGHT_IV = 5
WEIGHT_DTE = 5
WEIGHT_MONEYNESS = 5


# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[05 FLOW] {now} | {message}"
    )


# ============================================================
# SAFE NUMERIC
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# NORMALIZE OPTION TYPE
# ============================================================

def normalize_option_type(value):

    value = str(value).upper().strip()

    if value in {"CALL", "C"}:
        return "CALL"

    if value in {"PUT", "P"}:
        return "PUT"

    return "UNKNOWN"


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def percentile_score(series):

    series = numeric(series)

    if series.notna().sum() <= 1:
        return pd.Series(
            0.5,
            index=series.index
        )

    rank = series.rank(
        pct=True,
        method="average"
    )

    return rank.fillna(0.0)


# ============================================================
# MONEYNESS
# ============================================================

def calculate_moneyness(row):

    try:

        price = float(
            row["underlying_price"]
        )

        strike = float(
            row["strike"]
        )

        if price <= 0 or strike <= 0:
            return np.nan

        return (
            (strike - price)
            / price
        )

    except Exception:

        return np.nan


# ============================================================
# FLOW DIRECTION ESTIMATE
#
# IMPORTANT:
# This is NOT actual exchange trade-side data.
#
# It is estimated from:
#   bid
#   ask
#   lastPrice
#
# Priority:
#
# 1. Last near ask  -> BUY EST.
# 2. Last near bid  -> SELL EST.
# 3. Last above mid -> BUY EST.
# 4. Last below mid -> SELL EST.
# 5. Exact/near mid -> UNKNOWN
#
# The result MUST remain labelled ESTIMATED.
# ============================================================

def estimate_trade_side(row):

    try:

        bid = float(row["bid"])
        ask = float(row["ask"])
        last = float(row["lastPrice"])

        if not np.isfinite(bid):
            return "UNKNOWN"

        if not np.isfinite(ask):
            return "UNKNOWN"

        if not np.isfinite(last):
            return "UNKNOWN"

        if ask <= 0:
            return "UNKNOWN"

        if bid < 0:
            return "UNKNOWN"

        if ask < bid:
            return "UNKNOWN"

        spread = ask - bid

        # ----------------------------------------------------
        # Zero spread
        # ----------------------------------------------------

        if spread == 0:

            if abs(last - ask) <= 0.01:
                return "BUY EST."

            return "UNKNOWN"

        # ----------------------------------------------------
        # Last outside quoted market
        # ----------------------------------------------------

        if last > ask:
            return "BUY EST."

        if last < bid:
            return "SELL EST."

        # ----------------------------------------------------
        # Distance from bid / ask
        # ----------------------------------------------------

        distance_to_bid = abs(
            last - bid
        )

        distance_to_ask = abs(
            ask - last
        )

        # 25% of spread is used for edge classification.
        edge_tolerance = max(
            spread * 0.25,
            0.01
        )

        if distance_to_ask <= edge_tolerance:
            return "BUY EST."

        if distance_to_bid <= edge_tolerance:
            return "SELL EST."

        # ----------------------------------------------------
        # Relative position inside spread
        #
        # This is intentionally conservative.
        #
        # Clearly above the midpoint -> BUY EST.
        # Clearly below the midpoint -> SELL EST.
        # Near midpoint -> UNKNOWN.
        # ----------------------------------------------------

        midpoint = (
            bid + ask
        ) / 2.0

        midpoint_tolerance = max(
            spread * 0.10,
            0.01
        )

        if last > (
            midpoint
            + midpoint_tolerance
        ):
            return "BUY EST."

        if last < (
            midpoint
            - midpoint_tolerance
        ):
            return "SELL EST."

        return "UNKNOWN"

    except Exception:

        return "UNKNOWN"


# ============================================================
# OPEN / CLOSE ESTIMATE
#
# IMPORTANT:
# Volume/OI alone cannot prove OPEN or CLOSE.
#
# This remains an ESTIMATE only.
# ============================================================

def estimate_open_close(row):

    try:

        volume = float(
            row["volume"]
        )

        oi = float(
            row["openInterest"]
        )

        side = row[
            "trade_side_estimate"
        ]

        if not np.isfinite(volume):
            return "UNKNOWN"

        if not np.isfinite(oi):
            return "UNKNOWN"

        if volume <= 0:
            return "UNKNOWN"

        if side not in {
            "BUY EST.",
            "SELL EST."
        }:
            return "UNKNOWN"

        # New contracts with zero OI:
        # reasonable OPEN estimate.
        if oi <= 0:
            return "OPEN EST."

        ratio = (
            volume / oi
        )

        # Very high volume relative to OI.
        if ratio >= 1.0:
            return "OPEN EST."

        # Otherwise we cannot reliably determine
        # open versus close from this dataset.
        return "UNKNOWN"

    except Exception:

        return "UNKNOWN"


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

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

    df = pd.read_csv(
        INPUT_FILE
    )

    input_rows = len(df)

    log(
        f"INPUT ROWS : {input_rows:,}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "symbol",
        "strike",
        "DTE",
        "option_type",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "delta",
        "gamma",
        "vega",
        "underlying_price",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    numeric_columns = [
        "strike",
        "DTE",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "delta",
        "gamma",
        "vega",
        "underlying_price",
    ]

    for column in numeric_columns:

        df[column] = numeric(
            df[column]
        )

    # --------------------------------------------------------
    # OPTION TYPE
    # --------------------------------------------------------

    df["option_type"] = (
        df["option_type"]
        .apply(
            normalize_option_type
        )
    )

    # --------------------------------------------------------
    # PREMIUM
    #
    # Estimated traded premium:
    #
    # Volume × Mid Price × 100
    #
    # This is NOT actual net institutional flow.
    # --------------------------------------------------------

    df["mid_price"] = (
        df["bid"]
        + df["ask"]
    ) / 2.0

    df["estimated_traded_premium"] = (
        df["volume"]
        * df["mid_price"]
        * 100
    )

    df["premium_source"] = (
        "CALCULATED"
    )

    # --------------------------------------------------------
    # VOLUME / OI
    # --------------------------------------------------------

    df["volume_oi_ratio"] = np.where(
        df["openInterest"] > 0,

        df["volume"]
        / df["openInterest"],

        np.nan
    )

    # --------------------------------------------------------
    # MONEYNESS
    # --------------------------------------------------------

    df["moneyness"] = (
        df.apply(
            calculate_moneyness,
            axis=1
        )
    )

    # --------------------------------------------------------
    # TRADE SIDE ESTIMATE
    # --------------------------------------------------------

    log(
        "ESTIMATING BUY / SELL"
    )

    df["trade_side_estimate"] = (
        df.apply(
            estimate_trade_side,
            axis=1
        )
    )

    df["trade_side_source"] = (
        "ESTIMATED_BID_ASK_LAST"
    )

    # --------------------------------------------------------
    # OPEN / CLOSE ESTIMATE
    # --------------------------------------------------------

    log(
        "ESTIMATING OPEN / CLOSE"
    )

    df["open_close_estimate"] = (
        df.apply(
            estimate_open_close,
            axis=1
        )
    )

    df["open_close_source"] = (
        "ESTIMATED_VOLUME_OI"
    )

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    log(
        "CALCULATING FLOW COMPONENT SCORES"
    )

    volume_score = percentile_score(
        np.log1p(
            df["volume"].clip(
                lower=0
            )
        )
    )

    volume_oi_score = percentile_score(
        np.log1p(
            df["volume_oi_ratio"]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .clip(
                lower=0
            )
        )
    )

    premium_score = percentile_score(
        np.log1p(
            df["estimated_traded_premium"]
            .clip(
                lower=0
            )
        )
    )

    delta_score = percentile_score(
        df["delta"].abs()
    )

    gamma_score = percentile_score(
        df["gamma"].abs()
    )

    iv_score = percentile_score(
        df["impliedVolatility"]
    )

    # --------------------------------------------------------
    # DTE SCORE
    # --------------------------------------------------------

    dte_inverse = 1.0 / (
        df["DTE"].clip(
            lower=1
        )
    )

    dte_score = percentile_score(
        dte_inverse
    )

    # --------------------------------------------------------
    # MONEYNESS SCORE
    # --------------------------------------------------------

    moneyness_distance = (
        df["moneyness"].abs()
    )

    moneyness_score = percentile_score(
        -moneyness_distance
    )

    # --------------------------------------------------------
    # FINAL OPTION FLOW SCORE
    # --------------------------------------------------------

    df["flow_score"] = (

        volume_score
        * WEIGHT_VOLUME

        + volume_oi_score
        * WEIGHT_VOLUME_OI

        + premium_score
        * WEIGHT_PREMIUM

        + delta_score
        * WEIGHT_DELTA

        + gamma_score
        * WEIGHT_GAMMA

        + iv_score
        * WEIGHT_IV

        + dte_score
        * WEIGHT_DTE

        + moneyness_score
        * WEIGHT_MONEYNESS
    )

    # --------------------------------------------------------
    # CALL / PUT FLOW
    # --------------------------------------------------------

    df["call_put_flow"] = np.where(

        df["option_type"] == "CALL",

        df["estimated_traded_premium"],

        np.where(
            df["option_type"] == "PUT",

            -df[
                "estimated_traded_premium"
            ],

            0.0
        )
    )

    # --------------------------------------------------------
    # BUY / SELL PREMIUM ESTIMATE
    # --------------------------------------------------------

    df["estimated_directional_premium"] = np.where(

        df["trade_side_estimate"]
        == "BUY EST.",

        df[
            "estimated_traded_premium"
        ],

        np.where(

            df["trade_side_estimate"]
            == "SELL EST.",

            -df[
                "estimated_traded_premium"
            ],

            0.0
        )
    )

    # --------------------------------------------------------
    # FLOW RANK
    # --------------------------------------------------------

    df = df.sort_values(
        "flow_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    df["flow_rank"] = (
        np.arange(
            1,
            len(df) + 1
        )
    )

    # --------------------------------------------------------
    # SYMBOL SUMMARY
    # --------------------------------------------------------

    log(
        "BUILDING SYMBOL FLOW SUMMARY"
    )

    symbol_summary = (
        df.groupby(
            "symbol",
            dropna=False
        )
        .agg(
            option_count=(
                "symbol",
                "size"
            ),

            total_volume=(
                "volume",
                "sum"
            ),

            total_premium=(
                "estimated_traded_premium",
                "sum"
            ),

            call_premium=(
                "estimated_traded_premium",
                lambda x:
                x[
                    df.loc[
                        x.index,
                        "option_type"
                    ] == "CALL"
                ].sum()
            ),

            put_premium=(
                "estimated_traded_premium",
                lambda x:
                x[
                    df.loc[
                        x.index,
                        "option_type"
                    ] == "PUT"
                ].sum()
            ),

            max_flow_score=(
                "flow_score",
                "max"
            ),

            avg_flow_score=(
                "flow_score",
                "mean"
            ),
        )
        .reset_index()
    )

    symbol_summary[
        "call_put_premium_imbalance"
    ] = np.where(

        symbol_summary[
            "call_premium"
        ]
        + symbol_summary[
            "put_premium"
        ] > 0,

        (
            symbol_summary[
                "call_premium"
            ]
            - symbol_summary[
                "put_premium"
            ]
        )
        /
        (
            symbol_summary[
                "call_premium"
            ]
            + symbol_summary[
                "put_premium"
            ]
        ),

        0.0
    )

    # --------------------------------------------------------
    # SYMBOL FLOW SCORE
    # --------------------------------------------------------

    symbol_summary[
        "symbol_flow_score"
    ] = (

        percentile_score(
            symbol_summary[
                "total_premium"
            ]
        ) * 40

        + percentile_score(
            symbol_summary[
                "total_volume"
            ]
        ) * 20

        + percentile_score(
            symbol_summary[
                "max_flow_score"
            ]
        ) * 25

        + (
            symbol_summary[
                "call_put_premium_imbalance"
            ]
            .abs()
            * 15
        )
    )

    symbol_summary = (
        symbol_summary
        .sort_values(
            "symbol_flow_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    symbol_summary[
        "symbol_flow_rank"
    ] = np.arange(
        1,
        len(symbol_summary) + 1
    )

    # --------------------------------------------------------
    # MERGE SYMBOL DATA
    # --------------------------------------------------------

    df = df.merge(

        symbol_summary[
            [
                "symbol",
                "option_count",
                "total_volume",
                "total_premium",
                "call_premium",
                "put_premium",
                "call_put_premium_imbalance",
                "symbol_flow_score",
                "symbol_flow_rank",
            ]
        ],

        on="symbol",

        how="left",
    )

    # --------------------------------------------------------
    # TOP REASON
    # --------------------------------------------------------

    def reason(row):

        reasons = []

        if row[
            "volume_oi_ratio"
        ] >= 1:

            reasons.append(
                "Volume/OI surge"
            )

        if row[
            "estimated_traded_premium"
        ] > 0:

            reasons.append(
                "Premium activity"
            )

        if abs(
            row[
                "call_put_premium_imbalance"
            ]
        ) >= 0.25:

            if (
                row[
                    "call_put_premium_imbalance"
                ] > 0
            ):

                reasons.append(
                    "Call premium dominance"
                )

            else:

                reasons.append(
                    "Put premium dominance"
                )

        if (
            row[
                "trade_side_estimate"
            ]
            == "BUY EST."
        ):

            reasons.append(
                "Buy-side estimate"
            )

        if (
            row[
                "trade_side_estimate"
            ]
            == "SELL EST."
        ):

            reasons.append(
                "Sell-side estimate"
            )

        if row["DTE"] <= 45:

            reasons.append(
                "Near-term concentration"
            )

        if not reasons:

            reasons.append(
                "Flow score activity"
            )

        return " | ".join(
            reasons[:4]
        )

    df["flow_reason"] = (
        df.apply(
            reason,
            axis=1
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    output_rows = len(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_flow_scores = (
        df["flow_score"]
        .notna()
        .sum()
    )

    buy_est = (
        df[
            "trade_side_estimate"
        ]
        == "BUY EST."
    ).sum()

    sell_est = (
        df[
            "trade_side_estimate"
        ]
        == "SELL EST."
    ).sum()

    unknown_est = (
        df[
            "trade_side_estimate"
        ]
        == "UNKNOWN"
    ).sum()

    open_est = (
        df[
            "open_close_estimate"
        ]
        == "OPEN EST."
    ).sum()

    unknown_open = (
        df[
            "open_close_estimate"
        ]
        == "UNKNOWN"
    ).sum()

    print()
    print("=" * 72)
    print("🔎 STEP 5 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT ROWS        : {input_rows:,}"
    )

    print(
        f"OUTPUT ROWS       : {output_rows:,}"
    )

    print(
        f"TICKERS           : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"VALID FLOW SCORE  : "
        f"{valid_flow_scores:,}"
    )

    print()

    print(
        f"BUY EST.          : {buy_est:,}"
    )

    print(
        f"SELL EST.         : {sell_est:,}"
    )

    print(
        f"UNKNOWN           : {unknown_est:,}"
    )

    print()

    print(
        f"OPEN EST.         : {open_est:,}"
    )

    print(
        f"UNKNOWN OPEN/CLOSE: "
        f"{unknown_open:,}"
    )

    print()

    print(
        "PREMIUM SOURCE    : CALCULATED"
    )

    print(
        "TRADE SIDE SOURCE : "
        "ESTIMATED_BID_ASK_LAST"
    )

    print(
        "OPEN/CLOSE SOURCE : "
        "ESTIMATED_VOLUME_OI"
    )

    print()

    if input_rows == output_rows:

        print(
            "ROW COUNT CHECK   : OK"
        )

    else:

        print(
            "ROW COUNT CHECK   : ERROR"
        )

        raise RuntimeError(
            "Input/output row count mismatch."
        )

    print()

    print(
        "🔥 TOP 10 OPTION FLOW"
    )

    print("=" * 72)

    top_columns = [
        "flow_rank",
        "symbol",
        "option_type",
        "strike",
        "DTE",
        "volume",
        "openInterest",
        "volume_oi_ratio",
        "estimated_traded_premium",
        "trade_side_estimate",
        "flow_score",
        "flow_reason",
    ]

    print(
        df[
            top_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("=" * 72)

    print(
        f"OUTPUT FILE      : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 72)

    log(
        "STEP 5 UNUSUAL FLOW COMPLETE"
    )


if __name__ == "__main__":
    main()

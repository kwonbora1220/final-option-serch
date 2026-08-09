from __future__ import annotations

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
    "unusual_flow.csv",
)


# ============================================================
# FLOW SCORE WEIGHTS
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
# TRADE SIDE THRESHOLDS
#
# These are intentionally conservative.
#
# The goal is NOT to minimize UNKNOWN.
# The goal is to avoid false BUY / SELL classifications.
# ============================================================

BUY_RELATIVE_THRESHOLD = 0.85
SELL_RELATIVE_THRESHOLD = 0.15

CONFIDENCE_AT_QUOTE = 0.95
CONFIDENCE_NEAR_QUOTE = 0.85


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
# NUMERIC
# ============================================================

def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# OPTION TYPE
# ============================================================

def normalize_option_type(value):

    value = (
        str(value)
        .upper()
        .strip()
    )

    if value in {
        "CALL",
        "C",
        "CALLS",
    }:
        return "CALL"

    if value in {
        "PUT",
        "P",
        "PUTS",
    }:
        return "PUT"

    return "UNKNOWN"


# ============================================================
# PERCENTILE SCORE
# ============================================================

def percentile_score(series):

    series = numeric(series)

    if series.notna().sum() <= 1:

        return pd.Series(
            0.5,
            index=series.index,
        )

    rank = series.rank(
        pct=True,
        method="average",
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

        if (
            not np.isfinite(price)
            or
            not np.isfinite(strike)
            or
            price <= 0
            or
            strike <= 0
        ):
            return np.nan

        return (
            (strike - price)
            / price
        )

    except Exception:

        return np.nan


# ============================================================
# TRADE SIDE ESTIMATION
#
# IMPORTANT:
#
# This is NOT actual exchange trade direction.
#
# This is an EOD estimation based on the relationship between:
#
#   bid
#   ask
#   lastPrice
#
# We intentionally do NOT use:
#
#   last > midpoint -> BUY
#   last < midpoint -> SELL
#
# because that creates many false directional classifications.
#
# Conservative logic:
#
#   last > ask              -> BUY
#   last < bid              -> SELL
#
#   last very near ask      -> BUY
#   last very near bid      -> SELL
#
#   middle of spread        -> UNKNOWN
#   exact midpoint          -> UNKNOWN
#   zero spread             -> UNKNOWN
#
# Confidence is attached separately.
# ============================================================

def estimate_trade_side_details(row):

    try:

        bid = float(
            row["bid"]
        )

        ask = float(
            row["ask"]
        )

        last = float(
            row["lastPrice"]
        )

    except Exception:

        return (
            "UNKNOWN",
            0.0,
            "INSUFFICIENT_QUOTE_DATA",
        )

    values = [
        bid,
        ask,
        last,
    ]

    if not all(
        np.isfinite(v)
        for v in values
    ):

        return (
            "UNKNOWN",
            0.0,
            "NON_FINITE_QUOTE",
        )

    if (
        bid < 0
        or
        ask < 0
        or
        last <= 0
    ):

        return (
            "UNKNOWN",
            0.0,
            "INVALID_QUOTE",
        )

    if ask < bid:

        return (
            "UNKNOWN",
            0.0,
            "CROSSED_QUOTE",
        )

    spread = ask - bid

    # --------------------------------------------------------
    # ZERO SPREAD
    #
    # If bid == ask there is no information about whether
    # the trade was buyer-initiated or seller-initiated.
    # --------------------------------------------------------

    if spread <= 0:

        return (
            "UNKNOWN",
            0.0,
            "ZERO_SPREAD",
        )

    # --------------------------------------------------------
    # STRONG OUTSIDE-QUOTE SIGNAL
    #
    # Strictly above ask / below bid.
    # --------------------------------------------------------

    if last > ask:

        return (
            "BUY EST.",
            0.98,
            "LAST_ABOVE_ASK",
        )

    if last < bid:

        return (
            "SELL EST.",
            0.98,
            "LAST_BELOW_BID",
        )

    # --------------------------------------------------------
    # EXACT QUOTE TOUCH
    #
    # A trade exactly at ask/bid is useful information,
    # but slightly less certain than an outside-quote print.
    # --------------------------------------------------------

    if np.isclose(
        last,
        ask,
        rtol=0.0,
        atol=max(0.01, spread * 0.01),
    ):

        return (
            "BUY EST.",
            CONFIDENCE_AT_QUOTE,
            "LAST_AT_ASK",
        )

    if np.isclose(
        last,
        bid,
        rtol=0.0,
        atol=max(0.01, spread * 0.01),
    ):

        return (
            "SELL EST.",
            CONFIDENCE_AT_QUOTE,
            "LAST_AT_BID",
        )

    # --------------------------------------------------------
    # POSITION INSIDE SPREAD
    # --------------------------------------------------------

    relative_position = (
        last - bid
    ) / spread

    # --------------------------------------------------------
    # NEAR ASK
    # --------------------------------------------------------

    if (
        relative_position
        >=
        BUY_RELATIVE_THRESHOLD
    ):

        return (
            "BUY EST.",
            CONFIDENCE_NEAR_QUOTE,
            "LAST_NEAR_ASK",
        )

    # --------------------------------------------------------
    # NEAR BID
    # --------------------------------------------------------

    if (
        relative_position
        <=
        SELL_RELATIVE_THRESHOLD
    ):

        return (
            "SELL EST.",
            CONFIDENCE_NEAR_QUOTE,
            "LAST_NEAR_BID",
        )

    # --------------------------------------------------------
    # MID / AMBIGUOUS REGION
    #
    # DO NOT use midpoint direction.
    # DO NOT use daily change as a tiebreaker.
    #
    # This is intentionally UNKNOWN.
    # --------------------------------------------------------

    midpoint = (
        bid + ask
    ) / 2.0

    midpoint_distance = abs(
        last - midpoint
    ) / spread

    return (
        "UNKNOWN",
        0.0,
        "MID_SPREAD_AMBIGUOUS",
    )


# ============================================================
# NORMALIZED TRADE SIDE
# ============================================================

def normalize_trade_side(value):

    if pd.isna(value):

        return "UNKNOWN"

    text = (
        str(value)
        .upper()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if text in {
        "BUY",
        "BUY_EST",
        "BUY_EST.",
        "B",
        "BOT",
        "BTO",
        "BUY_TO_OPEN",
        "BUY_TO_CLOSE",
        "BTC",
    }:

        return "BUY"

    if text in {
        "SELL",
        "SELL_EST",
        "SELL_EST.",
        "S",
        "SOLD",
        "STO",
        "SELL_TO_OPEN",
        "SELL_TO_CLOSE",
        "STC",
    }:

        return "SELL"

    return "UNKNOWN"


# ============================================================
# OPEN / CLOSE ESTIMATION
#
# EOD option chain data cannot reliably identify
# open/close.
#
# We therefore use volume/OI only as a weak estimate.
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

    except Exception:

        return "UNKNOWN"

    if (
        not np.isfinite(volume)
        or
        not np.isfinite(oi)
        or
        volume <= 0
    ):

        return "UNKNOWN"

    if side not in {
        "BUY EST.",
        "SELL EST.",
    }:

        return "UNKNOWN"

    if oi <= 0:

        return "OPEN EST."

    ratio = (
        volume
        /
        oi
    )

    if ratio >= 1.0:

        return "OPEN EST."

    return "UNKNOWN"


# ============================================================
# FLOW DIRECTION
#
# CALL BUY = bullish
# PUT SELL = bullish
# CALL SELL = bearish
# PUT BUY = bearish
# ============================================================

def directional_flow(row):

    option_type = row[
        "option_type"
    ]

    side = row[
        "trade_side"
    ]

    if option_type == "CALL":

        if side == "BUY":
            return "BULLISH"

        if side == "SELL":
            return "BEARISH"

    elif option_type == "PUT":

        if side == "SELL":
            return "BULLISH"

        if side == "BUY":
            return "BEARISH"

    return "UNKNOWN"


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # ========================================================
    # INPUT
    # ========================================================

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    input_rows = len(df)

    log(
        f"INPUT ROWS : {input_rows:,}"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

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
        "change",

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

        raise RuntimeError(
            "Missing required columns: "
            +
            ", ".join(missing)
        )

    # ========================================================
    # NUMERIC
    # ========================================================

    numeric_columns = [

        "strike",
        "DTE",

        "volume",
        "openInterest",

        "bid",
        "ask",
        "lastPrice",
        "change",

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

    # ========================================================
    # OPTION TYPE
    # ========================================================

    df["option_type"] = (
        df["option_type"]
        .apply(
            normalize_option_type
        )
    )

    # ========================================================
    # MID PRICE
    # ========================================================

    df["mid_price"] = (
        df["bid"]
        +
        df["ask"]
    ) / 2.0

    df["mid_price"] = (
        df["mid_price"]
        .where(
            df["mid_price"].gt(0),
            df["lastPrice"],
        )
    )

    # ========================================================
    # PREMIUM
    # ========================================================

    df[
        "estimated_traded_premium"
    ] = (

        df["volume"].clip(
            lower=0
        )

        *

        df["mid_price"].clip(
            lower=0
        )

        *

        100.0
    )

    df["premium_source"] = (
        "CALCULATED_MID_OR_LAST"
    )

    # ========================================================
    # VOLUME / OI
    # ========================================================

    df["volume_oi_ratio"] = np.where(

        df["openInterest"] > 0,

        df["volume"]
        /
        df["openInterest"],

        np.nan,
    )

    # ========================================================
    # MONEYNESS
    # ========================================================

    df["moneyness"] = (
        df.apply(
            calculate_moneyness,
            axis=1,
        )
    )

    # ========================================================
    # TRADE SIDE
    # ========================================================

    log(
        "ESTIMATING BUY / SELL"
    )

    side_details = (
        df.apply(
            estimate_trade_side_details,
            axis=1,
        )
    )

    df[
        "trade_side_estimate"
    ] = side_details.apply(
        lambda x: x[0]
    )

    df[
        "trade_side_confidence"
    ] = side_details.apply(
        lambda x: x[1]
    )

    df[
        "trade_side_source"
    ] = side_details.apply(
        lambda x: x[2]
    )

    df["trade_side"] = (
        df["trade_side_estimate"]
        .apply(
            normalize_trade_side
        )
    )

    # ========================================================
    # OPEN / CLOSE
    # ========================================================

    log(
        "ESTIMATING OPEN / CLOSE"
    )

    df[
        "open_close_estimate"
    ] = (

        df.apply(
            estimate_open_close,
            axis=1,
        )
    )

    df[
        "open_close_source"
    ] = (
        "ESTIMATED_VOLUME_OI"
    )

    # ========================================================
    # DIRECTIONAL FLOW
    # ========================================================

    df[
        "directional_flow"
    ] = (

        df.apply(
            directional_flow,
            axis=1,
        )
    )

    # ========================================================
    # COMPONENT SCORES
    # ========================================================

    log(
        "CALCULATING FLOW COMPONENT SCORES"
    )

    volume_score = percentile_score(

        np.log1p(
            df["volume"]
            .clip(lower=0)
        )
    )

    volume_oi_score = percentile_score(

        np.log1p(

            df["volume_oi_ratio"]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .clip(lower=0)
        )
    )

    premium_score = percentile_score(

        np.log1p(

            df[
                "estimated_traded_premium"
            ]
            .clip(lower=0)
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

    dte_score = percentile_score(

        1.0
        /
        df["DTE"].clip(
            lower=1
        )
    )

    moneyness_score = percentile_score(

        -df["moneyness"].abs()
    )

    # ========================================================
    # FLOW SCORE
    # ========================================================

    df["flow_score"] = (

        volume_score
        *
        WEIGHT_VOLUME

        +

        volume_oi_score
        *
        WEIGHT_VOLUME_OI

        +

        premium_score
        *
        WEIGHT_PREMIUM

        +

        delta_score
        *
        WEIGHT_DELTA

        +

        gamma_score
        *
        WEIGHT_GAMMA

        +

        iv_score
        *
        WEIGHT_IV

        +

        dte_score
        *
        WEIGHT_DTE

        +

        moneyness_score
        *
        WEIGHT_MONEYNESS
    )

    # ========================================================
    # CALL / PUT FLOW
    # ========================================================

    df["call_put_flow"] = np.where(

        df["option_type"] == "CALL",

        df[
            "estimated_traded_premium"
        ],

        np.where(

            df["option_type"] == "PUT",

            -df[
                "estimated_traded_premium"
            ],

            0.0,
        ),
    )

    # ========================================================
    # DIRECTIONAL PREMIUM
    #
    # IMPORTANT:
    #
    # The estimated direction is weighted by confidence.
    #
    # Example:
    #
    # BUY confidence 0.95
    # premium $1M
    #
    # -> directional premium +$950K
    #
    # UNKNOWN confidence 0
    #
    # -> directional premium $0
    #
    # This prevents uncertain trades from having
    # the same influence as strong bid/ask signals.
    # ========================================================

    raw_directional_premium = np.where(

        df["trade_side"] == "BUY",

        np.where(

            df["option_type"] == "CALL",

            df[
                "estimated_traded_premium"
            ],

            -df[
                "estimated_traded_premium"
            ],
        ),

        np.where(

            df["trade_side"] == "SELL",

            np.where(

                df["option_type"] == "PUT",

                df[
                    "estimated_traded_premium"
                ],

                -df[
                    "estimated_traded_premium"
                ],
            ),

            0.0,
        ),
    )

    df[
        "estimated_directional_premium_raw"
    ] = raw_directional_premium

    df[
        "estimated_directional_premium"
    ] = (

        raw_directional_premium
        *
        df[
            "trade_side_confidence"
        ].fillna(0.0)
    )

    # ========================================================
    # FLOW RANK
    # ========================================================

    df = (
        df
        .sort_values(
            "flow_score",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    df["flow_rank"] = (
        np.arange(
            1,
            len(df) + 1,
        )
    )

    # ========================================================
    # SYMBOL SUMMARY
    # ========================================================

    log(
        "BUILDING SYMBOL FLOW SUMMARY"
    )

    symbol_summary = (

        df.groupby(
            "symbol",
            dropna=False,
        )

        .agg(

            option_count=(
                "symbol",
                "size",
            ),

            total_volume=(
                "volume",
                "sum",
            ),

            total_premium=(
                "estimated_traded_premium",
                "sum",
            ),

            call_premium=(
                "estimated_traded_premium",
                lambda x:
                x[
                    df.loc[
                        x.index,
                        "option_type",
                    ]
                    ==
                    "CALL"
                ].sum(),
            ),

            put_premium=(
                "estimated_traded_premium",
                lambda x:
                x[
                    df.loc[
                        x.index,
                        "option_type",
                    ]
                    ==
                    "PUT"
                ].sum(),
            ),

            max_flow_score=(
                "flow_score",
                "max",
            ),

            avg_flow_score=(
                "flow_score",
                "mean",
            ),

            bullish_premium=(
                "estimated_directional_premium",

                lambda x:
                x[
                    x > 0
                ].sum(),
            ),

            bearish_premium=(
                "estimated_directional_premium",

                lambda x:
                -x[
                    x < 0
                ].sum(),
            ),

            directional_premium_abs=(
                "estimated_directional_premium",

                lambda x:
                x.abs().sum(),
            ),

            average_trade_side_confidence=(
                "trade_side_confidence",
                "mean",
            ),

            confident_trade_count=(
                "trade_side_confidence",

                lambda x:
                (
                    x >= 0.85
                ).sum(),
            ),
        )

        .reset_index()
    )

    # ========================================================
    # CALL / PUT IMBALANCE
    # ========================================================

    total_cp = (

        symbol_summary[
            "call_premium"
        ]

        +

        symbol_summary[
            "put_premium"
        ]
    )

    symbol_summary[
        "call_put_premium_imbalance"
    ] = np.where(

        total_cp > 0,

        (

            symbol_summary[
                "call_premium"
            ]

            -

            symbol_summary[
                "put_premium"
            ]
        )

        /

        total_cp,

        0.0,
    )

    # ========================================================
    # SYMBOL DIRECTIONAL RATIO
    # ========================================================

    directional_total = (

        symbol_summary[
            "bullish_premium"
        ]

        +

        symbol_summary[
            "bearish_premium"
        ]
    )

    symbol_summary[
        "directional_ratio"
    ] = np.where(

        directional_total > 0,

        (

            symbol_summary[
                "bullish_premium"
            ]

            /

            directional_total
        )

        *

        100.0,

        50.0,
    )

    # ========================================================
    # SYMBOL FLOW SCORE
    # ========================================================

    symbol_summary[
        "symbol_flow_score"
    ] = (

        percentile_score(
            symbol_summary[
                "total_premium"
            ]
        )
        *
        40.0

        +

        percentile_score(
            symbol_summary[
                "total_volume"
            ]
        )
        *
        20.0

        +

        percentile_score(
            symbol_summary[
                "max_flow_score"
            ]
        )
        *
        25.0

        +

        (
            symbol_summary[
                "call_put_premium_imbalance"
            ].abs()
            *
            15.0
        )
    )

    symbol_summary = (

        symbol_summary
        .sort_values(
            "symbol_flow_score",
            ascending=False,
        )
        .reset_index(
            drop=True,
        )
    )

    symbol_summary[
        "symbol_flow_rank"
    ] = np.arange(
        1,
        len(symbol_summary) + 1,
    )

    # ========================================================
    # SYMBOL FLOW DIRECTION
    # ========================================================

    def symbol_direction(row):

        ratio = row[
            "directional_ratio"
        ]

        if ratio >= 60:
            return "BULLISH"

        if ratio <= 40:
            return "BEARISH"

        return "MIXED"

    symbol_summary[
        "estimated_direction"
    ] = symbol_summary.apply(
        symbol_direction,
        axis=1,
    )

    # ========================================================
    # CALL / PUT DIRECTION
    # ========================================================

    def cp_direction(row):

        value = row[
            "call_put_premium_imbalance"
        ]

        if value >= 0.15:
            return "CALL DOMINANT"

        if value <= -0.15:
            return "PUT DOMINANT"

        return "BALANCED"

    symbol_summary[
        "flow_direction"
    ] = symbol_summary.apply(
        cp_direction,
        axis=1,
    )

    # ========================================================
    # MERGE SYMBOL DATA
    # ========================================================

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

                "directional_ratio",

                "estimated_direction",

                "flow_direction",

                "average_trade_side_confidence",

                "confident_trade_count",
            ]
        ],

        on="symbol",

        how="left",
    )

    # ========================================================
    # FLOW REASON
    # ========================================================

    def reason(row):

        reasons = []

        ratio = row[
            "volume_oi_ratio"
        ]

        if (
            np.isfinite(ratio)
            and
            ratio >= 1
        ):

            reasons.append(
                "Volume/OI surge"
            )

        if (
            row[
                "estimated_traded_premium"
            ]
            > 0
        ):

            reasons.append(
                "Premium activity"
            )

        imbalance = row[
            "call_put_premium_imbalance"
        ]

        if (
            np.isfinite(imbalance)
            and
            abs(imbalance) >= 0.25
        ):

            if imbalance > 0:

                reasons.append(
                    "Call premium dominance"
                )

            else:

                reasons.append(
                    "Put premium dominance"
                )

        side = row[
            "trade_side"
        ]

        if side == "BUY":

            reasons.append(
                "Buy-side estimate"
            )

        elif side == "SELL":

            reasons.append(
                "Sell-side estimate"
            )

        elif side == "UNKNOWN":

            reasons.append(
                "Side ambiguous"
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
            axis=1,
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    output_rows = len(df)

    buy_est = (
        df[
            "trade_side_estimate"
        ]
        ==
        "BUY EST."
    ).sum()

    sell_est = (
        df[
            "trade_side_estimate"
        ]
        ==
        "SELL EST."
    ).sum()

    unknown_est = (
        df[
            "trade_side_estimate"
        ]
        ==
        "UNKNOWN"
    ).sum()

    buy_normalized = (
        df[
            "trade_side"
        ]
        ==
        "BUY"
    ).sum()

    sell_normalized = (
        df[
            "trade_side"
        ]
        ==
        "SELL"
    ).sum()

    unknown_normalized = (
        df[
            "trade_side"
        ]
        ==
        "UNKNOWN"
    ).sum()

    open_est = (
        df[
            "open_close_estimate"
        ]
        ==
        "OPEN EST."
    ).sum()

    unknown_open = (
        df[
            "open_close_estimate"
        ]
        ==
        "UNKNOWN"
    ).sum()

    avg_confidence = (
        df[
            "trade_side_confidence"
        ]
        .mean()
    )

    high_confidence_count = (
        df[
            "trade_side_confidence"
        ]
        >=
        0.85
    ).sum()

    high_confidence_ratio = (

        high_confidence_count
        /
        len(df)
        *
        100.0
    )

    print()

    print(
        "=" * 72
    )

    print(
        "🔎 STEP 5 VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"INPUT ROWS        : "
        f"{input_rows:,}"
    )

    print(
        f"OUTPUT ROWS       : "
        f"{output_rows:,}"
    )

    print(
        f"TICKERS           : "
        f"{df['symbol'].nunique():,}"
    )

    print()

    print(
        f"BUY EST.          : "
        f"{buy_est:,}"
    )

    print(
        f"SELL EST.         : "
        f"{sell_est:,}"
    )

    print(
        f"UNKNOWN           : "
        f"{unknown_est:,}"
    )

    print()

    print(
        f"BUY NORMALIZED    : "
        f"{buy_normalized:,}"
    )

    print(
        f"SELL NORMALIZED   : "
        f"{sell_normalized:,}"
    )

    print(
        f"UNKNOWN NORMALIZED: "
        f"{unknown_normalized:,}"
    )

    print()

    print(
        f"AVG SIDE CONF.    : "
        f"{avg_confidence:.4f}"
    )

    print(
        f"HIGH CONF. SIDE   : "
        f"{high_confidence_count:,}"
    )

    print(
        f"HIGH CONF. RATIO  : "
        f"{high_confidence_ratio:.2f}%"
    )

    print()

    print(
        f"OPEN EST.         : "
        f"{open_est:,}"
    )

    print(
        f"UNKNOWN OPEN/CLOSE: "
        f"{unknown_open:,}"
    )

    print()

    print(
        "PREMIUM SOURCE    : "
        "CALCULATED_MID_OR_LAST"
    )

    print(
        "TRADE SIDE SOURCE : "
        "ESTIMATED_BID_ASK_CONSERVATIVE"
    )

    print(
        "OPEN/CLOSE SOURCE : "
        "ESTIMATED_VOLUME_OI"
    )

    print()

    if input_rows != output_rows:

        raise RuntimeError(
            "Input/output row count mismatch."
        )

    print(
        "ROW COUNT CHECK   : OK"
    )

    print()

    print(
        "🔥 TOP 20 OPTION FLOW"
    )

    print(
        "=" * 72
    )

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
        "trade_side",
        "trade_side_confidence",
        "trade_side_source",
        "directional_flow",
        "flow_score",
        "flow_reason",
    ]

    print(

        df[
            top_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print(
        "=" * 72
    )

    print(
        f"OUTPUT FILE      : "
        f"{OUTPUT_FILE}"
    )

    log(
        "STEP 5 UNUSUAL FLOW COMPLETE"
    )


if __name__ == "__main__":
    main()

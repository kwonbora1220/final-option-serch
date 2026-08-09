from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TOP20_FILE = "data/analysis/top20.csv"
GREEKS_FILE = "data/analysis/options_greeks.csv"

OUTPUT_DIR = "data/analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "option_search.csv",
)

TOP_CALLS = 5
TOP_PUTS = 5

BUY_THRESHOLD = 0.80
SELL_THRESHOLD = 0.20


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
        f"[07 SEARCH] {now} | {message}"
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
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0,
):

    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):

            return value

    except Exception:

        pass

    return default


# ============================================================
# NORMALIZE SIDE
# ============================================================

def normalize_side(value):

    if pd.isna(value):

        return "UNKNOWN"

    text = (
        str(value)
        .upper()
        .strip()
    )

    if (
        "BUY"
        in text
    ):

        return "BUY"

    if (
        "SELL"
        in text
    ):

        return "SELL"

    return "UNKNOWN"


# ============================================================
# TRADE SIDE ESTIMATION
# ============================================================

def estimate_trade_side(row):

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
            "INVALID_QUOTE",
        )

    if not all(
        np.isfinite(v)
        for v in [
            bid,
            ask,
            last,
        ]
    ):

        return (
            "UNKNOWN",
            0.0,
            "INVALID_QUOTE",
        )

    if (
        bid < 0
        or
        ask < 0
        or
        last <= 0
        or
        ask < bid
    ):

        return (
            "UNKNOWN",
            0.0,
            "INVALID_QUOTE",
        )

    spread = (
        ask
        -
        bid
    )

    if spread <= 0:

        return (
            "UNKNOWN",
            0.0,
            "ZERO_SPREAD",
        )

    # --------------------------------------------------------
    # AT ASK
    # --------------------------------------------------------

    if last >= ask:

        return (
            "BUY EST.",
            1.0,
            "AT_OR_ABOVE_ASK",
        )

    # --------------------------------------------------------
    # AT BID
    # --------------------------------------------------------

    if last <= bid:

        return (
            "SELL EST.",
            1.0,
            "AT_OR_BELOW_BID",
        )

    # --------------------------------------------------------
    # POSITION INSIDE SPREAD
    # --------------------------------------------------------

    position = (
        last
        -
        bid
    ) / spread

    # --------------------------------------------------------
    # NEAR ASK
    # --------------------------------------------------------

    if position >= BUY_THRESHOLD:

        confidence = (
            0.80
            +
            (
                position
                -
                BUY_THRESHOLD
            )
            /
            (
                1.0
                -
                BUY_THRESHOLD
            )
            *
            0.20
        )

        confidence = min(
            1.0,
            max(
                0.80,
                confidence,
            ),
        )

        return (
            "BUY EST.",
            confidence,
            "NEAR_ASK",
        )

    # --------------------------------------------------------
    # NEAR BID
    # --------------------------------------------------------

    if position <= SELL_THRESHOLD:

        confidence = (
            0.80
            +
            (
                SELL_THRESHOLD
                -
                position
            )
            /
            SELL_THRESHOLD
            *
            0.20
        )

        confidence = min(
            1.0,
            max(
                0.80,
                confidence,
            ),
        )

        return (
            "SELL EST.",
            confidence,
            "NEAR_BID",
        )

    # --------------------------------------------------------
    # MID / AMBIGUOUS
    # --------------------------------------------------------

    return (
        "UNKNOWN",
        0.0,
        "MID_AMBIGUOUS",
    )


# ============================================================
# PERCENTILE SCORE
# ============================================================

def percentile_score(series):

    series = numeric(
        series
    )

    if (
        series.notna().sum()
        <= 1
    ):

        return pd.Series(
            0.5,
            index=series.index,
        )

    return (
        series
        .rank(
            pct=True,
            method="average",
        )
        .fillna(0)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # ========================================================
    # FILE CHECK
    # ========================================================

    for file_path in [
        TOP20_FILE,
        GREEKS_FILE,
    ]:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Required file not found: "
                f"{file_path}"
            )

    # ========================================================
    # LOAD
    # ========================================================

    top20 = pd.read_csv(
        TOP20_FILE
    )

    options = pd.read_csv(
        GREEKS_FILE
    )

    log(
        f"TOP20 ROWS : "
        f"{len(top20):,}"
    )

    log(
        f"OPTION ROWS : "
        f"{len(options):,}"
    )

    # ========================================================
    # INPUT CHECK
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "🔎 STEP 7 INPUT COLUMN CHECK"
    )

    print(
        "=" * 72
    )

    print(
        "GREEKS COLUMNS:"
    )

    print(
        ", ".join(
            options.columns.tolist()
        )
    )

    print(
        "=" * 72
    )

    # ========================================================
    # REQUIRED TOP20
    # ========================================================

    required_top20 = [
        "symbol"
    ]

    missing = [

        column

        for column
        in required_top20

        if column
        not in top20.columns
    ]

    if missing:

        raise ValueError(
            "Missing TOP20 columns: "
            +
            ", ".join(
                missing
            )
        )

    # ========================================================
    # REQUIRED GREEKS
    # ========================================================

    required_options = [

        "symbol",
        "option_type",
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

    missing = [

        column

        for column
        in required_options

        if column
        not in options.columns
    ]

    if missing:

        raise ValueError(
            "Missing Greeks columns: "
            +
            ", ".join(
                missing
            )
        )

    # ========================================================
    # TOP20 SYMBOLS
    # ========================================================

    top_symbols = (

        top20[
            "symbol"
        ]

        .dropna()

        .astype(str)

        .str.upper()

        .unique()

        .tolist()
    )

    print()

    print(
        "TOP20 SYMBOLS:"
    )

    print(
        ", ".join(
            top_symbols
        )
    )

    # ========================================================
    # NORMALIZE SYMBOL
    # ========================================================

    options[
        "symbol"
    ] = (
        options[
            "symbol"
        ]
        .astype(str)
        .str.upper()
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

        "impliedVolatility",
        "delta",
        "gamma",
        "vega",

        "underlying_price",
    ]

    for column in numeric_columns:

        options[
            column
        ] = numeric(
            options[
                column
            ]
        )

    # ========================================================
    # FILTER TOP20
    # ========================================================

    search = options[
        options[
            "symbol"
        ].isin(
            top_symbols
        )
    ].copy()

    # ========================================================
    # DTE
    # ========================================================

    search = search[
        (
            search[
                "DTE"
            ] >= 0
        )
        &
        (
            search[
                "DTE"
            ] <= 180
        )
    ].copy()

    # ========================================================
    # OPTION TYPE
    # ========================================================

    search[
        "option_type"
    ] = (
        search[
            "option_type"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ========================================================
    # MID PRICE
    # ========================================================

    search[
        "mid_price"
    ] = (

        search[
            "bid"
        ]
        +
        search[
            "ask"
        ]

    ) / 2.0

    search[
        "mid_price"
    ] = (
        search[
            "mid_price"
        ]
        .where(
            search[
                "mid_price"
            ].gt(0),

            search[
                "lastPrice"
            ],
        )
    )

    # ========================================================
    # VOLUME / OI
    # ========================================================

    search[
        "volume_oi_ratio"
    ] = np.where(

        search[
            "openInterest"
        ] > 0,

        search[
            "volume"
        ]
        /
        search[
            "openInterest"
        ],

        np.nan,
    )

    # ========================================================
    # PREMIUM
    # ========================================================

    if (
        "estimated_traded_premium"
        in search.columns
    ):

        search[
            "estimated_traded_premium"
        ] = numeric(
            search[
                "estimated_traded_premium"
            ]
        )

    else:

        search[
            "estimated_traded_premium"
        ] = (

            search[
                "volume"
            ].clip(
                lower=0
            )

            *

            search[
                "mid_price"
            ].clip(
                lower=0
            )

            *

            100.0
        )

    # ========================================================
    # TRADE SIDE
    # ========================================================

    log(
        "ESTIMATING OPTION TRADE SIDE"
    )

    side_result = search.apply(
        estimate_trade_side,
        axis=1,
        result_type="expand",
    )

    side_result.columns = [

        "trade_side_estimate",
        "trade_side_confidence",
        "trade_side_method",
    ]

    search = pd.concat(
        [
            search,
            side_result,
        ],
        axis=1,
    )

    search[
        "trade_side"
    ] = (
        search[
            "trade_side_estimate"
        ]
        .apply(
            normalize_side
        )
    )

    search[
        "trade_side_source"
    ] = (
        "STEP7_BID_ASK_ESTIMATE"
    )

    # ========================================================
    # OPTION SCORES
    # ========================================================

    search[
        "premium_score"
    ] = percentile_score(

        np.log1p(
            search[
                "estimated_traded_premium"
            ].clip(
                lower=0
            )
        )
    )

    search[
        "volume_oi_score"
    ] = percentile_score(

        np.log1p(

            search[
                "volume_oi_ratio"
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .clip(
                lower=0
            )
        )
    )

    search[
        "gamma_score"
    ] = percentile_score(
        search[
            "gamma"
        ].abs()
    )

    search[
        "delta_score"
    ] = percentile_score(
        search[
            "delta"
        ].abs()
    )

    # ========================================================
    # MONEYNESS
    # ========================================================

    search[
        "moneyness_distance"
    ] = np.where(

        search[
            "underlying_price"
        ] > 0,

        (
            search[
                "strike"
            ]
            -
            search[
                "underlying_price"
            ]
        ).abs()
        /
        search[
            "underlying_price"
        ],

        np.nan,
    )

    search[
        "moneyness_score"
    ] = percentile_score(
        -search[
            "moneyness_distance"
        ]
    )

    # ========================================================
    # OPTION IMPORTANCE
    # ========================================================

    search[
        "option_importance_score"
    ] = (

        search[
            "premium_score"
        ]
        *
        30

        +

        search[
            "volume_oi_score"
        ]
        *
        20

        +

        search[
            "gamma_score"
        ]
        *
        20

        +

        search[
            "delta_score"
        ]
        *
        10

        +

        search[
            "moneyness_score"
        ]
        *
        10

        +

        search[
            "trade_side_confidence"
        ]
        *
        10
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # PER SYMBOL
    # ========================================================

    for symbol in top_symbols:

        group = search[
            search[
                "symbol"
            ]
            ==
            symbol
        ].copy()

        if group.empty:

            continue

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        current_price = safe_float(
            group[
                "underlying_price"
            ].median()
        )

        # ----------------------------------------------------
        # CALL
        # ----------------------------------------------------

        calls = group[
            group[
                "option_type"
            ]
            ==
            "CALL"
        ].copy()

        # ----------------------------------------------------
        # PUT
        # ----------------------------------------------------

        puts = group[
            group[
                "option_type"
            ]
            ==
            "PUT"
        ].copy()

        # ----------------------------------------------------
        # TOP CALLS
        # ----------------------------------------------------

        top_calls = (
            calls
            .sort_values(
                "option_importance_score",
                ascending=False,
            )
            .head(
                TOP_CALLS
            )
        )

        # ----------------------------------------------------
        # TOP PUTS
        # ----------------------------------------------------

        top_puts = (
            puts
            .sort_values(
                "option_importance_score",
                ascending=False,
            )
            .head(
                TOP_PUTS
            )
        )

        # ----------------------------------------------------
        # PREMIUM
        # ----------------------------------------------------

        call_premium = safe_float(
            calls[
                "estimated_traded_premium"
            ].sum()
        )

        put_premium = safe_float(
            puts[
                "estimated_traded_premium"
            ].sum()
        )

        total_premium = (
            call_premium
            +
            put_premium
        )

        if total_premium > 0:

            call_put_imbalance = (

                call_premium
                -
                put_premium

            ) / total_premium

        else:

            call_put_imbalance = 0.0

        # ----------------------------------------------------
        # STRUCTURE BIAS
        # ----------------------------------------------------

        if (
            call_put_imbalance
            >=
            0.25
        ):

            structure_bias = (
                "CALL DOMINANT"
            )

        elif (
            call_put_imbalance
            <=
            -0.25
        ):

            structure_bias = (
                "PUT DOMINANT"
            )

        else:

            structure_bias = (
                "BALANCED"
            )

        # ----------------------------------------------------
        # WALL SCORE
        # ----------------------------------------------------

        if not calls.empty:

            calls[
                "wall_score"
            ] = (

                percentile_score(
                    calls[
                        "openInterest"
                    ]
                )
                *
                35

                +

                percentile_score(
                    calls[
                        "volume"
                    ]
                )
                *
                20

                +

                percentile_score(
                    calls[
                        "estimated_traded_premium"
                    ]
                )
                *
                20

                +

                percentile_score(
                    calls[
                        "gamma"
                    ].abs()
                )
                *
                15

                +

                percentile_score(
                    -calls[
                        "moneyness_distance"
                    ]
                )
                *
                10
            )

            call_wall = (
                calls
                .sort_values(
                    "wall_score",
                    ascending=False,
                )
                .iloc[0]
            )

            call_wall_strike = safe_float(
                call_wall[
                    "strike"
                ]
            )

        else:

            call_wall_strike = np.nan

        # ----------------------------------------------------
        # PUT WALL
        # ----------------------------------------------------

        if not puts.empty:

            puts[
                "wall_score"
            ] = (

                percentile_score(
                    puts[
                        "openInterest"
                    ]
                )
                *
                35

                +

                percentile_score(
                    puts[
                        "volume"
                    ]
                )
                *
                20

                +

                percentile_score(
                    puts[
                        "estimated_traded_premium"
                    ]
                )
                *
                20

                +

                percentile_score(
                    puts[
                        "gamma"
                    ].abs()
                )
                *
                15

                +

                percentile_score(
                    -puts[
                        "moneyness_distance"
                    ]
                )
                *
                10
            )

            put_wall = (
                puts
                .sort_values(
                    "wall_score",
                    ascending=False,
                )
                .iloc[0]
            )

            put_wall_strike = safe_float(
                put_wall[
                    "strike"
                ]
            )

        else:

            put_wall_strike = np.nan

        # ----------------------------------------------------
        # GEX
        # ----------------------------------------------------

        if "gex" in group.columns:

            total_gex = safe_float(
                group[
                    "gex"
                ].sum()
            )

            call_gex = safe_float(
                calls[
                    "gex"
                ].sum()
            )

            put_gex = safe_float(
                puts[
                    "gex"
                ].sum()
            )

        else:

            total_gex = np.nan
            call_gex = np.nan
            put_gex = np.nan

        # ----------------------------------------------------
        # FORMAT OPTION
        # ----------------------------------------------------

        def format_option(row):

            iv = safe_float(
                row[
                    "impliedVolatility"
                ]
            )

            if iv < 2:

                iv_display = (
                    iv * 100
                )

            else:

                iv_display = iv

            return (

                f"${safe_float(row['strike']):.2f}"
                f" {row['option_type']}"

                f" | DTE "
                f"{int(safe_float(row['DTE']))}"

                f" | Vol "
                f"{int(safe_float(row['volume']))}"

                f" | OI "
                f"{int(safe_float(row['openInterest']))}"

                f" | IV "
                f"{iv_display:.1f}%"

                f" | Delta "
                f"{safe_float(row['delta']):+.2f}"

                f" | Gamma "
                f"{safe_float(row['gamma']):.4f}"

                f" | Premium "
                f"${safe_float(row['estimated_traded_premium']):,.0f}"

                f" | "
                f"{row['trade_side_estimate']}"

                f" | Conf "
                f"{safe_float(row['trade_side_confidence']):.2f}"
            )

        top_call_text = " || ".join(

            format_option(row)

            for _, row
            in top_calls.iterrows()
        )

        top_put_text = " || ".join(

            format_option(row)

            for _, row
            in top_puts.iterrows()
        )

        # ----------------------------------------------------
        # RISK REVERSAL
        # ----------------------------------------------------
        #
        # Only reliable estimates are allowed.
        #
        # CALL:
        #   BUY + confidence >= 0.80
        #
        # PUT:
        #   SELL + confidence >= 0.80
        #
        # DTE difference <= 14
        #
        # CALL strike > PUT strike
        # ----------------------------------------------------

        best_rr = None

        if (
            not calls.empty
            and
            not puts.empty
        ):

            reliable_calls = calls[
                (
                    calls[
                        "trade_side"
                    ]
                    ==
                    "BUY"
                )
                &
                (
                    calls[
                        "trade_side_confidence"
                    ]
                    >=
                    0.80
                )
            ]

            reliable_puts = puts[
                (
                    puts[
                        "trade_side"
                    ]
                    ==
                    "SELL"
                )
                &
                (
                    puts[
                        "trade_side_confidence"
                    ]
                    >=
                    0.80
                )
            ]

            for _, call in (
                reliable_calls.iterrows()
            ):

                for _, put in (
                    reliable_puts.iterrows()
                ):

                    call_dte = safe_float(
                        call[
                            "DTE"
                        ]
                    )

                    put_dte = safe_float(
                        put[
                            "DTE"
                        ]
                    )

                    if (
                        abs(
                            call_dte
                            -
                            put_dte
                        )
                        >
                        14
                    ):

                        continue

                    call_strike = safe_float(
                        call[
                            "strike"
                        ]
                    )

                    put_strike = safe_float(
                        put[
                            "strike"
                        ]
                    )

                    if (
                        call_strike
                        <=
                        put_strike
                    ):

                        continue

                    rr_score = (

                        safe_float(
                            call[
                                "option_importance_score"
                            ]
                        )

                        +

                        safe_float(
                            put[
                                "option_importance_score"
                            ]
                        )

                        +

                        safe_float(
                            call[
                                "trade_side_confidence"
                            ]
                        )
                        *
                        10

                        +

                        safe_float(
                            put[
                                "trade_side_confidence"
                            ]
                        )
                        *
                        10
                    )

                    candidate = (
                        rr_score,
                        call,
                        put,
                    )

                    if (
                        best_rr is None
                        or
                        rr_score
                        >
                        best_rr[0]
                    ):

                        best_rr = candidate

        # ----------------------------------------------------
        # RISK REVERSAL OUTPUT
        # ----------------------------------------------------

        if best_rr is not None:

            (
                rr_score,
                rr_call,
                rr_put,
            ) = best_rr

            rr_status = (
                "BULLISH "
                "RISK-REVERSAL EST."
            )

            rr_call_strike = safe_float(
                rr_call[
                    "strike"
                ]
            )

            rr_call_dte = safe_float(
                rr_call[
                    "DTE"
                ]
            )

            rr_call_premium = safe_float(
                rr_call[
                    "estimated_traded_premium"
                ]
            )

            rr_put_strike = safe_float(
                rr_put[
                    "strike"
                ]
            )

            rr_put_dte = safe_float(
                rr_put[
                    "DTE"
                ]
            )

            rr_put_premium = safe_float(
                rr_put[
                    "estimated_traded_premium"
                ]
            )

        else:

            rr_status = (
                "NONE DETECTED"
            )

            rr_score = np.nan

            rr_call_strike = np.nan
            rr_call_dte = np.nan
            rr_call_premium = np.nan

            rr_put_strike = np.nan
            rr_put_dte = np.nan
            rr_put_premium = np.nan

        # ----------------------------------------------------
        # RESULT ROW
        # ----------------------------------------------------

        results.append(

            {
                "symbol":
                    symbol,

                "current_price":
                    current_price,

                "option_rows_analyzed":
                    len(group),

                "expiration_count":
                    group[
                        "DTE"
                    ].nunique(),

                "dte_min":
                    group[
                        "DTE"
                    ].min(),

                "dte_max":
                    group[
                        "DTE"
                    ].max(),

                "call_count":
                    len(calls),

                "put_count":
                    len(puts),

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "call_put_imbalance":
                    call_put_imbalance,

                "structure_bias":
                    structure_bias,

                "call_wall":
                    call_wall_strike,

                "put_wall":
                    put_wall_strike,

                "total_gex":
                    total_gex,

                "call_gex":
                    call_gex,

                "put_gex":
                    put_gex,

                "top_calls":
                    top_call_text,

                "top_puts":
                    top_put_text,

                "risk_reversal":
                    rr_status,

                "rr_score":
                    rr_score,

                "rr_call_strike":
                    rr_call_strike,

                "rr_call_dte":
                    rr_call_dte,

                "rr_call_premium":
                    rr_call_premium,

                "rr_put_strike":
                    rr_put_strike,

                "rr_put_dte":
                    rr_put_dte,

                "rr_put_premium":
                    rr_put_premium,
            }
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    result = pd.DataFrame(
        results
    )

    if result.empty:

        raise RuntimeError(
            "STEP 7 generated no results."
        )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    side_buy = (
        search[
            "trade_side"
        ]
        ==
        "BUY"
    ).sum()

    side_sell = (
        search[
            "trade_side"
        ]
        ==
        "SELL"
    ).sum()

    side_unknown = (
        search[
            "trade_side"
        ]
        ==
        "UNKNOWN"
    ).sum()

    rr_count = (
        result[
            "risk_reversal"
        ]
        !=
        "NONE DETECTED"
    ).sum()

    print()

    print(
        "=" * 72
    )

    print(
        "🔎 STEP 7 VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"TOP20 INPUT        : "
        f"{len(top_symbols):,}"
    )

    print(
        f"SYMBOLS ANALYZED   : "
        f"{result['symbol'].nunique():,}"
    )

    print(
        f"OPTION ROWS INPUT  : "
        f"{len(options):,}"
    )

    print(
        f"OPTION ROWS SEARCH : "
        f"{len(search):,}"
    )

    print(
        f"DTE MIN            : "
        f"{search['DTE'].min()}"
    )

    print(
        f"DTE MAX            : "
        f"{search['DTE'].max()}"
    )

    print()

    print(
        "CALL / PUT SEARCH  : OK"
    )

    print(
        "0-180 DTE SEARCH   : OK"
    )

    print(
        "FLOW DATA LINK     : DIRECT FROM GREEKS"
    )

    print()

    print(
        f"BUY EST.           : "
        f"{side_buy:,}"
    )

    print(
        f"SELL EST.          : "
        f"{side_sell:,}"
    )

    print(
        f"UNKNOWN            : "
        f"{side_unknown:,}"
    )

    print()

    print(
        f"RISK REVERSAL      : "
        f"{rr_count:,}"
    )

    print()

    print(
        "UNUSUAL FLOW CSV   : NOT USED"
    )

    print()

    print(
        "=" * 72
    )

    print(
        "🔥 OPTION SEARCH SUMMARY"
    )

    print(
        "=" * 72
    )

    print(

        result[
            [
                "symbol",
                "current_price",
                "option_rows_analyzed",
                "dte_min",
                "dte_max",
                "call_count",
                "put_count",
                "structure_bias",
                "call_wall",
                "put_wall",
                "risk_reversal",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 72
    )

    print(
        "🔥 RISK REVERSAL SUMMARY"
    )

    print(
        "=" * 72
    )

    for _, row in (
        result.iterrows()
    ):

        print(
            f"{row['symbol']} → "
            f"{row['risk_reversal']}"
        )

        if (
            row[
                "risk_reversal"
            ]
            !=
            "NONE DETECTED"
        ):

            print(
                f"   CALL "
                f"${row['rr_call_strike']:.2f}"
                f" / DTE "
                f"{int(row['rr_call_dte'])}"
                f" / Premium "
                f"${row['rr_call_premium']:,.0f}"
            )

            print(
                f"   PUT "
                f"${row['rr_put_strike']:.2f}"
                f" / DTE "
                f"{int(row['rr_put_dte'])}"
                f" / Premium "
                f"${row['rr_put_premium']:,.0f}"
            )

    print()

    print(
        f"OUTPUT FILE       : "
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 72
    )

    log(
        "STEP 7 OPTION SEARCH COMPLETE"
    )


if __name__ == "__main__":

    main()

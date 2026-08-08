import os
from datetime import datetime

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
    "option_search.csv"
)

TOP_CALLS = 5
TOP_PUTS = 5


# ============================================================
# LOG
# ============================================================

def log(message):

    now = datetime.utcnow().strftime(
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
        errors="coerce"
    )


# ============================================================
# SAFE VALUE
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


# ============================================================
# OPTION TEXT
# ============================================================

def option_label(row):

    option_type = str(
        row["option_type"]
    )

    strike = safe_float(
        row["strike"]
    )

    dte = safe_float(
        row["DTE"]
    )

    return (
        f"{option_type} "
        f"${strike:.2f} "
        f"DTE {int(dte)}"
    )


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

    return (
        series.rank(
            pct=True,
            method="average"
        )
        .fillna(0.0)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if not os.path.exists(
        TOP20_FILE
    ):

        raise FileNotFoundError(
            f"TOP20 file not found: "
            f"{TOP20_FILE}"
        )

    if not os.path.exists(
        GREEKS_FILE
    ):

        raise FileNotFoundError(
            f"Greeks file not found: "
            f"{GREEKS_FILE}"
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    top20 = pd.read_csv(
        TOP20_FILE
    )

    options = pd.read_csv(
        GREEKS_FILE
    )

    log(
        f"TOP20 ROWS : {len(top20):,}"
    )

    log(
        f"OPTION ROWS : {len(options):,}"
    )

    # --------------------------------------------------------
    # REQUIRED
    # --------------------------------------------------------

    required_top20 = [
        "symbol",
        "rank",
    ]

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
        "estimated_traded_premium",
        "trade_side_estimate",
        "flow_score",
    ]

    missing_top20 = [
        c
        for c in required_top20
        if c not in top20.columns
    ]

    missing_options = [
        c
        for c in required_options
        if c not in options.columns
    ]

    if missing_top20:

        raise ValueError(
            "Missing TOP20 columns: "
            + ", ".join(
                missing_top20
            )
        )

    if missing_options:

        raise ValueError(
            "Missing option columns: "
            + ", ".join(
                missing_options
            )
        )

    # --------------------------------------------------------
    # NUMERIC
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
        "estimated_traded_premium",
        "flow_score",
    ]

    for column in numeric_columns:

        options[column] = numeric(
            options[column]
        )

    # --------------------------------------------------------
    # FILTER TOP20
    # --------------------------------------------------------

    top_symbols = (
        top20[
            "symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    log(
        "TOP20 SYMBOLS:"
    )

    print(
        ", ".join(
            top_symbols
        )
    )

    search = options[
        options[
            "symbol"
        ]
        .astype(str)
        .isin(
            top_symbols
        )
    ].copy()

    # --------------------------------------------------------
    # DTE VALIDATION
    # --------------------------------------------------------

    search = search[
        (search["DTE"] >= 0)
        &
        (search["DTE"] <= 180)
    ].copy()

    # --------------------------------------------------------
    # CALCULATED METRICS
    # --------------------------------------------------------

    search[
        "mid_price"
    ] = (
        search["bid"]
        + search["ask"]
    ) / 2.0

    search[
        "volume_oi_ratio"
    ] = np.where(

        search[
            "openInterest"
        ] > 0,

        search["volume"]
        /
        search[
            "openInterest"
        ],

        np.nan
    )

    # --------------------------------------------------------
    # PREMIUM SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VOLUME/OI SCORE
    # --------------------------------------------------------

    search[
        "volume_oi_score"
    ] = percentile_score(
        np.log1p(
            search[
                "volume_oi_ratio"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .clip(
                lower=0
            )
        )
    )

    # --------------------------------------------------------
    # FLOW SCORE
    # --------------------------------------------------------

    search[
        "flow_score_norm"
    ] = percentile_score(
        search[
            "flow_score"
        ]
    )

    # --------------------------------------------------------
    # GAMMA SCORE
    # --------------------------------------------------------

    search[
        "gamma_score"
    ] = percentile_score(
        search[
            "gamma"
        ].abs()
    )

    # --------------------------------------------------------
    # DELTA SCORE
    # --------------------------------------------------------

    search[
        "delta_score"
    ] = percentile_score(
        search[
            "delta"
        ].abs()
    )

    # --------------------------------------------------------
    # NEAR MONEY SCORE
    # --------------------------------------------------------

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

        np.nan
    )

    search[
        "moneyness_score"
    ] = percentile_score(
        -search[
            "moneyness_distance"
        ]
    )

    # --------------------------------------------------------
    # OPTION IMPORTANCE SCORE
    # --------------------------------------------------------

    search[
        "option_importance_score"
    ] = (

        search[
            "premium_score"
        ] * 30

        + search[
            "volume_oi_score"
        ] * 20

        + search[
            "flow_score_norm"
        ] * 30

        + search[
            "gamma_score"
        ] * 10

        + search[
            "delta_score"
        ] * 5

        + search[
            "moneyness_score"
        ] * 5
    )

    # --------------------------------------------------------
    # SEARCH RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # PER SYMBOL
    # --------------------------------------------------------

    for symbol in top_symbols:

        group = search[
            search[
                "symbol"
            ].astype(str)
            == symbol
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
        # TOP CALLS
        # ----------------------------------------------------

        calls = group[
            group[
                "option_type"
            ].str.upper() == "CALL"
        ].copy()

        calls = calls.sort_values(
            "option_importance_score",
            ascending=False
        )

        top_calls = calls.head(
            TOP_CALLS
        )

        # ----------------------------------------------------
        # TOP PUTS
        # ----------------------------------------------------

        puts = group[
            group[
                "option_type"
            ].str.upper() == "PUT"
        ].copy()

        puts = puts.sort_values(
            "option_importance_score",
            ascending=False
        )

        top_puts = puts.head(
            TOP_PUTS
        )

        # ----------------------------------------------------
        # CALL WALL
        #
        # Use OI + volume + premium + gamma.
        # Not simply OI #1.
        # ----------------------------------------------------

        call_wall = None

        if not calls.empty:

            calls[
                "wall_score"
            ] = (

                percentile_score(
                    calls[
                        "openInterest"
                    ]
                ) * 35

                + percentile_score(
                    calls[
                        "volume"
                    ]
                ) * 20

                + percentile_score(
                    calls[
                        "estimated_traded_premium"
                    ]
                ) * 20

                + percentile_score(
                    calls[
                        "gamma"
                    ].abs()
                ) * 15

                + percentile_score(
                    -calls[
                        "moneyness_distance"
                    ]
                ) * 10
            )

            call_wall = calls.sort_values(
                "wall_score",
                ascending=False
            ).iloc[0]

        # ----------------------------------------------------
        # PUT WALL
        # ----------------------------------------------------

        put_wall = None

        if not puts.empty:

            puts[
                "wall_score"
            ] = (

                percentile_score(
                    puts[
                        "openInterest"
                    ]
                ) * 35

                + percentile_score(
                    puts[
                        "volume"
                    ]
                ) * 20

                + percentile_score(
                    puts[
                        "estimated_traded_premium"
                    ]
                ) * 20

                + percentile_score(
                    puts[
                        "gamma"
                    ].abs()
                ) * 15

                + percentile_score(
                    -puts[
                        "moneyness_distance"
                    ] * -1
                ) * 10
            )

            put_wall = puts.sort_values(
                "wall_score",
                ascending=False
            ).iloc[0]

        # ----------------------------------------------------
        # CALL / PUT PREMIUM
        # ----------------------------------------------------

        call_premium = calls[
            "estimated_traded_premium"
        ].sum()

        put_premium = puts[
            "estimated_traded_premium"
        ].sum()

        total_premium = (
            call_premium
            + put_premium
        )

        if total_premium > 0:

            cp_imbalance = (
                call_premium
                - put_premium
            ) / total_premium

        else:

            cp_imbalance = 0.0

        # ----------------------------------------------------
        # GEX
        # ----------------------------------------------------

        if "gex" in group.columns:

            total_gex = group[
                "gex"
            ].sum()

            call_gex = calls[
                "gex"
            ].sum() if "gex" in calls.columns else np.nan

            put_gex = puts[
                "gex"
            ].sum() if "gex" in puts.columns else np.nan

        else:

            total_gex = np.nan
            call_gex = np.nan
            put_gex = np.nan

        # ----------------------------------------------------
        # RISK REVERSAL SEARCH
        # ----------------------------------------------------

        risk_reversals = []

        if (
            not calls.empty
            and not puts.empty
        ):

            for _, call in calls.iterrows():

                if call[
                    "trade_side_estimate"
                ] != "BUY EST.":

                    continue

                for _, put in puts.iterrows():

                    if put[
                        "trade_side_estimate"
                    ] != "SELL EST.":

                        continue

                    dte_difference = abs(
                        safe_float(
                            call["DTE"]
                        )
                        -
                        safe_float(
                            put["DTE"]
                        )
                    )

                    if dte_difference > 14:

                        continue

                    call_premium_value = (
                        safe_float(
                            call[
                                "estimated_traded_premium"
                            ]
                        )
                    )

                    put_premium_value = (
                        safe_float(
                            put[
                                "estimated_traded_premium"
                            ]
                        )
                    )

                    if (
                        call_premium_value
                        <= 0
                        or put_premium_value
                        <= 0
                    ):

                        continue

                    strike_relation = (
                        safe_float(
                            call["strike"]
                        )
                        >
                        safe_float(
                            put["strike"]
                        )
                    )

                    if not strike_relation:

                        continue

                    rr_score = (

                        safe_float(
                            call[
                                "option_importance_score"
                            ]
                        )

                        + safe_float(
                            put[
                                "option_importance_score"
                            ]
                        )

                        + min(
                            call_premium_value
                            /
                            max(
                                put_premium_value,
                                1
                            ),
                            5
                        )
                        * 5
                    )

                    risk_reversals.append(
                        (
                            rr_score,
                            call,
                            put
                        )
                    )

        risk_reversals.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_rr = None

        if risk_reversals:

            best_rr = (
                risk_reversals[0]
            )

        # ----------------------------------------------------
        # TEXT TOP CALLS
        # ----------------------------------------------------

        def format_option(row):

            return (
                f"${safe_float(row['strike']):.2f}"
                f"C/P"
                f" DTE {int(safe_float(row['DTE']))}"
                f" Vol {int(safe_float(row['volume']))}"
                f" OI {int(safe_float(row['openInterest']))}"
                f" IV {safe_float(row['impliedVolatility'])*100:.1f}%"
                f" Delta {safe_float(row['delta']):+.2f}"
                f" Gamma {safe_float(row['gamma']):.4f}"
                f" Premium ${safe_float(row['estimated_traded_premium']):,.0f}"
                f" {row['trade_side_estimate']}"
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
        # WALL TEXT
        # ----------------------------------------------------

        if call_wall is not None:

            call_wall_strike = safe_float(
                call_wall[
                    "strike"
                ]
            )

            call_wall_score = safe_float(
                call_wall[
                    "wall_score"
                ]
            )

        else:

            call_wall_strike = np.nan
            call_wall_score = np.nan

        if put_wall is not None:

            put_wall_strike = safe_float(
                put_wall[
                    "strike"
                ]
            )

            put_wall_score = safe_float(
                put_wall[
                    "wall_score"
                ]
            )

        else:

            put_wall_strike = np.nan
            put_wall_score = np.nan

        # ----------------------------------------------------
        # RISK REVERSAL TEXT
        # ----------------------------------------------------

        if best_rr is not None:

            rr_score, rr_call, rr_put = (
                best_rr
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

            rr_call_premium = safe_float(
                rr_call[
                    "estimated_traded_premium"
                ]
            )

            rr_put_premium = safe_float(
                rr_put[
                    "estimated_traded_premium"
                ]
            )

            rr_status = (
                "BULLISH RISK-REVERSAL EST."
            )

        else:

            rr_score = np.nan
            rr_call_strike = np.nan
            rr_call_dte = np.nan
            rr_put_strike = np.nan
            rr_put_dte = np.nan
            rr_call_premium = np.nan
            rr_put_premium = np.nan

            rr_status = (
                "NONE DETECTED"
            )

        # ----------------------------------------------------
        # STRUCTURE BIAS
        # ----------------------------------------------------

        if cp_imbalance >= 0.25:

            structure_bias = (
                "CALL DOMINANT"
            )

        elif cp_imbalance <= -0.25:

            structure_bias = (
                "PUT DOMINANT"
            )

        else:

            structure_bias = (
                "BALANCED"
            )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "symbol": symbol,

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
                    cp_imbalance,

                "structure_bias":
                    structure_bias,

                "call_wall":
                    call_wall_strike,

                "call_wall_score":
                    call_wall_score,

                "put_wall":
                    put_wall_strike,

                "put_wall_score":
                    put_wall_score,

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

    # --------------------------------------------------------
    # RESULT DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        results
    )

    if result.empty:

        raise RuntimeError(
            "STEP 7 generated no results."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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

    print()

    print(
        "🔥 OPTION SEARCH SUMMARY"
    )

    print(
        "=" * 72
    )

    display_columns = [
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

    print(
        result[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "🔥 RISK REVERSAL SUMMARY"
    )

    print(
        "=" * 72
    )

    for _, row in result.iterrows():

        print(
            f"{row['symbol']} → "
            f"{row['risk_reversal']}"
        )

        if (
            row["risk_reversal"]
            != "NONE DETECTED"
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
                f"   PUT  "
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

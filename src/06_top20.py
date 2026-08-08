import os
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/analysis/unusual_flow.csv"

OUTPUT_DIR = "data/analysis"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "top20.csv"
)

TOP_N = 20


# ============================================================
# LOG
# ============================================================

def log(message):

    now = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[06 TOP20] {now} | {message}"
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
# PERCENTILE SCORE
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
        "option_type",
        "volume",
        "openInterest",
        "DTE",
        "estimated_traded_premium",
        "flow_score",
        "trade_side_estimate",
        "call_put_premium_imbalance",
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
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = [
        "volume",
        "openInterest",
        "DTE",
        "estimated_traded_premium",
        "flow_score",
        "call_put_premium_imbalance",
    ]

    for column in numeric_columns:

        df[column] = numeric(
            df[column]
        )

    # --------------------------------------------------------
    # SYMBOL GROUPING
    # --------------------------------------------------------

    log(
        "BUILDING SYMBOL LEVEL FLOW"
    )

    grouped = []

    for symbol, group in df.groupby(
        "symbol",
        dropna=False
    ):

        total_volume = group[
            "volume"
        ].sum()

        total_premium = group[
            "estimated_traded_premium"
        ].sum()

        call_group = group[
            group["option_type"] == "CALL"
        ]

        put_group = group[
            group["option_type"] == "PUT"
        ]

        call_premium = call_group[
            "estimated_traded_premium"
        ].sum()

        put_premium = put_group[
            "estimated_traded_premium"
        ].sum()

        # ----------------------------------------------------
        # BUY / SELL ESTIMATE
        # ----------------------------------------------------

        buy_group = group[
            group[
                "trade_side_estimate"
            ] == "BUY EST."
        ]

        sell_group = group[
            group[
                "trade_side_estimate"
            ] == "SELL EST."
        ]

        buy_premium = buy_group[
            "estimated_traded_premium"
        ].sum()

        sell_premium = sell_group[
            "estimated_traded_premium"
        ].sum()

        # ----------------------------------------------------
        # VOLUME / OI
        # ----------------------------------------------------

        volume_oi = np.where(

            group["openInterest"] > 0,

            group["volume"]
            / group["openInterest"],

            np.nan
        )

        max_volume_oi = (
            np.nanmax(volume_oi)
            if np.isfinite(
                volume_oi
            ).any()
            else 0.0
        )

        avg_volume_oi = (
            np.nanmean(volume_oi)
            if np.isfinite(
                volume_oi
            ).any()
            else 0.0
        )

        # ----------------------------------------------------
        # FLOW SCORE
        # ----------------------------------------------------

        max_flow_score = group[
            "flow_score"
        ].max()

        avg_flow_score = group[
            "flow_score"
        ].mean()

        # ----------------------------------------------------
        # TOP OPTION
        # ----------------------------------------------------

        top_option = group.sort_values(
            "flow_score",
            ascending=False
        ).iloc[0]

        # ----------------------------------------------------
        # DTE
        # ----------------------------------------------------

        top_dte = top_option[
            "DTE"
        ]

        # ----------------------------------------------------
        # CALL / PUT IMBALANCE
        # ----------------------------------------------------

        if (
            call_premium
            + put_premium
        ) > 0:

            cp_imbalance = (
                call_premium
                - put_premium
            ) / (
                call_premium
                + put_premium
            )

        else:

            cp_imbalance = 0.0

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

        if cp_imbalance >= 0.25:

            flow_direction = (
                "CALL DOMINANT"
            )

        elif cp_imbalance <= -0.25:

            flow_direction = (
                "PUT DOMINANT"
            )

        else:

            flow_direction = (
                "BALANCED"
            )

        # ----------------------------------------------------
        # BUY / SELL DIRECTION
        # ----------------------------------------------------

        directional_premium = (
            buy_premium
            - sell_premium
        )

        if (
            buy_premium
            + sell_premium
        ) > 0:

            directional_ratio = (
                directional_premium
                /
                (
                    buy_premium
                    + sell_premium
                )
            )

        else:

            directional_ratio = 0.0

        if directional_ratio >= 0.25:

            estimated_direction = (
                "BUY EST. DOMINANT"
            )

        elif directional_ratio <= -0.25:

            estimated_direction = (
                "SELL EST. DOMINANT"
            )

        else:

            estimated_direction = (
                "MIXED / UNKNOWN"
            )

        # ----------------------------------------------------
        # TOP OPTIONS
        # ----------------------------------------------------

        top_calls = group[
            group["option_type"] == "CALL"
        ].sort_values(
            "flow_score",
            ascending=False
        ).head(3)

        top_puts = group[
            group["option_type"] == "PUT"
        ].sort_values(
            "flow_score",
            ascending=False
        ).head(3)

        def option_text(row):

            return (
                f"{row['option_type']} "
                f"${row['strike']:.2f} "
                f"DTE {int(row['DTE'])} "
                f"Score {row['flow_score']:.1f}"
            )

        top_call_text = " / ".join(
            option_text(row)
            for _, row
            in top_calls.iterrows()
        )

        top_put_text = " / ".join(
            option_text(row)
            for _, row
            in top_puts.iterrows()
        )

        # ----------------------------------------------------
        # REASON
        # ----------------------------------------------------

        reasons = []

        if max_flow_score >= 90:

            reasons.append(
                "Very high option flow score"
            )

        elif max_flow_score >= 75:

            reasons.append(
                "High option flow score"
            )

        if max_volume_oi >= 1:

            reasons.append(
                "Volume/OI surge"
            )

        elif max_volume_oi >= 0.5:

            reasons.append(
                "Elevated Volume/OI"
            )

        if total_premium > 0:

            reasons.append(
                "Large estimated premium"
            )

        if cp_imbalance >= 0.25:

            reasons.append(
                "Call premium dominance"
            )

        elif cp_imbalance <= -0.25:

            reasons.append(
                "Put premium dominance"
            )

        if directional_ratio >= 0.25:

            reasons.append(
                "Buy-side estimate dominance"
            )

        elif directional_ratio <= -0.25:

            reasons.append(
                "Sell-side estimate dominance"
            )

        if not reasons:

            reasons.append(
                "Unusual option activity"
            )

        selection_reason = " | ".join(
            reasons[:5]
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        grouped.append(
            {
                "symbol": symbol,

                "option_count":
                    len(group),

                "total_volume":
                    total_volume,

                "total_premium":
                    total_premium,

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "call_put_imbalance":
                    cp_imbalance,

                "max_volume_oi":
                    max_volume_oi,

                "avg_volume_oi":
                    avg_volume_oi,

                "max_flow_score":
                    max_flow_score,

                "avg_flow_score":
                    avg_flow_score,

                "buy_premium_est":
                    buy_premium,

                "sell_premium_est":
                    sell_premium,

                "directional_ratio":
                    directional_ratio,

                "flow_direction":
                    flow_direction,

                "estimated_direction":
                    estimated_direction,

                "top_dte":
                    top_dte,

                "top_call_options":
                    top_call_text,

                "top_put_options":
                    top_put_text,

                "selection_reason":
                    selection_reason,
            }
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        grouped
    )

    if result.empty:

        raise RuntimeError(
            "No symbol-level flow data generated."
        )

    # --------------------------------------------------------
    # SYMBOL SCORE
    # --------------------------------------------------------

    result[
        "premium_score"
    ] = percentile_score(
        np.log1p(
            result[
                "total_premium"
            ].clip(
                lower=0
            )
        )
    )

    result[
        "volume_score"
    ] = percentile_score(
        np.log1p(
            result[
                "total_volume"
            ].clip(
                lower=0
            )
        )
    )

    result[
        "flow_score_score"
    ] = percentile_score(
        result[
            "max_flow_score"
        ]
    )

    result[
        "volume_oi_score"
    ] = percentile_score(
        np.log1p(
            result[
                "max_volume_oi"
            ].clip(
                lower=0
            )
        )
    )

    result[
        "imbalance_score"
    ] = (
        result[
            "call_put_imbalance"
        ]
        .abs()
        .clip(
            lower=0,
            upper=1
        )
    )

    # --------------------------------------------------------
    # FINAL SYMBOL SCORE
    # --------------------------------------------------------

    result[
        "top20_score"
    ] = (

        result[
            "premium_score"
        ] * 35

        + result[
            "volume_score"
        ] * 15

        + result[
            "flow_score_score"
        ] * 30

        + result[
            "volume_oi_score"
        ] * 10

        + result[
            "imbalance_score"
        ] * 10
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "top20_score",
            "max_flow_score",
            "total_premium",
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TOP 20
    # --------------------------------------------------------

    top20 = result.head(
        TOP_N
    ).copy()

    top20[
        "rank"
    ] = np.arange(
        1,
        len(top20) + 1
    )

    # --------------------------------------------------------
    # FINAL COLUMN ORDER
    # --------------------------------------------------------

    columns = [
        "rank",
        "symbol",
        "top20_score",
        "max_flow_score",
        "avg_flow_score",
        "total_volume",
        "total_premium",
        "call_premium",
        "put_premium",
        "call_put_imbalance",
        "max_volume_oi",
        "avg_volume_oi",
        "buy_premium_est",
        "sell_premium_est",
        "directional_ratio",
        "flow_direction",
        "estimated_direction",
        "top_dte",
        "top_call_options",
        "top_put_options",
        "selection_reason",
    ]

    top20 = top20[
        columns
    ]

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    top20.to_csv(
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
        "🔎 STEP 6 VALIDATION"
    )

    print(
        "=" * 72
    )

    print(
        f"INPUT ROWS       : "
        f"{input_rows:,}"
    )

    print(
        f"INPUT TICKERS    : "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"TOP20 ROWS       : "
        f"{len(top20):,}"
    )

    print(
        f"OUTPUT FILE      : "
        f"{OUTPUT_FILE}"
    )

    print(
        "=" * 72
    )

    print(
        "🔥 TOP 20 UNUSUAL OPTION FLOW"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "rank",
        "symbol",
        "top20_score",
        "max_flow_score",
        "total_premium",
        "call_put_imbalance",
        "max_volume_oi",
        "estimated_direction",
    ]

    print(
        top20[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "=" * 72
    )

    print(
        "📌 SELECTION REASONS"
    )

    print(
        "=" * 72
    )

    for _, row in top20.iterrows():

        print(
            f"{int(row['rank']):02d}. "
            f"{row['symbol']} → "
            f"{row['selection_reason']}"
        )

    print(
        "=" * 72
    )

    if len(top20) > 0:

        print(
            "TOP20 CHECK      : OK"
        )

    else:

        raise RuntimeError(
            "TOP20 selection failed."
        )

    print(
        "=" * 72
    )

    log(
        "STEP 6 TOP20 COMPLETE"
    )


if __name__ == "__main__":
    main()

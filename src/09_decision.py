from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# PATH
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

MARKET_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv",
)

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv",
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv",
)

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv",
)


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
        f"[09 DECISION] {now} | {message}"
    )


# ============================================================
# HELPERS
# ============================================================

def find_col(df, names):

    normalized = {
        str(c)
        .strip()
        .lower()
        .replace(" ", "_"): c
        for c in df.columns
    }

    for name in names:

        key = (
            name
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

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def clamp(value):

    if not np.isfinite(value):
        return 50.0

    return max(
        0.0,
        min(
            100.0,
            float(value),
        ),
    )


def text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .upper()
        .strip()
    )


# ============================================================
# MARKET
# ============================================================

def load_market():

    df = pd.read_csv(
        MARKET_FILE
    )

    score_col = find_col(
        df,
        [
            "market_score",
            "market_regime_score",
            "score",
        ],
    )

    if score_col is None:

        raise RuntimeError(
            "market_score missing"
        )

    score = numeric(
        df.iloc[-1][
            score_col
        ]
    )

    regime_col = find_col(
        df,
        [
            "market_regime",
            "regime",
        ],
    )

    regime = (
        text(
            df.iloc[-1][
                regime_col
            ]
        )
        if regime_col
        else
        "UNKNOWN"
    )

    directions = {}

    for name in [
        "ndx_direction",
        "spy_direction",
        "soxx_direction",
        "dia_direction",
    ]:

        col = find_col(
            df,
            [name],
        )

        directions[name] = (

            text(
                df.iloc[-1][col]
            )

            if col

            else

            "UNAVAILABLE"
        )

    return {
        "market_score": clamp(
            score
        ),

        "market_regime": regime,

        **directions,
    }


# ============================================================
# TOP20
# ============================================================

def load_top20():

    df = pd.read_csv(
        TOP20_FILE
    )

    ticker_col = find_col(
        df,
        [
            "ticker",
            "symbol",
        ],
    )

    if ticker_col is None:

        raise RuntimeError(
            "TOP20 ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_fields = [

        "top20_score",
        "max_flow_score",
        "avg_flow_score",

        "total_volume",
        "total_premium",

        "call_premium",
        "put_premium",

        "call_put_imbalance",
        "directional_ratio",
        "top_dte",
    ]

    for field in numeric_fields:

        col = find_col(
            df,
            [field],
        )

        if col:

            df[field] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        else:

            df[field] = np.nan

    text_fields = [

        "flow_direction",
        "estimated_direction",
    ]

    for field in text_fields:

        col = find_col(
            df,
            [field],
        )

        if col:

            df[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

        else:

            df[field] = ""

    return (
        df
        .drop_duplicates(
            "ticker"
        )
        .head(20)
    )


# ============================================================
# FLOW
# ============================================================

def load_flow():

    df = pd.read_csv(
        FLOW_FILE
    )

    ticker_col = find_col(
        df,
        [
            "ticker",
            "symbol",
        ],
    )

    score_col = find_col(
        df,
        [
            "flow_score",
            "option_flow_score",
            "score",
        ],
    )

    if ticker_col is None:

        raise RuntimeError(
            "FLOW ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if score_col:

        df["flow_value"] = (
            pd.to_numeric(
                df[score_col],
                errors="coerce",
            )
        )

    else:

        df["flow_value"] = np.nan

    return (
        df
        .groupby(
            "ticker",
            as_index=False,
        )
        .agg(
            flow_score=(
                "flow_value",
                "max",
            )
        )
    )


# ============================================================
# STRUCTURE
# ============================================================

def load_structure():

    df = pd.read_csv(
        STRUCTURE_FILE
    )

    ticker_col = find_col(
        df,
        [
            "ticker",
            "symbol",
        ],
    )

    if ticker_col is None:

        raise RuntimeError(
            "STRUCTURE ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_fields = [

        "current_price",

        "call_wall",
        "put_wall",

        "support",
        "resistance",

        "call_gex",
        "put_gex",
        "net_gex",
    ]

    for field in numeric_fields:

        col = find_col(
            df,
            [field],
        )

        if col:

            df[field] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        else:

            df[field] = np.nan

    text_fields = [

        "structure",
        "price_location",
        "gex_structure",
        "wall_structure",
    ]

    for field in text_fields:

        col = find_col(
            df,
            [field],
        )

        if col:

            df[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )

        else:

            df[field] = ""

    return (
        df
        .drop_duplicates(
            "ticker"
        )
    )


# ============================================================
# DIRECTION SCORE
# ============================================================

def direction_score(row):

    direction = text(
        row[
            "estimated_direction"
        ]
    )

    flow_direction = text(
        row[
            "flow_direction"
        ]
    )

    ratio = numeric(
        row[
            "directional_ratio"
        ]
    )

    if "BUY" in direction:

        base = 80.0

    elif "BULL" in direction:

        base = 80.0

    elif "SELL" in direction:

        base = 20.0

    elif "BEAR" in direction:

        base = 20.0

    elif "MIXED" in direction:

        base = 50.0

    else:

        base = 50.0

    if np.isfinite(ratio):

        ratio = clamp(
            ratio
        )

        base = (
            base * 0.65
            +
            ratio * 0.35
        )

    if (
        "CALL DOMINANT"
        in flow_direction
    ):

        base += 5.0

    elif (
        "PUT DOMINANT"
        in flow_direction
    ):

        base -= 5.0

    return clamp(
        base
    )


# ============================================================
# GEX STRENGTH
#
# This is the important improvement.
#
# We do NOT simply use:
#
# positive = 70
# negative = 30
#
# Instead:
#
# GEX BIAS =
#
# NET GEX
# /
# (ABS(CALL GEX) + ABS(PUT GEX))
#
# This gives a normalized -1 ~ +1 measure.
#
# +1 = strongly call-heavy
#  0 = balanced
# -1 = strongly put-heavy
#
# Then convert to 0 ~ 100.
# ============================================================

def calculate_gex_strength(row):

    call_gex = numeric(
        row["call_gex"]
    )

    put_gex = numeric(
        row["put_gex"]
    )

    net_gex = numeric(
        row["net_gex"]
    )

    if not all(
        np.isfinite(v)
        for v in [
            call_gex,
            put_gex,
            net_gex,
        ]
    ):

        return 50.0

    denominator = (
        abs(call_gex)
        +
        abs(put_gex)
    )

    if denominator <= 0:

        return 50.0

    bias = (
        net_gex
        /
        denominator
    )

    bias = max(
        -1.0,
        min(
            1.0,
            bias,
        ),
    )

    return (
        50.0
        +
        bias * 50.0
    )


# ============================================================
# GEX STRUCTURE SCORE
# ============================================================

def structure_score(row):

    structure = text(
        row["structure"]
    )

    wall = text(
        row["wall_structure"]
    )

    gex = text(
        row["gex_structure"]
    )

    # --------------------------------------------------------
    # BASE STRUCTURE
    # --------------------------------------------------------

    if structure == "BULLISH":

        base = 82.0

    elif structure == "BEARISH":

        base = 22.0

    elif (
        "POSITIVE GEX STRUCTURE"
        in structure
    ):

        base = 65.0

    elif (
        "NEGATIVE GEX STRUCTURE"
        in structure
    ):

        base = 35.0

    else:

        base = 50.0

    # --------------------------------------------------------
    # GEX INTENSITY
    # --------------------------------------------------------

    gex_strength = (
        calculate_gex_strength(
            row
        )
    )

    # GEX contribution:
    #
    # 50 = neutral
    # 100 = extremely positive
    #   0 = extremely negative
    #
    # Blend rather than blindly add 5 points.

    base = (
        base * 0.65
        +
        gex_strength * 0.35
    )

    # --------------------------------------------------------
    # WALL
    # --------------------------------------------------------

    if wall == "BULLISH BREAKOUT":

        base += 8.0

    elif wall == "ABOVE CALL WALL":

        base += 5.0

    elif wall == "BEARISH BREAKDOWN":

        base -= 8.0

    elif wall == "BELOW PUT WALL":

        base -= 5.0

    # --------------------------------------------------------
    # EXPLICIT GEX DIRECTION
    # --------------------------------------------------------

    if gex == "POSITIVE GEX":

        base += 2.0

    elif gex == "NEGATIVE GEX":

        base -= 2.0

    return clamp(
        base
    )


# ============================================================
# PRICE SCORE
# ============================================================

def price_score(row):

    price = numeric(
        row["current_price"]
    )

    support = numeric(
        row["support"]
    )

    resistance = numeric(
        row["resistance"]
    )

    if not np.isfinite(price):

        return 50.0

    score = 50.0

    if np.isfinite(support):

        d = (
            price - support
        ) / price

        if 0 <= d <= 0.02:

            score += 12.0

        elif d > 0:

            score += 5.0

        else:

            score -= 10.0

    if np.isfinite(resistance):

        d = (
            resistance - price
        ) / price

        if 0 <= d <= 0.02:

            score += 8.0

        elif d > 0:

            score += 4.0

        else:

            score -= 8.0

    return clamp(
        score
    )


# ============================================================
# INDEX SCORE
# ============================================================

def index_score(market):

    values = [

        market[
            "ndx_direction"
        ],

        market[
            "spy_direction"
        ],

        market[
            "soxx_direction"
        ],

        market[
            "dia_direction"
        ],
    ]

    bull = sum(
        "BULL" in value
        for value in values
    )

    bear = sum(
        "BEAR" in value
        for value in values
    )

    neutral = (
        len(values)
        -
        bull
        -
        bear
    )

    if bull >= 3 and bear == 0:

        score = 90.0

    elif bull > bear:

        score = 70.0

    elif bear >= 3 and bull == 0:

        score = 20.0

    elif bear > bull:

        score = 35.0

    else:

        score = 50.0

    return (
        clamp(score),
        bull,
        bear,
        neutral,
    )


# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(
    score,
    direction,
    structure,
    market_score,
):

    structure = text(
        structure
    )

    if direction < 30:

        return "🟡 관망"

    if structure == "BEARISH":

        return "🔴 회피"

    if market_score < 45:

        if (
            score >= 82
            and
            direction >= 70
        ):

            return "🟡 관망"

        return "🔴 회피"

    if (
        score >= 75
        and
        direction >= 55
    ):

        return "🟢 진입"

    if score >= 55:

        return "🟡 관망"

    return "🔴 회피"


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # ========================================================
    # INPUT CHECK
    # ========================================================

    for path in [

        MARKET_FILE,
        FLOW_FILE,
        TOP20_FILE,
        STRUCTURE_FILE,

    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                path
            )

    market = load_market()

    top20 = load_top20()

    flow = load_flow()

    structure = load_structure()

    market_component = (
        market[
            "market_score"
        ]
    )

    (
        index_component,
        bull,
        bear,
        neutral,
    ) = index_score(
        market
    )

    rows = []

    # ========================================================
    # DECISION LOOP
    # ========================================================

    for rank, top in enumerate(

        top20.to_dict(
            "records"
        ),

        start=1,
    ):

        ticker = top[
            "ticker"
        ]

        flow_row = flow[
            flow["ticker"]
            ==
            ticker
        ]

        structure_row = structure[
            structure["ticker"]
            ==
            ticker
        ]

        # ====================================================
        # FLOW
        # ====================================================

        if flow_row.empty:

            flow_score = numeric(
                top[
                    "max_flow_score"
                ]
            )

        else:

            flow_score = numeric(
                flow_row.iloc[0][
                    "flow_score"
                ]
            )

        if not np.isfinite(
            flow_score
        ):

            flow_score = numeric(
                top[
                    "avg_flow_score"
                ]
            )

        if not np.isfinite(
            flow_score
        ):

            flow_score = 50.0

        # ====================================================
        # STRUCTURE
        # ====================================================

        if structure_row.empty:

            s = {

                "current_price":
                    np.nan,

                "call_wall":
                    np.nan,

                "put_wall":
                    np.nan,

                "support":
                    np.nan,

                "resistance":
                    np.nan,

                "call_gex":
                    np.nan,

                "put_gex":
                    np.nan,

                "net_gex":
                    np.nan,

                "structure":
                    "NO DATA",

                "price_location":
                    "",

                "gex_structure":
                    "GEX UNAVAILABLE",

                "wall_structure":
                    "WALL UNAVAILABLE",
            }

        else:

            s = (
                structure_row
                .iloc[0]
                .to_dict()
            )

        # ====================================================
        # COMPONENTS
        # ====================================================

        direction = (
            direction_score(
                top
            )
        )

        structure_component = (
            structure_score(
                s
            )
        )

        price_component = (
            price_score(
                s
            )
        )

        top20_score = numeric(
            top[
                "top20_score"
            ]
        )

        if np.isfinite(
            top20_score
        ):

            top20_component = clamp(
                top20_score
            )

        else:

            top20_component = (
                flow_score
            )

        # ====================================================
        # FINAL SCORE
        #
        # Market       20%
        # Flow         25%
        # Direction    20%
        # Structure    10%
        # Price        15%
        # Index        10%
        #
        # Structure now contains a meaningful GEX intensity
        # component rather than a simple +/- fixed value.
        # ====================================================

        score = (

            market_component
            * 0.20

            +

            flow_score
            * 0.25

            +

            direction
            * 0.20

            +

            structure_component
            * 0.10

            +

            price_component
            * 0.15

            +

            index_component
            * 0.10
        )

        score = clamp(
            score
        )

        decision = final_decision(

            score,

            direction,

            s[
                "structure"
            ],

            market_component,
        )

        # ====================================================
        # REASONS
        # ====================================================

        reasons = []

        if market_component >= 70:

            reasons.append(
                "Strong market regime"
            )

        elif market_component < 45:

            reasons.append(
                "Weak market regime"
            )

        if flow_score >= 80:

            reasons.append(
                "Very high option flow"
            )

        elif flow_score >= 65:

            reasons.append(
                "Strong option flow"
            )

        direction_text = text(
            top[
                "estimated_direction"
            ]
        )

        if (
            "BUY" in direction_text
            or
            "BULL" in direction_text
        ):

            reasons.append(
                "Buy-side estimate"
            )

        elif (
            "SELL" in direction_text
            or
            "BEAR" in direction_text
        ):

            reasons.append(
                "Sell-side estimate"
            )

        elif "MIXED" in direction_text:

            reasons.append(
                "Mixed directional flow"
            )

        flow_direction = text(
            top[
                "flow_direction"
            ]
        )

        if (
            "CALL DOMINANT"
            in flow_direction
        ):

            reasons.append(
                "Call dominant flow"
            )

        elif (
            "PUT DOMINANT"
            in flow_direction
        ):

            reasons.append(
                "Put dominant flow"
            )

        elif (
            "BALANCED"
            in flow_direction
        ):

            reasons.append(
                "Balanced flow"
            )

        # ====================================================
        # STRUCTURE
        # ====================================================

        structure_text = text(
            s[
                "structure"
            ]
        )

        gex_text = text(
            s[
                "gex_structure"
            ]
        )

        wall_text = text(
            s[
                "wall_structure"
            ]
        )

        if structure_text:

            reasons.append(
                structure_text.replace(
                    "_",
                    " ",
                )
            )

        # ----------------------------------------------------
        # GEX DETAIL
        # ----------------------------------------------------

        gex_strength = (
            calculate_gex_strength(
                s
            )
        )

        if gex_text == "POSITIVE GEX":

            reasons.append(
                "Positive GEX "
                f"{gex_strength:.1f}"
            )

        elif gex_text == "NEGATIVE GEX":

            reasons.append(
                "Negative GEX "
                f"{gex_strength:.1f}"
            )

        else:

            reasons.append(
                "GEX unavailable"
            )

        if wall_text:

            reasons.append(
                wall_text.replace(
                    "_",
                    " ",
                )
            )

        # ====================================================
        # PRICE
        # ====================================================

        price_location = text(
            s[
                "price_location"
            ]
        )

        if price_location:

            for item in (
                price_location.split(
                    "|"
                )
            ):

                item = item.strip()

                if item:

                    reasons.append(
                        item
                    )

        # ====================================================
        # INDEX
        # ====================================================

        reasons.append(
            "Index alignment "
            f"BULL {bull} / "
            f"BEAR {bear} / "
            f"NEUTRAL {neutral}"
        )

        # ====================================================
        # OUTPUT ROW
        # ====================================================

        rows.append({

            "rank":
                rank,

            "ticker":
                ticker,

            "market_score":
                market_component,

            "market_regime":
                market[
                    "market_regime"
                ],

            "ndx_direction":
                market[
                    "ndx_direction"
                ],

            "spy_direction":
                market[
                    "spy_direction"
                ],

            "soxx_direction":
                market[
                    "soxx_direction"
                ],

            "dia_direction":
                market[
                    "dia_direction"
                ],

            "top20_score":
                top20_score,

            "flow_score":
                flow_score,

            "direction_score":
                direction,

            "structure_score":
                structure_component,

            "price_score":
                price_component,

            "index_score":
                index_component,

            "current_price":
                numeric(
                    s[
                        "current_price"
                    ]
                ),

            "call_wall":
                numeric(
                    s[
                        "call_wall"
                    ]
                ),

            "put_wall":
                numeric(
                    s[
                        "put_wall"
                    ]
                ),

            "support":
                numeric(
                    s[
                        "support"
                    ]
                ),

            "resistance":
                numeric(
                    s[
                        "resistance"
                    ]
                ),

            "call_gex":
                numeric(
                    s[
                        "call_gex"
                    ]
                ),

            "put_gex":
                numeric(
                    s[
                        "put_gex"
                    ]
                ),

            "net_gex":
                numeric(
                    s[
                        "net_gex"
                    ]
                ),

            "structure":
                s[
                    "structure"
                ],

            "price_location":
                s[
                    "price_location"
                ],

            "gex_structure":
                s[
                    "gex_structure"
                ],

            "wall_structure":
                s[
                    "wall_structure"
                ],

            "index_bull":
                bull,

            "index_bear":
                bear,

            "index_neutral":
                neutral,

            "decision_score":
                score,

            "decision":
                decision,

            "reason":
                " | ".join(
                    reasons
                ),

            "data_source":
                "STEP8_GEX_PROXY_INTENSITY",
        })

    # ========================================================
    # OUTPUT
    # ========================================================

    output = pd.DataFrame(
        rows
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(output) != 20:

        raise RuntimeError(
            "STEP 9 must contain exactly 20 rows"
        )

    if (
        output[
            "ticker"
        ].nunique()
        != 20
    ):

        raise RuntimeError(
            "STEP 9 must contain 20 unique tickers"
        )

    if (
        output[
            "decision_score"
        ].isna()
        .any()
    ):

        raise RuntimeError(
            "STEP 9 decision_score contains NaN"
        )

    if (
        (
            output[
                "decision_score"
            ] < 0
        )
        |
        (
            output[
                "decision_score"
            ] > 100
        )
    ).any():

        raise RuntimeError(
            "STEP 9 decision_score outside 0-100"
        )

    valid_decisions = {

        "🟢 진입",
        "🟡 관망",
        "🔴 회피",
    }

    invalid = (
        set(
            output[
                "decision"
            ]
            .astype(str)
            .str.strip()
        )
        -
        valid_decisions
    )

    if invalid:

        raise RuntimeError(
            "Invalid decision labels: "
            +
            ", ".join(
                sorted(
                    invalid
                )
            )
        )

    # ========================================================
    # SORT
    # ========================================================

    output = (
        output
        .sort_values(
            "decision_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    output["rank"] = (
        np.arange(
            1,
            len(output) + 1,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "STEP 9 DECISION"
    )

    print(
        "=========================================="
    )

    print(
        output[
            [
                "rank",
                "ticker",
                "direction_score",
                "structure_score",
                "decision_score",
                "decision",
                "gex_structure",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "GEX INTENSITY"
    )

    print()

    gex_table = []

    for _, row in output.iterrows():

        gex_table.append({

            "ticker":
                row["ticker"],

            "call_gex":
                row["call_gex"],

            "put_gex":
                row["put_gex"],

            "net_gex":
                row["net_gex"],

            "gex_strength":
                calculate_gex_strength(
                    row
                ),

            "gex_structure":
                row[
                    "gex_structure"
                ],
        })

    print(
        pd.DataFrame(
            gex_table
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "DECISION SUMMARY"
    )

    print()

    print(
        output[
            "decision"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print(
        "GEX SUMMARY"
    )

    print()

    print(
        output[
            "gex_structure"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    print(
        f"ROWS : {len(output)}"
    )

    print(
        f"COLUMNS : "
        f"{list(output.columns)}"
    )

    print(
        "STEP 9 OUTPUT : OK"
    )


if __name__ == "__main__":

    main()

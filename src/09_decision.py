import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# STEP 9 - FINAL DECISION ENGINE
#
# PURPOSE
# ------------------------------------------------------------
# TOP20 + STEP 1 MARKET REGIME + STEP 8 STRUCTURE
# 를 종합하여 최종 점수와 진입/관망/회피를 결정한다.
#
# IMPORTANT
# ------------------------------------------------------------
# STEP 8의 GEX가 0이면 억지로 Positive/Negative로
# 판단하지 않는다.
#
# 대신 실제 사용 가능한:
#   - Market
#   - Flow
#   - Flow Direction
#   - Price Location
#   - Support / Resistance
#   - Call Wall / Put Wall
#   - Wall Structure
#   - Index Alignment
# 을 이용한다.
# ============================================================


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
    "analysis"
)

MARKET_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv"
)

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
)


# ============================================================
# DECISION WEIGHTS
#
# TOTAL = 100
# ============================================================

MARKET_WEIGHT = 20.0
FLOW_WEIGHT = 25.0
DIRECTION_WEIGHT = 15.0
STRUCTURE_WEIGHT = 15.0
PRICE_WEIGHT = 15.0
INDEX_WEIGHT = 10.0

ENTER_THRESHOLD = 75.0
WATCH_THRESHOLD = 55.0


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
# COLUMN FINDER
# ============================================================

def find_column(df, candidates):

    normalized = {
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): col
        for col in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
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
# TEXT NORMALIZER
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
# FILE CHECK
# ============================================================

def check_files():

    print()
    print("=" * 78)
    print("STEP 9 REQUIRED FILE CHECK")
    print("=" * 78)

    files = {

        "MARKET REGIME":
            MARKET_FILE,

        "UNUSUAL FLOW":
            FLOW_FILE,

        "TOP20":
            TOP20_FILE,

        "STRUCTURE":
            STRUCTURE_FILE,
    }

    for name, path in files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{name} file not found: {path}"
            )

        print(
            f"{name:<18} : OK"
        )

    print("=" * 78)


# ============================================================
# MARKET REGIME
# ============================================================

def load_market_regime():

    log(
        "Loading STEP 1 market regime"
    )

    df = pd.read_csv(
        MARKET_FILE
    )

    if df.empty:

        raise ValueError(
            "market_regime.csv is empty"
        )

    score_col = find_column(
        df,
        [
            "market_score",
            "market_regime_score",
            "score"
        ]
    )

    if score_col is None:

        raise ValueError(
            "market_score column not found"
        )

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce"
    ).dropna()

    if scores.empty:

        raise ValueError(
            "No valid market score"
        )

    market_score = float(
        scores.iloc[-1]
    )

    market_score = max(
        0.0,
        min(
            100.0,
            market_score
        )
    )

    regime_col = find_column(
        df,
        [
            "regime",
            "market_regime"
        ]
    )

    if regime_col is not None:

        regimes = (
            df[regime_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if not regimes.empty:

            regime = regimes.iloc[-1]

        else:

            regime = "UNKNOWN"

    else:

        regime = "UNKNOWN"

    latest = df.iloc[-1]

    direction_map = {

        "ndx_direction": [
            "ndx_direction"
        ],

        "spy_direction": [
            "spy_direction"
        ],

        "soxx_direction": [
            "soxx_direction"
        ],

        "dia_direction": [
            "dia_direction"
        ],
    }

    directions = {}

    for key, candidates in direction_map.items():

        column = find_column(
            df,
            candidates
        )

        if column is None:

            directions[key] = "UNAVAILABLE"

            continue

        value = latest[column]

        if pd.isna(value):

            directions[key] = "UNAVAILABLE"

        else:

            directions[key] = clean_text(
                value
            )

    print()
    print("=" * 78)
    print("STEP 1 MARKET REGIME")
    print("=" * 78)

    print(
        f"MARKET SCORE : {market_score:.2f}"
    )

    print(
        f"REGIME       : {regime}"
    )

    print(
        f"NDX          : "
        f"{directions['ndx_direction']}"
    )

    print(
        f"SPY          : "
        f"{directions['spy_direction']}"
    )

    print(
        f"SOXX         : "
        f"{directions['soxx_direction']}"
    )

    print(
        f"DIA          : "
        f"{directions['dia_direction']}"
    )

    print("=" * 78)

    return {

        "market_score":
            market_score,

        "market_regime":
            regime,

        **directions
    }


# ============================================================
# UNUSUAL FLOW
# ============================================================

def prepare_flow(df):

    ticker_col = find_column(
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

    if ticker_col is None:

        raise ValueError(
            "Ticker column missing "
            "in unusual_flow.csv"
        )

    flow_score_col = find_column(
        df,
        [
            "flow_score",
            "option_flow_score",
            "score"
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

    volume_oi_col = find_column(
        df,
        [
            "volume_oi",
            "volume_oi_ratio",
            "vol_oi"
        ]
    )

    result = pd.DataFrame()

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["flow_score"] = numeric(
        df,
        flow_score_col
    )

    result["premium"] = numeric(
        df,
        premium_col
    )

    result["volume_oi"] = numeric(
        df,
        volume_oi_col
    )

    grouped = (
        result
        .groupby(
            "ticker",
            as_index=False
        )
        .agg({

            "flow_score": "max",

            "premium": "sum",

            "volume_oi": "max"
        })
    )

    return grouped


def load_flow():

    log(
        "Loading unusual flow"
    )

    df = pd.read_csv(
        FLOW_FILE
    )

    if df.empty:

        raise ValueError(
            "unusual_flow.csv is empty"
        )

    result = prepare_flow(
        df
    )

    log(
        f"FLOW TICKERS : "
        f"{len(result)}"
    )

    return result


# ============================================================
# TOP20
#
# IMPORTANT
# ------------------------------------------------------------
# TOP20의 추가 정보를 그대로 보존한다.
#
# top20_score
# max_flow_score
# avg_flow_score
# call_put_imbalance
# directional_ratio
# flow_direction
# estimated_direction
# top_dte
# ============================================================

def load_top20():

    log(
        "Loading TOP20"
    )

    df = pd.read_csv(
        TOP20_FILE
    )

    if df.empty:

        raise ValueError(
            "top20.csv is empty"
        )

    ticker_col = find_column(
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

    if ticker_col is None:

        raise ValueError(
            "Unable to identify "
            "TOP20 ticker column"
        )

    result = df.copy()

    result["ticker"] = (
        result[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # NUMERIC TOP20 FIELDS
    # --------------------------------------------------------

    numeric_map = {

        "top20_score": [
            "top20_score",
            "score"
        ],

        "max_flow_score": [
            "max_flow_score"
        ],

        "avg_flow_score": [
            "avg_flow_score"
        ],

        "total_volume": [
            "total_volume"
        ],

        "total_premium": [
            "total_premium"
        ],

        "call_premium": [
            "call_premium"
        ],

        "put_premium": [
            "put_premium"
        ],

        "call_put_imbalance": [
            "call_put_imbalance"
        ],

        "max_volume_oi": [
            "max_volume_oi"
        ],

        "avg_volume_oi": [
            "avg_volume_oi"
        ],

        "directional_ratio": [
            "directional_ratio"
        ],

        "top_dte": [
            "top_dte"
        ],
    }

    for target, candidates in numeric_map.items():

        column = find_column(
            result,
            candidates
        )

        if column is None:

            result[target] = np.nan

        else:

            result[target] = numeric(
                result,
                column
            )

    # --------------------------------------------------------
    # TEXT TOP20 FIELDS
    # --------------------------------------------------------

    text_map = {

        "flow_direction": [
            "flow_direction"
        ],

        "estimated_direction": [
            "estimated_direction"
        ],

        "selection_reason": [
            "selection_reason"
        ],
    }

    for target, candidates in text_map.items():

        column = find_column(
            result,
            candidates
        )

        if column is None:

            result[target] = ""

        else:

            result[target] = (
                result[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # --------------------------------------------------------
    # KEEP FIRST ROW PER TICKER
    # --------------------------------------------------------

    result = (
        result
        .drop_duplicates(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )

    log(
        f"TOP20 TICKERS : "
        f"{len(result)}"
    )

    return result


# ============================================================
# STRUCTURE
# ============================================================

def load_structure():

    log(
        "Loading STEP 8 structure"
    )

    df = pd.read_csv(
        STRUCTURE_FILE
    )

    if df.empty:

        raise ValueError(
            "structure.csv is empty"
        )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol"
        ]
    )

    if ticker_col is None:

        raise ValueError(
            "Ticker column missing "
            "in structure.csv"
        )

    result = pd.DataFrame()

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = {

        "current_price": [
            "current_price",
            "price",
            "spot"
        ],

        "call_wall": [
            "call_wall"
        ],

        "put_wall": [
            "put_wall"
        ],

        "support": [
            "support"
        ],

        "resistance": [
            "resistance"
        ],

        "call_gex": [
            "call_gex"
        ],

        "put_gex": [
            "put_gex"
        ],

        "net_gex": [
            "net_gex",
            "gex"
        ],
    }

    for target, candidates in numeric_columns.items():

        column = find_column(
            df,
            candidates
        )

        result[target] = numeric(
            df,
            column
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_columns = {

        "structure": [
            "structure",
            "structure_type"
        ],

        "price_location": [
            "price_location"
        ],

        "gex_structure": [
            "gex_structure"
        ],

        "wall_structure": [
            "wall_structure"
        ],

        "gex_source": [
            "gex_source"
        ],
    }

    for target, candidates in text_columns.items():

        column = find_column(
            df,
            candidates
        )

        if column is None:

            result[target] = ""

        else:

            result[target] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    result = (
        result
        .drop_duplicates(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )

    log(
        f"STRUCTURE ROWS : "
        f"{len(result)}"
    )

    return result


# ============================================================
# MARKET COMPONENT
# ============================================================

def calculate_market_component(
    market_score
):

    market_score = max(
        0.0,
        min(
            100.0,
            float(market_score)
        )
    )

    return (
        market_score
        / 100.0
        * MARKET_WEIGHT
    )


# ============================================================
# FLOW COMPONENT
# ============================================================

def calculate_flow_component(
    flow_score,
    top20_score
):

    values = []

    if not pd.isna(flow_score):

        values.append(
            float(flow_score)
        )

    if not pd.isna(top20_score):

        values.append(
            float(top20_score)
        )

    if not values:

        score = 50.0

    else:

        score = float(
            np.mean(values)
        )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return (
        score
        / 100.0
        * FLOW_WEIGHT,
        score
    )


# ============================================================
# FLOW DIRECTION COMPONENT
#
# CALL DOMINANT itself is NOT automatically bullish.
#
# We combine:
#   flow_direction
#   estimated_direction
#   directional_ratio
#   call_put_imbalance
# ============================================================

def calculate_direction_component(
    flow_direction,
    estimated_direction,
    directional_ratio,
    call_put_imbalance
):

    score = 50.0

    reasons = []

    flow_direction = clean_text(
        flow_direction
    )

    estimated_direction = clean_text(
        estimated_direction
    )

    # --------------------------------------------------------
    # ESTIMATED DIRECTION
    # --------------------------------------------------------

    if (
        "BUY" in estimated_direction
        and "SELL" not in estimated_direction
    ):

        score += 25

        reasons.append(
            "Buy-side estimate"
        )

    elif (
        "SELL" in estimated_direction
        and "BUY" not in estimated_direction
    ):

        score -= 25

        reasons.append(
            "Sell-side estimate"
        )

    # --------------------------------------------------------
    # FLOW DIRECTION
    # --------------------------------------------------------

    if "CALL DOMINANT" in flow_direction:

        score += 10

        reasons.append(
            "Call dominant flow"
        )

    elif "PUT DOMINANT" in flow_direction:

        score -= 10

        reasons.append(
            "Put dominant flow"
        )

    elif "BALANCED" in flow_direction:

        reasons.append(
            "Balanced flow"
        )

    # --------------------------------------------------------
    # DIRECTIONAL RATIO
    # --------------------------------------------------------

    if not pd.isna(directional_ratio):

        ratio = float(
            directional_ratio
        )

        ratio = max(
            -1.0,
            min(
                1.0,
                ratio
            )
        )

        score += (
            ratio
            * 15.0
        )

        if ratio >= 0.30:

            reasons.append(
                "Positive directional ratio"
            )

        elif ratio <= -0.30:

            reasons.append(
                "Negative directional ratio"
            )

    # --------------------------------------------------------
    # CALL / PUT IMBALANCE
    # --------------------------------------------------------

    if not pd.isna(call_put_imbalance):

        imbalance = float(
            call_put_imbalance
        )

        imbalance = max(
            -1.0,
            min(
                1.0,
                imbalance
            )
        )

        score += (
            imbalance
            * 10.0
        )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    component = (
        score
        / 100.0
        * DIRECTION_WEIGHT
    )

    if not reasons:

        reasons.append(
            "Flow direction unavailable"
        )

    return (
        component,
        score,
        " | ".join(reasons)
    )


# ============================================================
# STRUCTURE COMPONENT
#
# STEP 8 실제 컬럼:
#
# structure
# price_location
# gex_structure
# wall_structure
#
# GEX가 0이면 GEX 방향성을 억지로 만들지 않는다.
# ============================================================

def calculate_structure_component(
    structure,
    price_location,
    gex_structure,
    wall_structure,
    call_gex,
    put_gex,
    net_gex
):

    score = 50.0

    reasons = []

    structure_text = clean_text(
        structure
    )

    price_text = clean_text(
        price_location
    )

    gex_text = clean_text(
        gex_structure
    )

    wall_text = clean_text(
        wall_structure
    )

    # --------------------------------------------------------
    # STRUCTURE DIRECTION
    # --------------------------------------------------------

    if "BULLISH" in structure_text:

        score += 25

        reasons.append(
            "Bullish structure"
        )

    elif "BEARISH" in structure_text:

        score -= 25

        reasons.append(
            "Bearish structure"
        )

    else:

        reasons.append(
            "Neutral structure"
        )

    # --------------------------------------------------------
    # PRICE LOCATION
    # --------------------------------------------------------

    if (
        "ABOVE SUPPORT" in price_text
        and "BELOW RESISTANCE" in price_text
    ):

        score += 10

        reasons.append(
            "Inside support/resistance range"
        )

    elif "ABOVE SUPPORT" in price_text:

        score += 5

        reasons.append(
            "Above support"
        )

    elif "BELOW SUPPORT" in price_text:

        score -= 15

        reasons.append(
            "Below support"
        )

    if "ABOVE RESISTANCE" in price_text:

        score -= 10

        reasons.append(
            "Above resistance"
        )

    # --------------------------------------------------------
    # WALL STRUCTURE
    # --------------------------------------------------------

    if "BULL" in wall_text:

        score += 10

        reasons.append(
            "Bullish wall structure"
        )

    elif "BEAR" in wall_text:

        score -= 10

        reasons.append(
            "Bearish wall structure"
        )

    elif "RANGE" in wall_text:

        reasons.append(
            "Range wall structure"
        )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------
    #
    # If actual GEX exists, use it.
    # If all GEX = 0, do NOT manufacture direction.
    # --------------------------------------------------------

    gex_available = False

    call_value = 0.0

    put_value = 0.0

    net_value = 0.0

    if not pd.isna(call_gex):

        call_value = float(
            call_gex
        )

    if not pd.isna(put_gex):

        put_value = float(
            put_gex
        )

    if not pd.isna(net_gex):

        net_value = float(
            net_gex
        )

    if (
        abs(call_value)
        + abs(put_value)
        > 0
    ):

        gex_available = True

    if gex_available:

        total = (
            abs(call_value)
            + abs(put_value)
        )

        ratio = (
            net_value
            / total
        )

        ratio = max(
            -1.0,
            min(
                1.0,
                ratio
            )
        )

        score += (
            ratio
            * 20.0
        )

        if ratio >= 0.30:

            reasons.append(
                "Positive GEX"
            )

        elif ratio <= -0.30:

            reasons.append(
                "Negative GEX"
            )

        else:

            reasons.append(
                "Neutral GEX"
            )

    else:

        reasons.append(
            "GEX unavailable"
        )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    component = (
        score
        / 100.0
        * STRUCTURE_WEIGHT
    )

    return (
        component,
        score,
        " | ".join(reasons)
    )


# ============================================================
# PRICE / WALL COMPONENT
# ============================================================

def calculate_price_component(
    current_price,
    support,
    resistance,
    call_wall,
    put_wall
):

    if pd.isna(current_price):

        return (
            50.0
            / 100.0
            * PRICE_WEIGHT,
            50.0,
            "Price unavailable"
        )

    current_price = float(
        current_price
    )

    score_parts = []

    reasons = []

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    if (
        not pd.isna(support)
        and float(support) > 0
    ):

        support = float(
            support
        )

        distance = (
            current_price - support
        ) / support

        if distance >= 0.05:

            score_parts.append(100)

            reasons.append(
                "Comfortably above support"
            )

        elif distance >= 0.02:

            score_parts.append(85)

            reasons.append(
                "Above support"
            )

        elif distance >= 0:

            score_parts.append(65)

            reasons.append(
                "Near support"
            )

        else:

            score_parts.append(15)

            reasons.append(
                "Below support"
            )

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    if (
        not pd.isna(resistance)
        and float(resistance) > 0
    ):

        resistance = float(
            resistance
        )

        distance = (
            resistance - current_price
        ) / resistance

        if distance >= 0.05:

            score_parts.append(100)

            reasons.append(
                "Room below resistance"
            )

        elif distance >= 0.02:

            score_parts.append(85)

            reasons.append(
                "Moderate room below resistance"
            )

        elif distance >= 0:

            score_parts.append(55)

            reasons.append(
                "Near resistance"
            )

        else:

            score_parts.append(20)

            reasons.append(
                "Above resistance"
            )

    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    if (
        not pd.isna(put_wall)
        and float(put_wall) > 0
    ):

        put_wall = float(
            put_wall
        )

        if current_price > put_wall:

            distance = (
                current_price - put_wall
            ) / put_wall

            if distance >= 0.03:

                score_parts.append(100)

                reasons.append(
                    "Above put wall"
                )

            else:

                score_parts.append(75)

                reasons.append(
                    "Near put wall"
                )

        else:

            score_parts.append(15)

            reasons.append(
                "Below put wall"
            )

    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    if (
        not pd.isna(call_wall)
        and float(call_wall) > 0
    ):

        call_wall = float(
            call_wall
        )

        if current_price < call_wall:

            distance = (
                call_wall - current_price
            ) / call_wall

            if distance >= 0.03:

                score_parts.append(100)

                reasons.append(
                    "Below call wall with room"
                )

            else:

                score_parts.append(70)

                reasons.append(
                    "Near call wall"
                )

        else:

            score_parts.append(30)

            reasons.append(
                "Above call wall"
            )

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if not score_parts:

        return (
            50.0
            / 100.0
            * PRICE_WEIGHT,
            50.0,
            "Price structure unavailable"
        )

    score = (
        sum(score_parts)
        / len(score_parts)
    )

    component = (
        score
        / 100.0
        * PRICE_WEIGHT
    )

    return (
        component,
        score,
        " | ".join(reasons)
    )


# ============================================================
# INDEX ALIGNMENT
# ============================================================

def calculate_index_component(
    directions
):

    values = [

        directions.get(
            "ndx_direction",
            "UNAVAILABLE"
        ),

        directions.get(
            "spy_direction",
            "UNAVAILABLE"
        ),

        directions.get(
            "soxx_direction",
            "UNAVAILABLE"
        ),

        directions.get(
            "dia_direction",
            "UNAVAILABLE"
        ),
    ]

    bullish = sum(
        value == "BULLISH"
        for value in values
    )

    bearish = sum(
        value == "BEARISH"
        for value in values
    )

    neutral = sum(
        value == "NEUTRAL"
        for value in values
    )

    available = (
        bullish
        + bearish
        + neutral
    )

    if available == 0:

        score = 50.0

        reason = (
            "Index direction unavailable"
        )

    else:

        score = (
            50.0
            + bullish * 12.5
            - bearish * 12.5
        )

        score = max(
            0.0,
            min(
                100.0,
                score
            )
        )

        reason = (
            f"Index alignment "
            f"BULL {bullish} / "
            f"BEAR {bearish} / "
            f"NEUTRAL {neutral}"
        )

    component = (
        score
        / 100.0
        * INDEX_WEIGHT
    )

    return (
        component,
        score,
        reason
    )


# ============================================================
# HARD RISK FILTER
# ============================================================

def apply_hard_risk_filters(
    total,
    structure_row,
    direction_score
):

    hard_risk = False

    reasons = []

    structure_text = clean_text(
        structure_row["structure"]
    )

    current_price = (
        structure_row["current_price"]
    )

    support = (
        structure_row["support"]
    )

    resistance = (
        structure_row["resistance"]
    )

    # --------------------------------------------------------
    # BEARISH STRUCTURE
    # --------------------------------------------------------

    if "BEARISH" in structure_text:

        if total >= ENTER_THRESHOLD:

            total = min(
                total,
                ENTER_THRESHOLD - 0.1
            )

            hard_risk = True

            reasons.append(
                "Bearish structure blocks entry"
            )

    # --------------------------------------------------------
    # BELOW SUPPORT
    # --------------------------------------------------------

    if (
        not pd.isna(current_price)
        and not pd.isna(support)
        and float(current_price)
        < float(support)
    ):

        total = min(
            total,
            WATCH_THRESHOLD - 0.1
        )

        hard_risk = True

        reasons.append(
            "Price below calculated support"
        )

    # --------------------------------------------------------
    # STRONG NEGATIVE FLOW DIRECTION
    # --------------------------------------------------------

    if (
        not pd.isna(direction_score)
        and direction_score <= 25
    ):

        if total >= ENTER_THRESHOLD:

            total = min(
                total,
                ENTER_THRESHOLD - 0.1
            )

            hard_risk = True

            reasons.append(
                "Strong negative flow direction blocks entry"
            )

    # --------------------------------------------------------
    # ABOVE RESISTANCE
    # --------------------------------------------------------

    if (
        not pd.isna(current_price)
        and not pd.isna(resistance)
        and float(current_price)
        > float(resistance)
    ):

        if total >= ENTER_THRESHOLD:

            total = min(
                total,
                ENTER_THRESHOLD - 0.1
            )

            hard_risk = True

            reasons.append(
                "Price above calculated resistance"
            )

    return (
        total,
        hard_risk,
        reasons
    )


# ============================================================
# FINAL DECISION
# ============================================================

def calculate_decision(
    market,
    flow_score,
    top20_row,
    structure_row
):

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market_component = (
        calculate_market_component(
            market["market_score"]
        )
    )

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    flow_component, effective_flow_score = (
        calculate_flow_component(

            flow_score,

            top20_row[
                "top20_score"
            ]
        )
    )

    # --------------------------------------------------------
    # FLOW DIRECTION
    # --------------------------------------------------------

    (
        direction_component,
        direction_score,
        direction_reason
    ) = calculate_direction_component(

        top20_row[
            "flow_direction"
        ],

        top20_row[
            "estimated_direction"
        ],

        top20_row[
            "directional_ratio"
        ],

        top20_row[
            "call_put_imbalance"
        ]
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    (
        structure_component,
        structure_score,
        structure_reason
    ) = calculate_structure_component(

        structure_row[
            "structure"
        ],

        structure_row[
            "price_location"
        ],

        structure_row[
            "gex_structure"
        ],

        structure_row[
            "wall_structure"
        ],

        structure_row[
            "call_gex"
        ],

        structure_row[
            "put_gex"
        ],

        structure_row[
            "net_gex"
        ]
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    (
        price_component,
        price_score,
        price_reason
    ) = calculate_price_component(

        structure_row[
            "current_price"
        ],

        structure_row[
            "support"
        ],

        structure_row[
            "resistance"
        ],

        structure_row[
            "call_wall"
        ],

        structure_row[
            "put_wall"
        ]
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    (
        index_component,
        index_score,
        index_reason
    ) = calculate_index_component(
        market
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total = (

        market_component

        + flow_component

        + direction_component

        + structure_component

        + price_component

        + index_component
    )

    total = max(
        0.0,
        min(
            100.0,
            total
        )
    )

    # --------------------------------------------------------
    # HARD RISK
    # --------------------------------------------------------

    (
        total,
        hard_risk,
        hard_reasons
    ) = apply_hard_risk_filters(

        total,

        structure_row,

        direction_score
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if total >= ENTER_THRESHOLD:

        decision = "🟢 진입"

    elif total >= WATCH_THRESHOLD:

        decision = "🟡 관망"

    else:

        decision = "🔴 회피"

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    reasons = [

        f"Market {market['market_score']:.1f}",

        f"Flow {effective_flow_score:.1f}",

        f"Direction {direction_score:.1f}",

        direction_reason,

        structure_reason,

        price_reason,

        index_reason
    ]

    if hard_risk:

        reasons.extend(
            hard_reasons
        )

    return {

        "market_component":
            market_component,

        "flow_component":
            flow_component,

        "direction_component":
            direction_component,

        "structure_component":
            structure_component,

        "price_component":
            price_component,

        "index_component":
            index_component,

        "decision_score":
            total,

        "decision":
            decision,

        "effective_flow_score":
            effective_flow_score,

        "direction_score":
            direction_score,

        "structure_score":
            structure_score,

        "price_score":
            price_score,

        "index_score":
            index_score,

        "reason":
            " | ".join(reasons)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "START"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    market = load_market_regime()

    flow = load_flow()

    top20 = load_top20()

    structure = load_structure()

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    flow_lookup = (
        flow
        .drop_duplicates(
            "ticker"
        )
        .set_index(
            "ticker"
        )
    )

    structure_lookup = (
        structure
        .drop_duplicates(
            "ticker"
        )
        .set_index(
            "ticker"
        )
    )

    # --------------------------------------------------------
    # TOP20 ORDER
    # --------------------------------------------------------

    top20_tickers = (
        top20[
            "ticker"
        ]
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # TOP20 COUNT
    # --------------------------------------------------------

    if len(top20_tickers) != 20:

        raise ValueError(
            "TOP20 must contain exactly 20 unique tickers. "
            f"Found {len(top20_tickers)}"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    rows = []

    for rank, ticker in enumerate(
        top20_tickers,
        start=1
    ):

        # ----------------------------------------------------
        # STRUCTURE
        # ----------------------------------------------------

        if ticker not in structure_lookup.index:

            raise ValueError(
                f"STEP 8 structure missing for {ticker}"
            )

        structure_row = (
            structure_lookup
            .loc[ticker]
        )

        # ----------------------------------------------------
        # TOP20
        # ----------------------------------------------------

        top20_row = (
            top20[
                top20["ticker"]
                == ticker
            ]
            .iloc[0]
        )

        # ----------------------------------------------------
        # FLOW
        # ----------------------------------------------------

        if ticker in flow_lookup.index:

            flow_score = (
                flow_lookup
                .loc[ticker]
                .get(
                    "flow_score",
                    np.nan
                )
            )

        else:

            flow_score = np.nan

        # ----------------------------------------------------
        # CALCULATE
        # ----------------------------------------------------

        result = calculate_decision(

            market,

            flow_score,

            top20_row,

            structure_row
        )

        # ----------------------------------------------------
        # OUTPUT ROW
        # ----------------------------------------------------

        rows.append({

            "rank":
                rank,

            "ticker":
                ticker,

            # ------------------------------------------------
            # MARKET
            # ------------------------------------------------

            "market_score":
                market["market_score"],

            "market_regime":
                market["market_regime"],

            "ndx_direction":
                market["ndx_direction"],

            "spy_direction":
                market["spy_direction"],

            "soxx_direction":
                market["soxx_direction"],

            "dia_direction":
                market["dia_direction"],

            # ------------------------------------------------
            # TOP20
            # ------------------------------------------------

            "top20_score":
                top20_row[
                    "top20_score"
                ],

            "max_flow_score":
                top20_row[
                    "max_flow_score"
                ],

            "avg_flow_score":
                top20_row[
                    "avg_flow_score"
                ],

            "total_volume":
                top20_row[
                    "total_volume"
                ],

            "total_premium":
                top20_row[
                    "total_premium"
                ],

            "call_premium":
                top20_row[
                    "call_premium"
                ],

            "put_premium":
                top20_row[
                    "put_premium"
                ],

            "call_put_imbalance":
                top20_row[
                    "call_put_imbalance"
                ],

            "directional_ratio":
                top20_row[
                    "directional_ratio"
                ],

            "flow_direction":
                top20_row[
                    "flow_direction"
                ],

            "estimated_direction":
                top20_row[
                    "estimated_direction"
                ],

            "top_dte":
                top20_row[
                    "top_dte"
                ],

            # ------------------------------------------------
            # FLOW
            # ------------------------------------------------

            "flow_score":
                flow_score,

            "effective_flow_score":
                result[
                    "effective_flow_score"
                ],

            # ------------------------------------------------
            # STRUCTURE
            # ------------------------------------------------

            "current_price":
                structure_row[
                    "current_price"
                ],

            "call_wall":
                structure_row[
                    "call_wall"
                ],

            "put_wall":
                structure_row[
                    "put_wall"
                ],

            "support":
                structure_row[
                    "support"
                ],

            "resistance":
                structure_row[
                    "resistance"
                ],

            "call_gex":
                structure_row[
                    "call_gex"
                ],

            "put_gex":
                structure_row[
                    "put_gex"
                ],

            "net_gex":
                structure_row[
                    "net_gex"
                ],

            "gex_source":
                structure_row[
                    "gex_source"
                ],

            "structure":
                structure_row[
                    "structure"
                ],

            "price_location":
                structure_row[
                    "price_location"
                ],

            "gex_structure":
                structure_row[
                    "gex_structure"
                ],

            "wall_structure":
                structure_row[
                    "wall_structure"
                ],

            # ------------------------------------------------
            # COMPONENTS
            # ------------------------------------------------

            "market_component":
                result[
                    "market_component"
                ],

            "flow_component":
                result[
                    "flow_component"
                ],

            "direction_component":
                result[
                    "direction_component"
                ],

            "structure_component":
                result[
                    "structure_component"
                ],

            "price_component":
                result[
                    "price_component"
                ],

            "index_component":
                result[
                    "index_component"
                ],

            # ------------------------------------------------
            # RAW SCORES
            # ------------------------------------------------

            "direction_score":
                result[
                    "direction_score"
                ],

            "structure_score":
                result[
                    "structure_score"
                ],

            "price_score":
                result[
                    "price_score"
                ],

            "index_score":
                result[
                    "index_score"
                ],

            # ------------------------------------------------
            # FINAL
            # ------------------------------------------------

            "decision_score":
                result[
                    "decision_score"
                ],

            "decision":
                result[
                    "decision"
                ],

            "reason":
                result[
                    "reason"
                ],

            "data_source":
                "CALCULATED"
        })

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        log(
            f"{ticker} | "
            f"FLOW {result['effective_flow_score']:.1f} | "
            f"DIRECTION {result['direction_score']:.1f} | "
            f"STRUCTURE {result['structure_score']:.1f} | "
            f"PRICE {result['price_score']:.1f} | "
            f"FINAL {result['decision_score']:.2f} | "
            f"{result['decision']}"
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    output = pd.DataFrame(
        rows
    )

    if output.empty:

        raise ValueError(
            "Decision output is empty"
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    output = (
        output
        .sort_values(
            [
                "decision_score",
                "effective_flow_score"
            ],
            ascending=[
                False,
                False
            ]
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

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 78)
    print("🔎 STEP 9 VALIDATION")
    print("=" * 78)

    print(
        f"MARKET SCORE       : "
        f"{market['market_score']:.2f}"
    )

    print(
        f"MARKET REGIME      : "
        f"{market['market_regime']}"
    )

    print(
        f"TOP20 INPUT        : "
        f"{len(top20_tickers)}"
    )

    print(
        f"DECISION ROWS      : "
        f"{len(output)}"
    )

    print(
        f"UNIQUE TICKERS     : "
        f"{output['ticker'].nunique()}"
    )

    print(
        f"VALID SCORES       : "
        f"{output['decision_score'].notna().sum()}"
    )

    print(
        f"VALID DECISIONS    : "
        f"{output['decision'].notna().sum()}"
    )

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("SCORE DISTRIBUTION")
    print("-" * 78)

    print(
        f"MAX SCORE          : "
        f"{output['decision_score'].max():.2f}"
    )

    print(
        f"MIN SCORE          : "
        f"{output['decision_score'].min():.2f}"
    )

    print(
        f"MEAN SCORE         : "
        f"{output['decision_score'].mean():.2f}"
    )

    print(
        f"UNIQUE SCORES      : "
        f"{output['decision_score'].nunique()}"
    )

    # --------------------------------------------------------
    # COMPONENT CHECK
    # --------------------------------------------------------

    print()
    print("COMPONENT CHECK")
    print("-" * 78)

    for column in [

        "market_component",

        "flow_component",

        "direction_component",

        "structure_component",

        "price_component",

        "index_component"
    ]:

        print(
            f"{column:<24} "
            f"MIN={output[column].min():.2f} "
            f"MAX={output[column].max():.2f} "
            f"UNIQUE={output[column].nunique()}"
        )

    # --------------------------------------------------------
    # DECISION SUMMARY
    # --------------------------------------------------------

    print()
    print("DECISION SUMMARY")
    print("-" * 50)

    for decision in [

        "🟢 진입",

        "🟡 관망",

        "🔴 회피"
    ]:

        count = (
            output["decision"]
            == decision
        ).sum()

        print(
            f"{decision:<10} : {count}"
        )

    # --------------------------------------------------------
    # TOP DECISIONS
    # --------------------------------------------------------

    print()
    print("TOP DECISIONS")
    print("-" * 100)

    preview_columns = [

        "final_rank",

        "ticker",

        "top20_score",

        "flow_score",

        "flow_direction",

        "estimated_direction",

        "direction_score",

        "structure_score",

        "price_score",

        "decision_score",

        "decision"
    ]

    print(
        output[
            preview_columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "OUTPUT FILE : "
        "data/analysis/decision.csv"
    )

    print("=" * 78)

    log(
        "STEP 9 DECISION COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

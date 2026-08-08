import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# STEP 9 - FINAL DECISION ENGINE
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
# ============================================================

MARKET_WEIGHT = 25.0
FLOW_WEIGHT = 30.0
STRUCTURE_WEIGHT = 20.0
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
# FILE CHECK
# ============================================================

def check_files():

    print()
    print("=" * 78)
    print("STEP 9 REQUIRED FILE CHECK")
    print("=" * 78)

    files = {
        "MARKET REGIME": MARKET_FILE,
        "UNUSUAL FLOW": FLOW_FILE,
        "TOP20": TOP20_FILE,
        "STRUCTURE": STRUCTURE_FILE,
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

            directions[key] = (
                str(value)
                .strip()
                .upper()
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
        "market_score": market_score,
        "market_regime": regime,
        **directions
    }


# ============================================================
# FLOW
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

        for col in df.columns:

            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
            )

            sample = values.head(20)

            if sample.empty:
                continue

            valid = sample.str.match(
                r"^[A-Z]{1,6}$"
            ).sum()

            if valid >= 2:

                ticker_col = col
                break

    if ticker_col is None:

        raise ValueError(
            "Unable to identify "
            "TOP20 ticker column"
        )

    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    log(
        f"TOP20 TICKERS : "
        f"{len(tickers)}"
    )

    return tickers


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

    structure_col = find_column(
        df,
        [
            "structure",
            "structure_type"
        ]
    )

    if structure_col is not None:

        result["structure"] = (
            df[structure_col]
            .fillna("UNAVAILABLE")
            .astype(str)
            .str.strip()
        )

    else:

        result["structure"] = (
            "UNAVAILABLE"
        )

    gex_source_col = find_column(
        df,
        [
            "gex_source"
        ]
    )

    if gex_source_col is not None:

        result["gex_source"] = (
            df[gex_source_col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

    else:

        result["gex_source"] = "UNKNOWN"

    log(
        f"STRUCTURE ROWS : "
        f"{len(result)}"
    )

    return result


# ============================================================
# MARKET SCORE COMPONENT
# ============================================================

def calculate_market_component(
    market_score
):

    return (
        market_score
        / 100.0
        * MARKET_WEIGHT
    )


# ============================================================
# FLOW COMPONENT
# ============================================================

def calculate_flow_component(
    flow_score
):

    if pd.isna(flow_score):

        return (
            50.0
            / 100.0
            * FLOW_WEIGHT
        )

    flow_score = max(
        0.0,
        min(
            100.0,
            float(flow_score)
        )
    )

    return (
        flow_score
        / 100.0
        * FLOW_WEIGHT
    )


# ============================================================
# GEX COMPONENT
# ============================================================

def calculate_gex_component(
    call_gex,
    put_gex,
    net_gex
):

    if (
        pd.isna(net_gex)
        and pd.isna(call_gex)
        and pd.isna(put_gex)
    ):

        return (
            50.0
            / 100.0
            * STRUCTURE_WEIGHT
        ), "GEX unavailable"

    call = (
        0.0
        if pd.isna(call_gex)
        else abs(float(call_gex))
    )

    put = (
        0.0
        if pd.isna(put_gex)
        else abs(float(put_gex))
    )

    net = (
        0.0
        if pd.isna(net_gex)
        else float(net_gex)
    )

    total = call + put

    if total <= 0:

        return (
            50.0
            / 100.0
            * STRUCTURE_WEIGHT
        ), "Neutral GEX"

    ratio = net / total

    ratio = max(
        -1.0,
        min(
            1.0,
            ratio
        )
    )

    # -1 = 0 score
    #  0 = 50 score
    # +1 = 100 score

    normalized = (
        ratio + 1.0
    ) / 2.0

    component = (
        normalized
        * STRUCTURE_WEIGHT
    )

    if ratio >= 0.50:

        reason = "Strong positive GEX"

    elif ratio >= 0.15:

        reason = "Positive GEX"

    elif ratio <= -0.50:

        reason = "Strong negative GEX"

    elif ratio <= -0.15:

        reason = "Negative GEX"

    else:

        reason = "Neutral GEX"

    return component, reason


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
            "Price unavailable"
        )

    score_parts = []

    reasons = []

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    if (
        not pd.isna(support)
        and support > 0
    ):

        support_distance = (
            current_price - support
        ) / support

        if support_distance >= 0.05:

            support_score = 100

            reasons.append(
                "Comfortably above support"
            )

        elif support_distance >= 0.02:

            support_score = 80

            reasons.append(
                "Above support"
            )

        elif support_distance >= 0:

            support_score = 60

            reasons.append(
                "Near support"
            )

        else:

            support_score = 20

            reasons.append(
                "Below support"
            )

        score_parts.append(
            support_score
        )

    # --------------------------------------------------------
    # RESISTANCE
    # --------------------------------------------------------

    if (
        not pd.isna(resistance)
        and resistance > 0
    ):

        resistance_distance = (
            resistance - current_price
        ) / resistance

        if resistance_distance >= 0.05:

            resistance_score = 100

            reasons.append(
                "Room below resistance"
            )

        elif resistance_distance >= 0.02:

            resistance_score = 80

            reasons.append(
                "Moderate room below resistance"
            )

        elif resistance_distance >= 0:

            resistance_score = 55

            reasons.append(
                "Near resistance"
            )

        else:

            resistance_score = 25

            reasons.append(
                "Above resistance"
            )

        score_parts.append(
            resistance_score
        )

    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    if (
        not pd.isna(put_wall)
        and put_wall > 0
    ):

        if current_price > put_wall:

            put_distance = (
                current_price - put_wall
            ) / put_wall

            if put_distance >= 0.03:

                score_parts.append(100)

                reasons.append(
                    "Above put wall"
                )

            else:

                score_parts.append(75)

                reasons.append(
                    "Slightly above put wall"
                )

        else:

            score_parts.append(20)

            reasons.append(
                "Below put wall"
            )

    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    if (
        not pd.isna(call_wall)
        and call_wall > 0
    ):

        if current_price < call_wall:

            call_distance = (
                call_wall - current_price
            ) / call_wall

            if call_distance >= 0.03:

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

            score_parts.append(35)

            reasons.append(
                "Above call wall"
            )

    if not score_parts:

        return (
            50.0
            / 100.0
            * PRICE_WEIGHT,
            "Price structure unavailable"
        )

    price_score = (
        sum(score_parts)
        / len(score_parts)
    )

    component = (
        price_score
        / 100.0
        * PRICE_WEIGHT
    )

    return (
        component,
        " | ".join(reasons)
    )


# ============================================================
# STRUCTURE DIRECTION COMPONENT
# ============================================================

def calculate_structure_component(
    structure
):

    text = str(
        structure
    ).upper()

    score = 50.0
    reasons = []

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if "BULLISH" in text:

        score += 30

        reasons.append(
            "Bullish structure"
        )

    elif "BEARISH" in text:

        score -= 30

        reasons.append(
            "Bearish structure"
        )

    else:

        reasons.append(
            "Neutral structure"
        )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    if "POSITIVE GEX" in text:

        score += 10

        reasons.append(
            "Positive GEX structure"
        )

    elif "NEGATIVE GEX" in text:

        score -= 10

        reasons.append(
            "Negative GEX structure"
        )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    if "ABOVE SUPPORT" in text:

        score += 5

    if "BELOW RESISTANCE" in text:

        score += 5

    if "SUPPORT RISK" in text:

        score -= 5

    if "RESISTANCE RISK" in text:

        score -= 5

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

    available = (
        bullish
        + bearish
        + sum(
            value == "NEUTRAL"
            for value in values
        )
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
            f"BEAR {bearish}"
        )

    component = (
        score
        / 100.0
        * INDEX_WEIGHT
    )

    return (
        component,
        reason
    )


# ============================================================
# FINAL DECISION
# ============================================================

def calculate_decision(
    market,
    flow_score,
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

    flow_component = (
        calculate_flow_component(
            flow_score
        )
    )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    gex_component, gex_reason = (
        calculate_gex_component(
            structure_row["call_gex"],
            structure_row["put_gex"],
            structure_row["net_gex"]
        )
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    structure_component, structure_reason = (
        calculate_structure_component(
            structure_row["structure"]
        )
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price_component, price_reason = (
        calculate_price_component(
            structure_row["current_price"],
            structure_row["support"],
            structure_row["resistance"],
            structure_row["call_wall"],
            structure_row["put_wall"]
        )
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index_component, index_reason = (
        calculate_index_component(
            market
        )
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    total = (
        market_component
        + flow_component
        + gex_component
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
    # HARD RISK FILTERS
    # --------------------------------------------------------

    structure_text = str(
        structure_row["structure"]
    ).upper()

    hard_risk = False
    hard_reason = ""

    if "BEARISH" in structure_text:

        if total >= ENTER_THRESHOLD:

            total = min(
                total,
                ENTER_THRESHOLD - 0.1
            )

            hard_reason = (
                "Bearish structure blocks entry"
            )

            hard_risk = True

    if (
        not pd.isna(
            structure_row["current_price"]
        )
        and not pd.isna(
            structure_row["support"]
        )
        and structure_row["current_price"]
        < structure_row["support"]
    ):

        total = min(
            total,
            WATCH_THRESHOLD - 0.1
        )

        hard_reason = (
            "Price below calculated support"
        )

        hard_risk = True

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if total >= ENTER_THRESHOLD:

        decision = "🟢 진입"

    elif total >= WATCH_THRESHOLD:

        decision = "🟡 관망"

    else:

        decision = "🔴 회피"

    reasons = [
        f"Market {market['market_score']:.1f}",
        f"Flow {flow_score:.1f}"
        if not pd.isna(flow_score)
        else "Flow unavailable",
        gex_reason,
        structure_reason,
        price_reason,
        index_reason
    ]

    if hard_risk:

        reasons.append(
            hard_reason
        )

    return {
        "market_component": market_component,
        "flow_component": flow_component,
        "gex_component": gex_component,
        "structure_component": structure_component,
        "price_component": price_component,
        "index_component": index_component,
        "decision_score": total,
        "decision": decision,
        "reason": " | ".join(reasons)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # LOAD INPUTS
    # --------------------------------------------------------

    market = load_market_regime()

    flow = load_flow()

    top20 = load_top20()

    structure = load_structure()

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    rows = []

    for rank, ticker in enumerate(
        top20,
        start=1
    ):

        structure_rows = structure[
            structure["ticker"] == ticker
        ]

        if structure_rows.empty:

            log(
                f"{ticker} | "
                "STRUCTURE MISSING"
            )

            continue

        flow_rows = flow[
            flow["ticker"] == ticker
        ]

        structure_row = (
            structure_rows.iloc[0]
        )

        if flow_rows.empty:

            flow_score = np.nan

        else:

            flow_score = (
                flow_rows.iloc[0][
                    "flow_score"
                ]
            )

        result = calculate_decision(
            market,
            flow_score,
            structure_row
        )

        rows.append({

            "rank":
                rank,

            "ticker":
                ticker,

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

            "flow_score":
                flow_score,

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

            "market_component":
                result[
                    "market_component"
                ],

            "flow_component":
                result[
                    "flow_component"
                ],

            "gex_component":
                result[
                    "gex_component"
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

        log(
            f"{ticker} | "
            f"FLOW {flow_score:.1f}"
            if not pd.isna(flow_score)
            else f"{ticker} | FLOW N/A"
        )

        log(
            f"{ticker} | "
            f"FINAL SCORE "
            f"{result['decision_score']:.2f} | "
            f"{result['decision']}"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = pd.DataFrame(
        rows
    )

    if output.empty:

        raise ValueError(
            "Decision output is empty"
        )

    output = (
        output
        .sort_values(
            [
                "decision_score",
                "flow_score"
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
        f"{len(top20)}"
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

    print()
    print("TOP DECISIONS")
    print("-" * 78)

    preview_columns = [
        "final_rank",
        "ticker",
        "market_score",
        "flow_score",
        "market_component",
        "flow_component",
        "gex_component",
        "structure_component",
        "price_component",
        "index_component",
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

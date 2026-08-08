import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
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

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv"
)

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

SEARCH_FILE = os.path.join(
    ANALYSIS_DIR,
    "option_search.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

MARKET_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
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


def normalize_ticker(df):

    column = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol"
        ]
    )

    if column is None:

        raise ValueError(
            "Ticker column not found"
        )

    result = df.copy()

    result["_ticker"] = (
        result[column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return result


# ============================================================
# MARKET
# ============================================================

def load_market_regime():

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
            "score",
            "market_regime_score"
        ]
    )

    regime_col = find_column(
        df,
        [
            "market_regime",
            "regime"
        ]
    )

    if score_col is None:

        raise ValueError(
            "market_score missing"
        )

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce"
    ).dropna()

    if scores.empty:

        raise ValueError(
            "No valid market score"
        )

    score = float(
        scores.iloc[-1]
    )

    if not 0 <= score <= 100:

        raise ValueError(
            f"Invalid market score: {score}"
        )

    regime = "UNKNOWN"

    if regime_col is not None:

        values = (
            df[regime_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if not values.empty:

            regime = values.iloc[-1]

    log(
        f"MARKET SCORE : {score:.2f}"
    )

    log(
        f"MARKET REGIME : {regime}"
    )

    return score, regime


# ============================================================
# FLOW
# ============================================================

def prepare_flow(df):

    df = normalize_ticker(df)

    score_col = find_column(
        df,
        [
            "flow_score",
            "option_flow_score",
            "score"
        ]
    )

    result = pd.DataFrame()

    result["_ticker"] = df["_ticker"]

    result["flow_score"] = numeric(
        df,
        score_col
    )

    return (
        result
        .groupby(
            "_ticker",
            as_index=False
        )
        .agg({
            "flow_score": "max"
        })
    )


# ============================================================
# OPTION SEARCH
# ============================================================

def prepare_search(df):

    df = normalize_ticker(df)

    score_col = find_column(
        df,
        [
            "option_search_score",
            "search_score",
            "score"
        ]
    )

    signal_col = find_column(
        df,
        [
            "signal",
            "option_signal",
            "search_signal"
        ]
    )

    result = pd.DataFrame()

    result["_ticker"] = df["_ticker"]

    result["option_search_score"] = numeric(
        df,
        score_col
    )

    if signal_col is not None:

        result["option_signal"] = (
            df[signal_col]
            .astype(str)
            .str.strip()
        )

    else:

        result["option_signal"] = ""

    return (
        result
        .groupby(
            "_ticker",
            as_index=False
        )
        .agg({
            "option_search_score": "max",
            "option_signal": "first"
        })
    )


# ============================================================
# STRUCTURE
# ============================================================

def prepare_structure(df):

    df = normalize_ticker(df)

    result = pd.DataFrame()

    result["_ticker"] = df["_ticker"]

    for field in [
        "current_price",
        "call_wall",
        "put_wall",
        "support",
        "resistance",
        "net_gex"
    ]:

        column = find_column(
            df,
            [field]
        )

        result[field] = numeric(
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
            .astype(str)
            .str.strip()
        )

    else:

        result["structure"] = (
            "UNAVAILABLE"
        )

    return result


# ============================================================
# TOP20
# ============================================================

def load_top20():

    df = normalize_ticker(
        pd.read_csv(
            TOP20_FILE
        )
    )

    return (
        df["_ticker"]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# DECISION ENGINE
# ============================================================

def calculate_decision(
    market_score,
    flow_score,
    search_score,
    net_gex,
    structure,
    special_score=0
):

    score = 0.0
    reasons = []

    # ========================================================
    # MARKET : 20
    # ========================================================

    if market_score >= 70:

        score += 20

        reasons.append(
            "Strong bullish market regime"
        )

    elif market_score >= 55:

        score += 15

        reasons.append(
            "Positive market regime"
        )

    elif market_score >= 45:

        score += 10

        reasons.append(
            "Neutral market regime"
        )

    elif market_score >= 30:

        score += 5

        reasons.append(
            "Weak market regime"
        )

    else:

        reasons.append(
            "Risk-off market regime"
        )

    # ========================================================
    # FLOW : 25
    # ========================================================

    if not pd.isna(flow_score):

        score += (
            max(
                0,
                min(
                    100,
                    flow_score
                )
            ) * 0.25
        )

        if flow_score >= 80:

            reasons.append(
                "Very strong unusual option flow"
            )

        elif flow_score >= 60:

            reasons.append(
                "Strong unusual option flow"
            )

        elif flow_score >= 40:

            reasons.append(
                "Moderate unusual option flow"
            )

        else:

            reasons.append(
                "Weak unusual option flow"
            )

    else:

        reasons.append(
            "Flow score unavailable"
        )

    # ========================================================
    # OPTION SEARCH : 25
    # ========================================================

    if not pd.isna(search_score):

        score += (
            max(
                0,
                min(
                    100,
                    search_score
                )
            ) * 0.25
        )

        if search_score >= 80:

            reasons.append(
                "Strong actionable option setup"
            )

        elif search_score >= 60:

            reasons.append(
                "Good option setup"
            )

        elif search_score < 40:

            reasons.append(
                "Weak option setup"
            )

    else:

        reasons.append(
            "Option search unavailable"
        )

    # ========================================================
    # STRUCTURE / GEX : 20
    # ========================================================

    structure_text = str(
        structure
    ).upper()

    structure_points = 0

    if "BULLISH" in structure_text:

        structure_points += 12

        reasons.append(
            "Bullish option structure"
        )

    elif "BEARISH" in structure_text:

        structure_points -= 8

        reasons.append(
            "Bearish option structure"
        )

    elif "NEUTRAL" in structure_text:

        structure_points += 5

        reasons.append(
            "Neutral option structure"
        )

    if not pd.isna(net_gex):

        if net_gex > 0:

            structure_points += 8

            reasons.append(
                "Positive GEX"
            )

        elif net_gex < 0:

            structure_points -= 4

            reasons.append(
                "Negative GEX"
            )

    score += max(
        0,
        min(
            20,
            structure_points
        )
    )

    # ========================================================
    # SPECIAL STRUCTURE
    # ========================================================

    if special_score > 0:

        score += special_score

        reasons.append(
            "SPECIAL OPTION STRUCTURE detected"
        )

    # ========================================================
    # BOUNDS
    # ========================================================

    score = max(
        0,
        min(
            100,
            score
        )
    )

    # ========================================================
    # MARKET HARD FILTER
    # ========================================================

    if market_score < 30:

        decision = "🔴 회피"

        reasons.append(
            "Market regime below entry threshold"
        )

        return (
            score,
            decision,
            reasons
        )

    # ========================================================
    # FINAL
    # ========================================================

    if score >= 70:

        decision = "🟢 진입"

    elif score >= 50:

        decision = "🟡 관망"

    else:

        decision = "🔴 회피"

    return (
        score,
        decision,
        reasons
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    required = [
        STRUCTURE_FILE,
        FLOW_FILE,
        SEARCH_FILE,
        TOP20_FILE,
        MARKET_FILE
    ]

    for path in required:

        if not os.path.exists(path):

            raise FileNotFoundError(
                path
            )

    market_score, market_regime = (
        load_market_regime()
    )

    structure = prepare_structure(
        pd.read_csv(
            STRUCTURE_FILE
        )
    )

    flow = prepare_flow(
        pd.read_csv(
            FLOW_FILE
        )
    )

    search = prepare_search(
        pd.read_csv(
            SEARCH_FILE
        )
    )

    top20 = load_top20()

    flow_lookup = (
        flow
        .set_index("_ticker")
    )

    search_lookup = (
        search
        .set_index("_ticker")
    )

    structure_lookup = (
        structure
        .set_index("_ticker")
    )

    rows = []

    for rank, ticker in enumerate(
        top20,
        start=1
    ):

        if ticker not in structure_lookup.index:

            log(
                f"{ticker} | STRUCTURE MISSING"
            )

            continue

        s = structure_lookup.loc[
            ticker
        ]

        if ticker in flow_lookup.index:

            flow_score = (
                flow_lookup.loc[
                    ticker,
                    "flow_score"
                ]
            )

        else:

            flow_score = np.nan

        if ticker in search_lookup.index:

            search_score = (
                search_lookup.loc[
                    ticker,
                    "option_search_score"
                ]
            )

            option_signal = (
                search_lookup.loc[
                    ticker,
                    "option_signal"
                ]
            )

        else:

            search_score = np.nan
            option_signal = ""

        score, decision, reasons = (
            calculate_decision(
                market_score,
                flow_score,
                search_score,
                s["net_gex"],
                s["structure"],
                0
            )
        )

        rows.append({

            "rank": rank,

            "ticker": ticker,

            "market_score":
                market_score,

            "market_regime":
                market_regime,

            "flow_score":
                flow_score,

            "option_search_score":
                search_score,

            "option_signal":
                option_signal,

            "current_price":
                s["current_price"],

            "call_wall":
                s["call_wall"],

            "put_wall":
                s["put_wall"],

            "support":
                s["support"],

            "resistance":
                s["resistance"],

            "net_gex":
                s["net_gex"],

            "structure":
                s["structure"],

            "special_structure":
                "",

            "special_score":
                0,

            "decision_score":
                score,

            "decision":
                decision,

            "reason":
                " | ".join(reasons),

            "data_source":
                "CALCULATED"

        })

        log(
            f"{ticker} | "
            f"SCORE {score:.1f} | "
            f"{decision}"
        )

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
            "decision_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    output["final_rank"] = (
        output.index + 1
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 72)
    print("🔎 STEP 9 VALIDATION")
    print("=" * 72)

    print(
        f"MARKET SCORE       : {market_score:.2f}"
    )

    print(
        f"MARKET REGIME      : {market_regime}"
    )

    print(
        f"TOP20 INPUT        : {len(top20)}"
    )

    print(
        f"DECISION ROWS      : {len(output)}"
    )

    print(
        f"SCORES VALID       : "
        f"{output['decision_score'].notna().sum()}"
    )

    print()

    print(
        output[
            [
                "final_rank",
                "ticker",
                "market_score",
                "flow_score",
                "option_search_score",
                "decision_score",
                "decision"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "OUTPUT FILE : "
        "data/analysis/decision.csv"
    )

    print("=" * 72)

    log(
        "STEP 9 DECISION COMPLETE"
    )


if __name__ == "__main__":
    main()

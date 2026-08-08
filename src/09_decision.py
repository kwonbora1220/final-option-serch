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
# COLUMN HELPERS
# ============================================================

def find_column(
    df,
    candidates
):

    normalized = {
        str(col)
        .strip()
        .lower()
        .replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def numeric(
    df,
    column
):

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
# MARKET REGIME
# ============================================================

def load_market_regime():

    if not os.path.exists(
        MARKET_FILE
    ):

        raise FileNotFoundError(
            "STEP 1 market_regime.csv "
            "not found: "
            f"{MARKET_FILE}"
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
            "market_score column missing "
            "in market_regime.csv"
        )

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce"
    ).dropna()

    if scores.empty:

        raise ValueError(
            "No valid market score found"
        )

    market_score = float(
        scores.iloc[-1]
    )

    # --------------------------------------------------------
    # STEP 1 score MUST be 0~100
    # --------------------------------------------------------

    if (
        market_score < 0
        or market_score > 100
    ):

        raise ValueError(
            "Invalid STEP 1 market score. "
            "Expected 0~100, got "
            f"{market_score}"
        )

    market_regime = "UNKNOWN"

    if regime_col is not None:

        regimes = (
            df[regime_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        if not regimes.empty:

            market_regime = (
                regimes.iloc[-1]
            )

    log(
        f"STEP 1 MARKET SCORE : "
        f"{market_score:.2f}"
    )

    log(
        f"STEP 1 MARKET REGIME : "
        f"{market_regime}"
    )

    return {
        "score": market_score,
        "regime": market_regime
    }


# ============================================================
# FLOW SCORE
# ============================================================

def prepare_flow(df):

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying"
        ]
    )

    score_col = find_column(
        df,
        [
            "flow_score",
            "score",
            "option_flow_score"
        ]
    )

    premium_col = find_column(
        df,
        [
            "premium",
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

    if ticker_col is None:

        raise ValueError(
            "Ticker column missing "
            "in unusual_flow.csv"
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
        score_col
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


# ============================================================
# TOP20
# ============================================================

def load_top20():

    df = pd.read_csv(
        TOP20_FILE
    )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "stock"
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

    return tickers


# ============================================================
# STRUCTURE
# ============================================================

def prepare_structure(df):

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol"
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

    columns = {

        "current_price": [
            "current_price"
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

        "net_gex": [
            "net_gex"
        ]
    }

    for target, candidates in (
        columns.items()
    ):

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
            "structure"
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
# DECISION ENGINE
# ============================================================

def calculate_decision(
    market_score,
    flow_score,
    net_gex,
    structure
):

    score = 50.0

    reasons = []

    # ========================================================
    # MARKET
    #
    # STEP 1 SCORE = 0~100
    # ========================================================

    if market_score >= 70:

        score += 15

        reasons.append(
            "Strong bullish market regime"
        )

    elif market_score >= 55:

        score += 8

        reasons.append(
            "Positive market regime"
        )

    elif market_score <= 30:

        score -= 15

        reasons.append(
            "Risk-off market regime"
        )

    elif market_score < 45:

        score -= 8

        reasons.append(
            "Weak market regime"
        )

    else:

        reasons.append(
            "Neutral market regime"
        )

    # ========================================================
    # FLOW
    # ========================================================

    if not pd.isna(flow_score):

        if flow_score >= 80:

            score += 15

            reasons.append(
                "Very strong unusual "
                "option flow"
            )

        elif flow_score >= 60:

            score += 10

            reasons.append(
                "Strong unusual "
                "option flow"
            )

        elif flow_score >= 40:

            score += 5

            reasons.append(
                "Moderate unusual "
                "option flow"
            )

        elif flow_score < 20:

            score -= 8

            reasons.append(
                "Weak unusual option flow"
            )

    else:

        reasons.append(
            "Flow score unavailable"
        )

    # ========================================================
    # GEX
    # ========================================================

    if not pd.isna(net_gex):

        if net_gex > 0:

            reasons.append(
                "Positive calculated GEX"
            )

        elif net_gex < 0:

            score -= 3

            reasons.append(
                "Negative calculated GEX"
            )

    else:

        reasons.append(
            "GEX unavailable"
        )

    # ========================================================
    # STRUCTURE
    # ========================================================

    text = str(
        structure
    ).upper()

    if "BULLISH" in text:

        score += 8

        reasons.append(
            "Bullish option structure"
        )

    elif "BEARISH" in text:

        score -= 8

        reasons.append(
            "Bearish option structure"
        )

    elif "STABILIZED" in text:

        reasons.append(
            "Positive GEX "
            "stabilization structure"
        )

    elif "HIGHER VOLATILITY" in text:

        score -= 3

        reasons.append(
            "Higher volatility structure"
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
    # FINAL DECISION
    #
    # THIS IS THE ONLY PLACE WHERE
    # FINAL INVESTMENT DECISION IS CREATED.
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

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    required_files = [
        STRUCTURE_FILE,
        FLOW_FILE,
        TOP20_FILE,
        MARKET_FILE
    ]

    for file in required_files:

        if not os.path.exists(file):

            raise FileNotFoundError(
                file
            )

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market = load_market_regime()

    market_score = market[
        "score"
    ]

    market_regime = market[
        "regime"
    ]

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    log(
        "Loading structure"
    )

    structure_raw = pd.read_csv(
        STRUCTURE_FILE
    )

    structure = prepare_structure(
        structure_raw
    )

    log(
        f"STRUCTURE ROWS : "
        f"{len(structure)}"
    )

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    log(
        "Loading unusual flow"
    )

    flow_raw = pd.read_csv(
        FLOW_FILE
    )

    flow = prepare_flow(
        flow_raw
    )

    log(
        f"FLOW TICKERS : "
        f"{len(flow)}"
    )

    # --------------------------------------------------------
    # TOP20
    # --------------------------------------------------------

    log(
        "Loading TOP20"
    )

    top20 = load_top20()

    log(
        f"TOP20 TICKERS : "
        f"{len(top20)}"
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    rows = []

    for rank, ticker in enumerate(
        top20,
        start=1
    ):

        structure_row = (
            structure[
                structure["ticker"]
                == ticker
            ]
        )

        if structure_row.empty:

            log(
                f"{ticker} | "
                "STRUCTURE MISSING"
            )

            continue

        flow_row = (
            flow[
                flow["ticker"]
                == ticker
            ]
        )

        s = structure_row.iloc[0]

        if flow_row.empty:

            flow_score = np.nan

        else:

            flow_score = (
                flow_row.iloc[0][
                    "flow_score"
                ]
            )

        score, decision, reasons = (
            calculate_decision(
                market_score,
                flow_score,
                s["net_gex"],
                s["structure"]
            )
        )

        reason_text = (
            " | ".join(reasons)
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

            "decision_score":
                score,

            "decision":
                decision,

            "reason":
                reason_text,

            "data_source":
                "CALCULATED / ESTIMATED"
        })

        log(
            f"{ticker} | "
            f"MARKET {market_score:.1f} | "
            f"SCORE {score:.1f} | "
            f"{decision}"
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

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()

    print("=" * 72)

    print(
        "🔎 STEP 9 VALIDATION"
    )

    print("=" * 72)

    print(
        f"STEP 1 MARKET SCORE : "
        f"{market_score:.2f}"
    )

    print(
        f"STEP 1 MARKET REGIME: "
        f"{market_regime}"
    )

    print(
        f"TOP20 INPUT         : "
        f"{len(top20)}"
    )

    print(
        f"DECISION ROWS       : "
        f"{len(output)}"
    )

    print(
        f"DECISION TICKERS    : "
        f"{output['ticker'].nunique()}"
    )

    print(
        "SCORES VALID        : "
        f"{output['decision_score'].notna().sum()}"
    )

    print(
        "DECISIONS VALID     : "
        f"{output['decision'].notna().sum()}"
    )

    print()

    print(
        "DECISION SUMMARY"
    )

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
            f"{decision} : {count}"
        )

    print()

    print(
        "TOP DECISIONS"
    )

    print(
        output[
            [
                "final_rank",
                "ticker",
                "market_score",
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

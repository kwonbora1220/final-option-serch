
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

SEARCH_FILE = os.path.join(
    ANALYSIS_DIR,
    "option_search.csv"
)

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv"
)

DECISION_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "final_report.csv"
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
        f"[10 FINAL REPORT] {now} | {message}"
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
# TICKER EXTRACTION
# ============================================================

def extract_ticker_column(df):

    candidates = [
        "ticker",
        "symbol",
        "underlying",
        "underlying_symbol",
        "stock",
        "stock_symbol"
    ]

    column = find_column(
        df,
        candidates
    )

    if column is None:

        raise ValueError(
            "Ticker column not found"
        )

    return column


# ============================================================
# LOAD DATA
# ============================================================

def load_csv(path, name):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(path)

    log(
        f"{name} ROWS : {len(df):,}"
    )

    return df


# ============================================================
# NORMALIZE TICKER
# ============================================================

def normalize_ticker(df):

    ticker_col = extract_ticker_column(
        df
    )

    result = df.copy()

    result["_ticker"] = (
        result[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return result


# ============================================================
# DECISION NORMALIZATION
# ============================================================

def normalize_decision(value):

    if pd.isna(value):

        return "UNKNOWN"

    text = (
        str(value)
        .strip()
        .upper()
    )

    # Korean / English variants

    if (
        "진입" in text
        or "ENTER" in text
        or "BUY" in text
    ):
        return "🟢 진입"

    if (
        "관망" in text
        or "WATCH" in text
        or "HOLD" in text
    ):
        return "🟡 관망"

    if (
        "회피" in text
        or "AVOID" in text
        or "SELL" in text
    ):
        return "🔴 회피"

    return str(value)


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    log("Loading TOP20")

    top20 = normalize_ticker(
        load_csv(
            TOP20_FILE,
            "TOP20"
        )
    )

    log("Loading unusual flow")

    flow = normalize_ticker(
        load_csv(
            FLOW_FILE,
            "UNUSUAL FLOW"
        )
    )

    log("Loading option search")

    search = normalize_ticker(
        load_csv(
            SEARCH_FILE,
            "OPTION SEARCH"
        )
    )

    log("Loading structure")

    structure = normalize_ticker(
        load_csv(
            STRUCTURE_FILE,
            "STRUCTURE"
        )
    )

    log("Loading decision")

    decision = normalize_ticker(
        load_csv(
            DECISION_FILE,
            "DECISION"
        )
    )

    # --------------------------------------------------------
    # TOP TICKERS
    # --------------------------------------------------------

    top_tickers = (
        top20["_ticker"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    log(
        f"TOP TICKERS : {len(top_tickers)}"
    )

    # --------------------------------------------------------
    # TOP20 RANK
    # --------------------------------------------------------

    rank_col = find_column(
        top20,
        [
            "rank",
            "ranking",
            "top_rank",
            "score_rank"
        ]
    )

    top20_score_col = find_column(
        top20,
        [
            "score",
            "flow_score",
            "option_flow_score",
            "total_score"
        ]
    )

    top20_reason_col = find_column(
        top20,
        [
            "selection_reason",
            "reason",
            "reasons",
            "selection_reasons"
        ]
    )

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    flow_score_col = find_column(
        flow,
        [
            "flow_score",
            "option_flow_score",
            "score"
        ]
    )

    premium_col = find_column(
        flow,
        [
            "estimated_premium",
            "premium",
            "total_premium",
            "premium_flow"
        ]
    )

    call_premium_col = find_column(
        flow,
        [
            "call_premium",
            "call_premium_flow"
        ]
    )

    put_premium_col = find_column(
        flow,
        [
            "put_premium",
            "put_premium_flow"
        ]
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_score_col = find_column(
        search,
        [
            "option_search_score",
            "search_score",
            "score"
        ]
    )

    risk_col = find_column(
        search,
        [
            "risk_reversal",
            "risk_reversal_score",
            "risk_reversal_signal"
        ]
    )

    search_signal_col = find_column(
        search,
        [
            "signal",
            "option_signal",
            "search_signal"
        ]
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    price_col = find_column(
        structure,
        [
            "current_price",
            "price",
            "spot"
        ]
    )

    call_wall_col = find_column(
        structure,
        [
            "call_wall"
        ]
    )

    put_wall_col = find_column(
        structure,
        [
            "put_wall"
        ]
    )

    support_col = find_column(
        structure,
        [
            "support"
        ]
    )

    resistance_col = find_column(
        structure,
        [
            "resistance"
        ]
    )

    net_gex_col = find_column(
        structure,
        [
            "net_gex",
            "gex"
        ]
    )

    structure_col = find_column(
        structure,
        [
            "structure",
            "structure_type"
        ]
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    decision_score_col = find_column(
        decision,
        [
            "decision_score",
            "score",
            "final_score"
        ]
    )

    decision_col = find_column(
        decision,
        [
            "decision",
            "final_decision",
            "signal"
        ]
    )

    decision_reason_col = find_column(
        decision,
        [
            "reason",
            "decision_reason",
            "reasons"
        ]
    )

    # --------------------------------------------------------
    # BUILD LOOKUPS
    # --------------------------------------------------------

    top20_lookup = (
        top20
        .drop_duplicates("_ticker")
        .set_index("_ticker")
    )

    flow_lookup = (
        flow
        .drop_duplicates("_ticker")
        .set_index("_ticker")
    )

    search_lookup = (
        search
        .drop_duplicates("_ticker")
        .set_index("_ticker")
    )

    structure_lookup = (
        structure
        .drop_duplicates("_ticker")
        .set_index("_ticker")
    )

    decision_lookup = (
        decision
        .drop_duplicates("_ticker")
        .set_index("_ticker")
    )

    # --------------------------------------------------------
    # FINAL ROWS
    # --------------------------------------------------------

    rows = []

    for position, ticker in enumerate(
        top_tickers,
        start=1
    ):

        row = {
            "rank": position,
            "ticker": ticker
        }

        # ====================================================
        # TOP20
        # ====================================================

        if ticker in top20_lookup.index:

            source = top20_lookup.loc[
                ticker
            ]

            if rank_col is not None:

                row["top20_rank"] = (
                    source[rank_col]
                )

            if top20_score_col is not None:

                row["top20_score"] = (
                    source[top20_score_col]
                )

            if top20_reason_col is not None:

                row["selection_reason"] = (
                    source[top20_reason_col]
                )

        # ====================================================
        # FLOW
        # ====================================================

        if ticker in flow_lookup.index:

            source = flow_lookup.loc[
                ticker
            ]

            if flow_score_col is not None:

                row["flow_score"] = (
                    source[flow_score_col]
                )

            if premium_col is not None:

                row["estimated_premium"] = (
                    source[premium_col]
                )

            if call_premium_col is not None:

                row["call_premium"] = (
                    source[call_premium_col]
                )

            if put_premium_col is not None:

                row["put_premium"] = (
                    source[put_premium_col]
                )

        # ====================================================
        # OPTION SEARCH
        # ====================================================

        if ticker in search_lookup.index:

            source = search_lookup.loc[
                ticker
            ]

            if search_score_col is not None:

                row["option_search_score"] = (
                    source[search_score_col]
                )

            if risk_col is not None:

                row["risk_reversal"] = (
                    source[risk_col]
                )

            if search_signal_col is not None:

                row["option_signal"] = (
                    source[search_signal_col]
                )

        # ====================================================
        # STRUCTURE
        # ====================================================

        if ticker in structure_lookup.index:

            source = structure_lookup.loc[
                ticker
            ]

            if price_col is not None:

                row["current_price"] = (
                    source[price_col]
                )

            if call_wall_col is not None:

                row["call_wall"] = (
                    source[call_wall_col]
                )

            if put_wall_col is not None:

                row["put_wall"] = (
                    source[put_wall_col]
                )

            if support_col is not None:

                row["support"] = (
                    source[support_col]
                )

            if resistance_col is not None:

                row["resistance"] = (
                    source[resistance_col]
                )

            if net_gex_col is not None:

                row["net_gex"] = (
                    source[net_gex_col]
                )

            if structure_col is not None:

                row["structure"] = (
                    source[structure_col]
                )

        # ====================================================
        # DECISION
        # ====================================================

        if ticker in decision_lookup.index:

            source = decision_lookup.loc[
                ticker
            ]

            if decision_score_col is not None:

                row["decision_score"] = (
                    source[decision_score_col]
                )

            if decision_col is not None:

                row["decision"] = (
                    source[decision_col]
                )

            if decision_reason_col is not None:

                row["decision_reason"] = (
                    source[decision_reason_col]
                )

        # ====================================================
        # FINAL DECISION
        # ====================================================

        if "decision" in row:

            row["final_decision"] = (
                normalize_decision(
                    row["decision"]
                )
            )

        elif "decision_score" in row:

            score = pd.to_numeric(
                row["decision_score"],
                errors="coerce"
            )

            if pd.isna(score):

                row["final_decision"] = (
                    "UNKNOWN"
                )

            elif score >= 70:

                row["final_decision"] = (
                    "🟢 진입"
                )

            elif score >= 45:

                row["final_decision"] = (
                    "🟡 관망"
                )

            else:

                row["final_decision"] = (
                    "🔴 회피"
                )

        else:

            row["final_decision"] = (
                "UNKNOWN"
            )

        rows.append(row)

        log(
            f"{ticker} | "
            f"{row.get('final_decision', 'UNKNOWN')}"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = pd.DataFrame(
        rows
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
    print("=" * 72)
    print("🔎 STEP 10 VALIDATION")
    print("=" * 72)

    print(
        f"TOP TICKERS       : "
        f"{len(top_tickers)}"
    )

    print(
        f"OUTPUT ROWS       : "
        f"{len(output)}"
    )

    print(
        f"OUTPUT TICKERS    : "
        f"{output['ticker'].nunique()}"
    )

    print(
        "FINAL DECISION    : "
        f"{output['final_decision'].notna().sum()}"
    )

    print()
    print("FINAL DECISION COUNTS")
    print("----------------------------------------")

    print(
        output[
            "final_decision"
        ].value_counts()
    )

    print()
    print(
        "OUTPUT FILE       : "
        "data/analysis/final_report.csv"
    )

    print("=" * 72)

    log(
        "STEP 10 FINAL REPORT COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


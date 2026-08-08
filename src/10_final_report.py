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


def load(path, name):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(path)

    log(
        f"{name} ROWS : {len(df):,}"
    )

    return df


def normalize(df):

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
            "Ticker column not found"
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
# MAIN
# ============================================================

def main():

    log("START")

    top20 = normalize(
        load(
            TOP20_FILE,
            "TOP20"
        )
    )

    flow = normalize(
        load(
            FLOW_FILE,
            "UNUSUAL FLOW"
        )
    )

    search = normalize(
        load(
            SEARCH_FILE,
            "OPTION SEARCH"
        )
    )

    structure = normalize(
        load(
            STRUCTURE_FILE,
            "STRUCTURE"
        )
    )

    decision = normalize(
        load(
            DECISION_FILE,
            "DECISION"
        )
    )

    # --------------------------------------------------------
    # TOP20
    # --------------------------------------------------------

    top_tickers = (
        top20["_ticker"]
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # LOOKUPS
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
    # OUTPUT
    # --------------------------------------------------------

    rows = []

    for position, ticker in enumerate(
        top_tickers,
        start=1
    ):

        row = {

            "rank":
                position,

            "ticker":
                ticker

        }

        # ----------------------------------------------------
        # TOP20
        # ----------------------------------------------------

        if ticker in top20_lookup.index:

            source = top20_lookup.loc[
                ticker
            ]

            for field in [
                "rank",
                "score",
                "flow_score",
                "selection_reason",
                "reason"
            ]:

                if field in source.index:

                    row[
                        f"top20_{field}"
                    ] = source[field]

        # ----------------------------------------------------
        # FLOW
        # ----------------------------------------------------

        if ticker in flow_lookup.index:

            source = flow_lookup.loc[
                ticker
            ]

            for field in [
                "flow_score",
                "premium",
                "estimated_premium",
                "call_premium",
                "put_premium"
            ]:

                if field in source.index:

                    row[field] = source[field]

        # ----------------------------------------------------
        # OPTION SEARCH
        # ----------------------------------------------------

        if ticker in search_lookup.index:

            source = search_lookup.loc[
                ticker
            ]

            for field in [
                "option_search_score",
                "search_score",
                "signal",
                "option_signal",
                "risk_reversal",
                "strike",
                "expiration",
                "option_type",
                "premium"
            ]:

                if field in source.index:

                    row[field] = source[field]

        # ----------------------------------------------------
        # STRUCTURE
        # ----------------------------------------------------

        if ticker in structure_lookup.index:

            source = structure_lookup.loc[
                ticker
            ]

            for field in [
                "current_price",
                "call_wall",
                "put_wall",
                "support",
                "resistance",
                "call_gex",
                "put_gex",
                "net_gex",
                "structure"
            ]:

                if field in source.index:

                    row[field] = source[field]

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        if ticker in decision_lookup.index:

            source = decision_lookup.loc[
                ticker
            ]

            for field in [
                "market_score",
                "market_regime",
                "flow_score",
                "option_search_score",
                "option_signal",
                "special_structure",
                "special_score",
                "decision_score",
                "decision",
                "reason"
            ]:

                if field in source.index:

                    row[field] = source[field]

        # ----------------------------------------------------
        # FINAL DECISION
        #
        # STEP 9의 결과를 그대로 사용
        # ----------------------------------------------------

        row["final_decision"] = row.get(
            "decision",
            "UNKNOWN"
        )

        rows.append(row)

        log(
            f"{ticker} | "
            f"{row['final_decision']}"
        )

    output = pd.DataFrame(
        rows
    )

    if output.empty:

        raise ValueError(
            "Final report is empty"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 72)
    print("🔎 STEP 10 VALIDATION")
    print("=" * 72)

    print(
        f"TOP TICKERS       : {len(top_tickers)}"
    )

    print(
        f"OUTPUT ROWS       : {len(output)}"
    )

    print(
        f"OUTPUT TICKERS    : "
        f"{output['ticker'].nunique()}"
    )

    print(
        f"FINAL DECISIONS   : "
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
        "OUTPUT FILE : "
        "data/analysis/final_report.csv"
    )

    print("=" * 72)

    log(
        "STEP 10 FINAL REPORT COMPLETE"
    )


if __name__ == "__main__":
    main()

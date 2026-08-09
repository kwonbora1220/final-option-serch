from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd


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

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv",
)

DECISION_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv",
)

SEARCH_FILE = os.path.join(
    ANALYSIS_DIR,
    "option_search.csv",
)

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "final_report.csv",
)


def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[10 FINAL REPORT] {now} | {message}"
    )


def ticker_column(df):

    for column in [
        "ticker",
        "symbol",
        "underlying",
        "underlying_symbol",
    ]:

        if column in df.columns:
            return column

    raise RuntimeError(
        "Ticker column not found"
    )


def normalize(df):

    column = ticker_column(df)

    df = df.copy()

    df["ticker"] = (
        df[column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def main():

    log("START")

    top20 = normalize(
        pd.read_csv(TOP20_FILE)
    )

    decision = normalize(
        pd.read_csv(DECISION_FILE)
    )

    search = normalize(
        pd.read_csv(SEARCH_FILE)
    )

    structure = normalize(
        pd.read_csv(STRUCTURE_FILE)
    )

    top20 = (
        top20
        .drop_duplicates("ticker")
        .head(20)
        .reset_index(drop=True)
    )

    decision = (
        decision
        .drop_duplicates("ticker")
    )

    search = (
        search
        .drop_duplicates("ticker")
    )

    structure = (
        structure
        .drop_duplicates("ticker")
    )

    if len(top20) != 20:
        raise RuntimeError(
            "TOP20 does not contain exactly 20 tickers"
        )

    rows = []

    for rank, ticker in enumerate(
        top20["ticker"],
        start=1,
    ):

        row = {
            "rank": rank,
            "ticker": ticker,
        }

        d = decision[
            decision["ticker"] == ticker
        ]

        s = search[
            search["ticker"] == ticker
        ]

        st = structure[
            structure["ticker"] == ticker
        ]

        if d.empty:
            raise RuntimeError(
                f"Missing decision: {ticker}"
            )

        if st.empty:
            raise RuntimeError(
                f"Missing structure: {ticker}"
            )

        decision_row = d.iloc[0]
        structure_row = st.iloc[0]

        # -----------------------------------------------------
        # DECISION
        # -----------------------------------------------------

        for column in [
            "market_score",
            "market_regime",
            "ndx_direction",
            "spy_direction",
            "soxx_direction",
            "dia_direction",
            "flow_score",
            "direction_score",
            "structure_score",
            "price_score",
            "index_score",
            "decision_score",
            "decision",
            "reason",
        ]:

            if column in decision_row.index:
                row[column] = decision_row[column]

        # -----------------------------------------------------
        # STRUCTURE
        # -----------------------------------------------------

        for column in [
            "current_price",
            "call_wall",
            "put_wall",
            "support",
            "resistance",
            "call_gex",
            "put_gex",
            "net_gex",
            "structure",
            "price_location",
            "gex_structure",
            "wall_structure",
        ]:

            if column in structure_row.index:
                row[column] = structure_row[column]

        # -----------------------------------------------------
        # OPTION SEARCH
        # -----------------------------------------------------

        if not s.empty:

            search_row = s.iloc[0]

            for column in [
                "risk_reversal",
                "rr_score",
                "rr_call_strike",
                "rr_call_dte",
                "rr_call_premium",
                "rr_put_strike",
                "rr_put_dte",
                "rr_put_premium",
            ]:

                if column in search_row.index:
                    row[column] = search_row[column]

        # -----------------------------------------------------
        # FINAL DECISION
        # -----------------------------------------------------

        decision_value = (
            str(
                row.get(
                    "decision",
                    "",
                )
            )
            .strip()
        )

        if decision_value not in {
            "🟢 진입",
            "🟡 관망",
            "🔴 회피",
        }:

            raise RuntimeError(
                f"Invalid decision for "
                f"{ticker}: {decision_value}"
            )

        row["final_decision"] = (
            decision_value
        )

        rows.append(row)

    output = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    output = output.sort_values(
        "decision_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    output["rank"] = (
        range(
            1,
            len(output) + 1,
        )
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if len(output) != 20:
        raise RuntimeError(
            "FINAL REPORT must contain 20 rows"
        )

    if output["ticker"].nunique() != 20:
        raise RuntimeError(
            "FINAL REPORT contains duplicate tickers"
        )

    if output[
        "decision_score"
    ].isna().any():
        raise RuntimeError(
            "FINAL REPORT contains NaN decision score"
        )

    if (
        (
            output["decision_score"]
            < 0
        )
        |
        (
            output["decision_score"]
            > 100
        )
    ).any():

        raise RuntimeError(
            "FINAL REPORT score outside 0-100"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("STEP 10 FINAL REPORT")
    print("=" * 72)

    print(
        output[
            [
                "rank",
                "ticker",
                "decision_score",
                "final_decision",
            ]
        ].to_string(index=False)
    )

    print()
    print("DECISION SUMMARY")
    print(
        output[
            "final_decision"
        ].value_counts()
    )

    print()
    print(
        "ROWS              :",
        len(output),
    )

    print(
        "TICKERS           :",
        output["ticker"].nunique(),
    )

    print(
        "DECISION SCORE    :",
        output["decision_score"].notna().sum(),
    )

    print(
        "DECISION VALID    :",
        output["final_decision"].notna().sum(),
    )

    print()
    print(
        "FINAL REPORT : OK"
    )

    log(
        "STEP 10 FINAL REPORT COMPLETE"
    )


if __name__ == "__main__":
    main()

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ANALYSIS_DIR = "data/analysis"

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

DECISION_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
)

OUTPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "special_list.csv"
)

OUTPUT_MD = os.path.join(
    ANALYSIS_DIR,
    "special_list.md"
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
        f"[11 SPECIAL LIST] {now} | {message}"
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


def clean_text(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# OPTION TYPE
# ============================================================

def normalize_option_type(value):

    value = str(value).upper().strip()

    mapping = {
        "C": "CALL",
        "CALL": "CALL",
        "CALLS": "CALL",
        "P": "PUT",
        "PUT": "PUT",
        "PUTS": "PUT",
    }

    return mapping.get(
        value,
        value
    )


# ============================================================
# BUY / SELL DETECTION
# ============================================================

def is_buy(row):

    trade_side = str(
        row["trade_side"]
    ).upper().strip()

    open_close = str(
        row["open_close"]
    ).upper().strip()

    call_put_flow = str(
        row["call_put_flow"]
    ).upper().strip()

    # --------------------------------------------------------
    # Explicit BUY
    # --------------------------------------------------------

    buy_tokens = {
        "BUY",
        "BTO",
        "BOT",
        "BUY TO OPEN",
        "BUY_TO_OPEN",
        "BUY OPEN",
        "BTO OPEN",
    }

    if trade_side in buy_tokens:
        return True

    if open_close in {
        "BTO",
        "BUY TO OPEN",
        "BUY_TO_OPEN",
    }:
        return True

    if "BUY" in call_put_flow:
        return True

    if "BTO" in call_put_flow:
        return True

    return False


def is_sell(row):

    trade_side = str(
        row["trade_side"]
    ).upper().strip()

    open_close = str(
        row["open_close"]
    ).upper().strip()

    call_put_flow = str(
        row["call_put_flow"]
    ).upper().strip()

    # --------------------------------------------------------
    # Explicit SELL
    # --------------------------------------------------------

    sell_tokens = {
        "SELL",
        "STO",
        "SOLD",
        "SELL TO OPEN",
        "SELL_TO_OPEN",
        "SELL OPEN",
        "STO OPEN",
    }

    if trade_side in sell_tokens:
        return True

    if open_close in {
        "STO",
        "SELL TO OPEN",
        "SELL_TO_OPEN",
    }:
        return True

    if "SELL" in call_put_flow:
        return True

    if "STO" in call_put_flow:
        return True

    return False


# ============================================================
# STRUCTURE FILTER
# ============================================================

def classify_structure(
    ticker_flow
):

    call_rows = ticker_flow[
        ticker_flow["option_type"] == "CALL"
    ].copy()

    put_rows = ticker_flow[
        ticker_flow["option_type"] == "PUT"
    ].copy()

    if call_rows.empty:
        return None

    if put_rows.empty:
        return None

    call_buy = call_rows[
        call_rows.apply(
            is_buy,
            axis=1
        )
    ].copy()

    put_sell = put_rows[
        put_rows.apply(
            is_sell,
            axis=1
        )
    ].copy()

    if call_buy.empty:
        return None

    if put_sell.empty:
        return None

    # --------------------------------------------------------
    # Aggregate evidence
    # --------------------------------------------------------

    call_volume = call_buy["volume"].sum()

    put_volume = put_sell["volume"].sum()

    call_premium = (
        call_buy["premium"]
        .fillna(0)
        .sum()
    )

    put_premium = (
        put_sell["premium"]
        .fillna(0)
        .sum()
    )

    call_count = len(call_buy)
    put_count = len(put_sell)

    return {
        "call_buy_count": int(call_count),
        "put_sell_count": int(put_count),
        "call_buy_volume": float(call_volume),
        "put_sell_volume": float(put_volume),
        "call_buy_premium": float(call_premium),
        "put_sell_premium": float(put_premium),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    if not os.path.exists(FLOW_FILE):

        raise FileNotFoundError(
            f"Flow file not found: {FLOW_FILE}"
        )

    if not os.path.exists(DECISION_FILE):

        raise FileNotFoundError(
            f"Decision file not found: {DECISION_FILE}"
        )

    log(
        f"FLOW FILE : {FLOW_FILE}"
    )

    log(
        f"DECISION FILE : {DECISION_FILE}"
    )

    flow = pd.read_csv(
        FLOW_FILE
    )

    decision = pd.read_csv(
        DECISION_FILE
    )

    log(
        f"FLOW ROWS : {len(flow):,}"
    )

    log(
        f"DECISION ROWS : {len(decision)}"
    )

    # --------------------------------------------------------
    # COLUMN DETECTION
    # --------------------------------------------------------

    ticker_col = find_column(
        flow,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol"
        ]
    )

    option_type_col = find_column(
        flow,
        [
            "option_type",
            "type",
            "call_put",
            "contract_type"
        ]
    )

    trade_side_col = find_column(
        flow,
        [
            "trade_side_estimate",
            "trade_side",
            "side",
            "estimated_trade_side"
        ]
    )

    open_close_col = find_column(
        flow,
        [
            "open_close_estimate",
            "open_close",
            "openclose"
        ]
    )

    call_put_flow_col = find_column(
        flow,
        [
            "call_put_flow",
            "flow_type",
            "option_flow"
        ]
    )

    volume_col = find_column(
        flow,
        [
            "volume",
            "option_volume"
        ]
    )

    premium_col = find_column(
        flow,
        [
            "estimated_traded_premium",
            "traded_premium",
            "premium",
            "option_premium"
        ]
    )

    if ticker_col is None:
        raise ValueError(
            "Ticker column not found"
        )

    if option_type_col is None:
        raise ValueError(
            "Option type column not found"
        )

    if (
        trade_side_col is None
        and open_close_col is None
        and call_put_flow_col is None
    ):
        raise ValueError(
            "No trade direction columns found"
        )

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    data = pd.DataFrame()

    data["ticker"] = clean_text(
        flow[ticker_col]
    )

    data["option_type"] = (
        flow[option_type_col]
        .apply(normalize_option_type)
    )

    if trade_side_col is not None:

        data["trade_side"] = clean_text(
            flow[trade_side_col]
        )

    else:

        data["trade_side"] = ""

    if open_close_col is not None:

        data["open_close"] = clean_text(
            flow[open_close_col]
        )

    else:

        data["open_close"] = ""

    if call_put_flow_col is not None:

        data["call_put_flow"] = clean_text(
            flow[call_put_flow_col]
        )

    else:

        data["call_put_flow"] = ""

    if volume_col is not None:

        data["volume"] = numeric(
            flow[volume_col]
        ).fillna(0)

    else:

        data["volume"] = 0.0

    if premium_col is not None:

        data["premium"] = numeric(
            flow[premium_col]
        ).fillna(0)

    else:

        data["premium"] = 0.0

    # --------------------------------------------------------
    # DECISION MAP
    # --------------------------------------------------------

    decision_ticker_col = find_column(
        decision,
        [
            "ticker",
            "symbol",
            "underlying"
        ]
    )

    if decision_ticker_col is None:

        raise ValueError(
            "Decision ticker column not found"
        )

    decision["ticker"] = clean_text(
        decision[decision_ticker_col]
    )

    decision_map = {}

    for _, row in decision.iterrows():

        ticker = row["ticker"]

        decision_map[ticker] = {
            "decision": row.get(
                "decision",
                ""
            ),
            "decision_score": row.get(
                "decision_score",
                np.nan
            ),
            "market_regime": row.get(
                "market_regime",
                ""
            ),
            "flow_score": row.get(
                "flow_score",
                np.nan
            ),
        }

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    rows = []

    tickers = (
        data["ticker"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    log(
        f"FLOW TICKERS : {len(tickers)}"
    )

    for ticker in tickers:

        ticker_data = data[
            data["ticker"] == ticker
        ].copy()

        result = classify_structure(
            ticker_data
        )

        if result is None:
            continue

        info = decision_map.get(
            ticker,
            {}
        )

        rows.append({

            "ticker":
                ticker,

            "call_buy_count":
                result["call_buy_count"],

            "put_sell_count":
                result["put_sell_count"],

            "call_buy_volume":
                result["call_buy_volume"],

            "put_sell_volume":
                result["put_sell_volume"],

            "call_buy_premium":
                result["call_buy_premium"],

            "put_sell_premium":
                result["put_sell_premium"],

            "structure":
                "CALL BUY + PUT SELL",

            "market_regime":
                info.get(
                    "market_regime",
                    ""
                ),

            "flow_score":
                info.get(
                    "flow_score",
                    np.nan
                ),

            "decision_score":
                info.get(
                    "decision_score",
                    np.nan
                ),

            "decision":
                info.get(
                    "decision",
                    ""
                ),

            "data_source":
                "CALCULATED",

        })

    output = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    if not output.empty:

        output["total_premium"] = (
            output["call_buy_premium"]
            + output["put_sell_premium"].abs()
        )

        output = output.sort_values(
            [
                "decision_score",
                "total_premium",
                "call_buy_volume"
            ],
            ascending=False,
            na_position="last"
        ).reset_index(
            drop=True
        )

        output.insert(
            0,
            "special_rank",
            range(
                1,
                len(output) + 1
            )
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SAVE MARKDOWN
    # --------------------------------------------------------

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# STEP 11 - SPECIAL LIST\n\n"
        )

        f.write(
            "## CALL BUY + PUT SELL\n\n"
        )

        if output.empty:

            f.write(
                "No CALL BUY + PUT SELL structures found.\n"
            )

        else:

            f.write(
                f"**SPECIAL LIST : {len(output)} tickers**\n\n"
            )

            f.write(
                "| Rank | Ticker | CALL BUY | PUT SELL | CALL Volume | PUT Volume | Decision |\n"
            )

            f.write(
                "|---:|---|---:|---:|---:|---:|---|\n"
            )

            for _, row in output.iterrows():

                f.write(
                    f"| {int(row['special_rank'])} "
                    f"| {row['ticker']} "
                    f"| {int(row['call_buy_count'])} "
                    f"| {int(row['put_sell_count'])} "
                    f"| {row['call_buy_volume']:,.0f} "
                    f"| {row['put_sell_volume']:,.0f} "
                    f"| {row['decision']} |\n"
                )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    call_buy_valid = (
        output["call_buy_count"] > 0
    ).sum() if not output.empty else 0

    put_sell_valid = (
        output["put_sell_count"] > 0
    ).sum() if not output.empty else 0

    print()
    print("=" * 72)
    print("🔎 STEP 11 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT FLOW ROWS       : {len(flow):,}"
    )

    print(
        f"INPUT DECISION ROWS   : {len(decision)}"
    )

    print(
        f"SPECIAL LIST ROWS     : {len(output)}"
    )

    print(
        f"CALL BUY VALID        : {call_buy_valid}"
    )

    print(
        f"PUT SELL VALID        : {put_sell_valid}"
    )

    print(
        f"CALL BUY + PUT SELL   : {len(output)}"
    )

    print()

    if not output.empty:

        print(
            "## SPECIAL LIST"
        )

        print()

        print(
            output[
                [
                    "special_rank",
                    "ticker",
                    "call_buy_count",
                    "put_sell_count",
                    "call_buy_volume",
                    "put_sell_volume",
                    "decision"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "NO SPECIAL STRUCTURES FOUND"
        )

    print()

    print(
        f"OUTPUT CSV            : {OUTPUT_CSV}"
    )

    print(
        f"OUTPUT MARKDOWN       : {OUTPUT_MD}"
    )

    print("=" * 72)

    log(
        "STEP 11 SPECIAL LIST COMPLETE"
    )


if __name__ == "__main__":
    main()

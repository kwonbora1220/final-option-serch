import os
import re
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

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "special_list.csv"
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
# NORMALIZE TEXT
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value).upper()

    text = (
        text
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("+", " + ")
        .replace(",", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIDE DETECTION
# ============================================================

def detect_side(text):

    text = clean_text(text)

    if (
        "CALL BUY" in text
        or "BUY CALL" in text
        or "CALL BTO" in text
        or "BTO CALL" in text
    ):

        call_buy = True

    else:

        call_buy = False

    if (
        "PUT SELL" in text
        or "SELL PUT" in text
        or "PUT STO" in text
        or "STO PUT" in text
    ):

        put_sell = True

    else:

        put_sell = False

    if (
        "PUT BUY" in text
        or "BUY PUT" in text
        or "PUT BTO" in text
        or "BTO PUT" in text
    ):

        put_buy = True

    else:

        put_buy = False

    if (
        "CALL SELL" in text
        or "SELL CALL" in text
        or "CALL STO" in text
        or "STO CALL" in text
    ):

        call_sell = True

    else:

        call_sell = False

    return (
        call_buy,
        put_sell,
        put_buy,
        call_sell
    )


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def classify_special(text):

    (
        call_buy,
        put_sell,
        put_buy,
        call_sell
    ) = detect_side(text)

    # --------------------------------------------------------
    # PRIORITY 1
    # --------------------------------------------------------

    if call_buy and put_sell:

        return (
            "CALL BUY + PUT SELL",
            100,
            "BULLISH"
        )

    # --------------------------------------------------------
    # PRIORITY 2
    # --------------------------------------------------------

    if put_buy and call_sell:

        return (
            "PUT BUY + CALL SELL",
            100,
            "BEARISH"
        )

    # --------------------------------------------------------
    # CALL BUY + PUT BUY
    # --------------------------------------------------------

    if call_buy and put_buy:

        return (
            "CALL BUY + PUT BUY",
            70,
            "LONG VOLATILITY"
        )

    # --------------------------------------------------------
    # CALL BUY
    # --------------------------------------------------------

    if call_buy:

        return (
            "CALL BUY",
            55,
            "BULLISH"
        )

    # --------------------------------------------------------
    # PUT BUY
    # --------------------------------------------------------

    if put_buy:

        return (
            "PUT BUY",
            55,
            "BEARISH"
        )

    # --------------------------------------------------------
    # CALL SELL + PUT SELL
    # --------------------------------------------------------

    if call_sell and put_sell:

        return (
            "CALL SELL + PUT SELL",
            35,
            "SHORT VOLATILITY"
        )

    # --------------------------------------------------------
    # ROLL
    # --------------------------------------------------------

    if "ROLL" in text:

        return (
            "ROLL",
            10,
            "ROLL"
        )

    # --------------------------------------------------------
    # STRADDLE
    # --------------------------------------------------------

    if "STRADDLE" in text:

        return (
            "STRADDLE",
            10,
            "VOLATILITY"
        )

    # --------------------------------------------------------
    # STRANGLE
    # --------------------------------------------------------

    if "STRANGLE" in text:

        return (
            "STRANGLE",
            10,
            "VOLATILITY"
        )

    return (
        "",
        0,
        ""
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    if not os.path.exists(
        FLOW_FILE
    ):

        raise FileNotFoundError(
            FLOW_FILE
        )

    flow = pd.read_csv(
        FLOW_FILE
    )

    top20 = pd.read_csv(
        TOP20_FILE
    )

    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    ticker_col = find_column(
        flow,
        [
            "ticker",
            "symbol",
            "underlying"
        ]
    )

    if ticker_col is None:

        raise ValueError(
            "Ticker column missing"
        )

    description_col = find_column(
        flow,
        [
            "structure",
            "strategy",
            "flow_type",
            "trade_type",
            "transaction_type",
            "description",
            "signal",
            "option_signal"
        ]
    )

    if description_col is None:

        raise ValueError(
            "No structure / strategy column found "
            "in unusual_flow.csv"
        )

    premium_col = find_column(
        flow,
        [
            "premium",
            "estimated_premium",
            "total_premium",
            "premium_flow"
        ]
    )

    volume_col = find_column(
        flow,
        [
            "volume",
            "option_volume"
        ]
    )

    oi_col = find_column(
        flow,
        [
            "open_interest",
            "oi"
        ]
    )

    expiration_col = find_column(
        flow,
        [
            "expiration",
            "expiry",
            "expiration_date"
        ]
    )

    strike_col = find_column(
        flow,
        [
            "strike",
            "strike_price"
        ]
    )

    option_type_col = find_column(
        flow,
        [
            "option_type",
            "type",
            "call_put"
        ]
    )

    flow_score_col = find_column(
        flow,
        [
            "flow_score",
            "option_flow_score",
            "score"
        ]
    )

    # --------------------------------------------------------
    # TOP20 TICKERS
    # --------------------------------------------------------

    top_ticker_col = find_column(
        top20,
        [
            "ticker",
            "symbol",
            "underlying",
            "stock"
        ]
    )

    top_tickers = set()

    if top_ticker_col is not None:

        top_tickers = set(
            top20[
                top_ticker_col
            ]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .tolist()
        )

    # --------------------------------------------------------
    # BUILD COMBINED TEXT
    # --------------------------------------------------------

    text_columns = []

    for col in [
        description_col,
        option_type_col
    ]:

        if col is not None:

            text_columns.append(
                flow[col]
                .fillna("")
                .astype(str)
            )

    combined = (
        pd.Series(
            "",
            index=flow.index
        )
    )

    for series in text_columns:

        combined = (
            combined
            + " "
            + series
        )

    flow["_combined_text"] = (
        combined.apply(
            clean_text
        )
    )

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    rows = []

    for index, source in flow.iterrows():

        ticker = (
            str(
                source[ticker_col]
            )
            .upper()
            .strip()
        )

        text = source[
            "_combined_text"
        ]

        (
            structure,
            special_score,
            direction
        ) = classify_special(
            text
        )

        if not structure:

            continue

        premium = np.nan

        if premium_col is not None:

            premium = pd.to_numeric(
                source[premium_col],
                errors="coerce"
            )

        volume = np.nan

        if volume_col is not None:

            volume = pd.to_numeric(
                source[volume_col],
                errors="coerce"
            )

        oi = np.nan

        if oi_col is not None:

            oi = pd.to_numeric(
                source[oi_col],
                errors="coerce"
            )

        flow_score = np.nan

        if flow_score_col is not None:

            flow_score = pd.to_numeric(
                source[flow_score_col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Quality bonus
        # ----------------------------------------------------

        quality_score = (
            float(special_score)
        )

        if not pd.isna(flow_score):

            quality_score += (
                min(
                    20,
                    max(
                        0,
                        flow_score * 0.20
                    )
                )
            )

        if (
            ticker in top_tickers
            and ticker != ""
        ):

            quality_score += 10

        quality_score = min(
            130,
            quality_score
        )

        rows.append({

            "ticker":
                ticker,

            "special_structure":
                structure,

            "direction":
                direction,

            "special_score":
                special_score,

            "quality_score":
                quality_score,

            "flow_score":
                flow_score,

            "premium":
                premium,

            "volume":
                volume,

            "open_interest":
                oi,

            "expiration":
                (
                    source[expiration_col]
                    if expiration_col is not None
                    else np.nan
                ),

            "strike":
                (
                    source[strike_col]
                    if strike_col is not None
                    else np.nan
                ),

            "option_type":
                (
                    source[option_type_col]
                    if option_type_col is not None
                    else ""
                ),

            "top20":
                ticker in top_tickers,

            "source_text":
                text,

            "data_source":
                "CALCULATED"

        })

    output = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if output.empty:

        output = pd.DataFrame(
            columns=[
                "ticker",
                "special_structure",
                "direction",
                "special_score",
                "quality_score",
                "flow_score",
                "premium",
                "volume",
                "open_interest",
                "expiration",
                "strike",
                "option_type",
                "top20",
                "source_text",
                "data_source"
            ]
        )

    else:

        output = (
            output
            .sort_values(
                [
                    "special_score",
                    "quality_score",
                    "flow_score"
                ],
                ascending=False
            )
            .reset_index(
                drop=True
            )
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
    # OUTPUT
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
    print("=" * 72)
    print("🔥 STEP 11 SPECIAL LIST VALIDATION")
    print("=" * 72)

    print(
        f"FLOW ROWS         : {len(flow):,}"
    )

    print(
        f"SPECIAL ROWS      : {len(output):,}"
    )

    if not output.empty:

        print()

        print(
            "SPECIAL STRUCTURES"
        )

        print(
            output[
                "special_structure"
            ]
            .value_counts()
            .to_string()
        )

        print()

        print(
            "TOP SPECIAL LIST"
        )

        print(
            output[
                [
                    "special_rank",
                    "ticker",
                    "special_structure",
                    "direction",
                    "quality_score",
                    "flow_score",
                    "top20"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print()

    print(
        "OUTPUT FILE : "
        "data/analysis/special_list.csv"
    )

    print("=" * 72)

    log(
        "STEP 11 SPECIAL LIST COMPLETE"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


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

INPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "prediction_history.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "performance_history.csv"
)


# ============================================================
# HORIZONS
# ============================================================

HORIZONS = {
    "d1": 1,
    "d3": 3,
    "d5": 5,
}


# ============================================================
# LOAD HISTORY
# ============================================================

history = pd.read_csv(
    INPUT_FILE
)

if history.empty:
    raise RuntimeError(
        "prediction_history.csv is empty"
    )


history["signal_date"] = pd.to_datetime(
    history["signal_date"],
    errors="coerce"
).dt.tz_localize(None)


# ============================================================
# TICKERS
# ============================================================

tickers = sorted(
    history["ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .unique()
)


print("==========================================")
print("STEP 14 - PERFORMANCE TRACKER")
print("==========================================")
print()
print(
    "TRACKING TICKERS :",
    len(tickers)
)


# ============================================================
# PRICE CACHE
#
# We need:
# Open / High / Low / Close
# ============================================================

price_cache = {}


for ticker in tickers:

    try:

        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if data is None or data.empty:

            print(
                "NO DATA :",
                ticker
            )

            continue


        # ------------------------------------------------------
        # MultiIndex handling
        # ------------------------------------------------------

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = [
                col[0]
                for col in data.columns
            ]


        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
        ]


        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]


        if missing:

            print(
                "MISSING OHLC :",
                ticker,
                missing
            )

            continue


        data = data[
            required_columns
        ].copy()


        data.index = pd.to_datetime(
            data.index
        )


        # Remove timezone if present
        try:

            data.index = (
                data.index
                .tz_localize(None)
            )

        except TypeError:

            pass


        data = data.sort_index()


        price_cache[ticker] = data


    except Exception as exc:

        print(
            "PRICE ERROR :",
            ticker,
            exc
        )


# ============================================================
# HELPERS
# ============================================================

def get_future_bars(
    data,
    signal_date,
    days
):

    if data is None:
        return pd.DataFrame()

    future = data[
        data.index > signal_date
    ]

    if len(future) < days:
        return pd.DataFrame()

    return future.iloc[
        :days
    ]


def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# UPDATE EACH SIGNAL
# ============================================================

for index, row in history.iterrows():

    ticker = (
        str(row["ticker"])
        .upper()
        .strip()
    )


    signal_date = row[
        "signal_date"
    ]


    if pd.isna(
        signal_date
    ):
        continue


    data = price_cache.get(
        ticker
    )


    if data is None:
        continue


    signal_price = safe_float(
        row["signal_price"]
    )


    if not np.isfinite(
        signal_price
    ):
        continue


    # ========================================================
    # D1 / D3 / D5
    # ========================================================

    for label, days in HORIZONS.items():

        bars = get_future_bars(
            data,
            signal_date,
            days
        )


        if bars.empty:
            continue


        # ----------------------------------------------------
        # END PRICE
        # ----------------------------------------------------

        end_price = safe_float(
            bars["Close"].iloc[-1]
        )


        if np.isfinite(
            end_price
        ):

            history.loc[
                index,
                f"{label}_price"
            ] = end_price


            history.loc[
                index,
                f"{label}_return"
            ] = (
                (
                    end_price
                    /
                    signal_price
                )
                - 1.0
            ) * 100.0


        # ----------------------------------------------------
        # MFE
        #
        # Maximum favorable excursion
        #
        # Highest High after signal
        # relative to signal price
        # ----------------------------------------------------

        highest_price = safe_float(
            bars["High"].max()
        )


        if np.isfinite(
            highest_price
        ):

            mfe = (
                (
                    highest_price
                    /
                    signal_price
                )
                - 1.0
            ) * 100.0


            history.loc[
                index,
                f"{label}_mfe"
            ] = mfe


        # ----------------------------------------------------
        # MAE
        #
        # Maximum adverse excursion
        #
        # Lowest Low after signal
        # relative to signal price
        #
        # Negative number means drawdown.
        # ----------------------------------------------------

        lowest_price = safe_float(
            bars["Low"].min()
        )


        if np.isfinite(
            lowest_price
        ):

            mae = (
                (
                    lowest_price
                    /
                    signal_price
                )
                - 1.0
            ) * 100.0


            history.loc[
                index,
                f"{label}_mae"
            ] = mae


        # ----------------------------------------------------
        # HIT
        #
        # Positive closing return
        # ----------------------------------------------------

        return_value = history.loc[
            index,
            f"{label}_return"
        ]


        if pd.notna(
            return_value
        ):

            history.loc[
                index,
                f"hit_{label}"
            ] = (
                float(return_value) > 0
            )


# ============================================================
# OVERALL MFE / MAE
#
# D5 is our primary evaluation window.
#
# If D5 exists, use it.
# Otherwise keep D1/D3 available.
# ============================================================

if "d5_mfe" in history.columns:

    history["mfe"] = history[
        "d5_mfe"
    ]

else:

    history["mfe"] = np.nan


if "d5_mae" in history.columns:

    history["mae"] = history[
        "d5_mae"
    ]

else:

    history["mae"] = np.nan


# ============================================================
# SAVE
# ============================================================

history.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("PERFORMANCE SUMMARY")
print("==========================================")

print(
    "HISTORY ROWS :",
    len(history)
)

print(
    "D1 COMPLETE  :",
    history["d1_return"]
    .notna()
    .sum()
)

print(
    "D3 COMPLETE  :",
    history["d3_return"]
    .notna()
    .sum()
)

print(
    "D5 COMPLETE  :",
    history["d5_return"]
    .notna()
    .sum()
)


print()

print(
    "D5 AVG RETURN :",
    round(
        pd.to_numeric(
            history["d5_return"],
            errors="coerce"
        ).mean(),
        2
    )
)


print(
    "D5 AVG MFE    :",
    round(
        pd.to_numeric(
            history["d5_mfe"],
            errors="coerce"
        ).mean(),
        2
    )
)


print(
    "D5 AVG MAE    :",
    round(
        pd.to_numeric(
            history["d5_mae"],
            errors="coerce"
        ).mean(),
        2
    )
)


print()
print(
    "OUTPUT :",
    OUTPUT_FILE
)

print()
print("STEP 14 OUTPUT : OK")

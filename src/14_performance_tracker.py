from __future__ import annotations

import os
from datetime import datetime, timezone

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
    "performance_history.csv",
)


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = {
    "d1": 1,
    "d3": 3,
    "d5": 5,
}


# ============================================================
# LOAD
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
    errors="coerce",
)


# ============================================================
# DOWNLOAD PRICE HISTORY
# ============================================================

tickers = sorted(
    history["ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .unique()
)


print(
    "TRACKING TICKERS :",
    len(tickers)
)


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
                ticker,
            )

            continue

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            close = data[
                "Close"
            ].iloc[:, 0]

        else:

            close = data[
                "Close"
            ]

        close.index = (
            pd.to_datetime(
                close.index
            ).tz_localize(
                None
            )
        )

        price_cache[
            ticker
        ] = close

    except Exception as exc:

        print(
            "PRICE ERROR:",
            ticker,
            exc,
        )


# ============================================================
# HELPERS
# ============================================================

def next_trading_price(
    prices,
    signal_date,
    trading_days_after,
):

    if prices is None:
        return np.nan

    future = prices[
        prices.index >
        signal_date
    ]

    if len(future) < trading_days_after:
        return np.nan

    return float(
        future.iloc[
            trading_days_after - 1
        ]
    )


# ============================================================
# UPDATE
# ============================================================

for index, row in history.iterrows():

    ticker = str(
        row["ticker"]
    ).upper().strip()

    signal_date = row[
        "signal_date"
    ]

    if pd.isna(
        signal_date
    ):
        continue

    prices = price_cache.get(
        ticker
    )

    if prices is None:
        continue

    signal_price = row[
        "signal_price"
    ]

    try:
        signal_price = float(
            signal_price
        )
    except Exception:
        signal_price = np.nan

    if not np.isfinite(
        signal_price
    ):
        continue


    # ========================================================
    # D1 / D3 / D5
    # ========================================================

    for label, days in HORIZONS.items():

        price = next_trading_price(
            prices,
            signal_date,
            days,
        )

        if not np.isfinite(
            price
        ):
            continue

        history.loc[
            index,
            f"{label}_price"
        ] = price

        history.loc[
            index,
            f"{label}_return"
        ] = (
            price
            /
            signal_price
            - 1.0
        ) * 100.0


    # ========================================================
    # HIT
    #
    # For now:
    #
    # Positive return = hit
    #
    # This is deliberately simple.
    # Later we can add target/stop rules.
    # ========================================================

    for label in [
        "d1",
        "d3",
        "d5",
    ]:

        return_value = history.loc[
            index,
            f"{label}_return"
        ]

        if pd.isna(
            return_value
        ):
            continue

        history.loc[
            index,
            f"hit_{label}"
        ] = (
            float(return_value)
            > 0
        )


# ============================================================
# SAVE
# ============================================================

history.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "=========================================="
)

print(
    "STEP 14 - PERFORMANCE TRACKER"
)

print(
    "=========================================="
)

print(
    "HISTORY ROWS :",
    len(history),
)

print(
    "D1 COMPLETE   :",
    history["d1_return"].notna().sum(),
)

print(
    "D3 COMPLETE   :",
    history["d3_return"].notna().sum(),
)

print(
    "D5 COMPLETE   :",
    history["d5_return"].notna().sum(),
)

print()

print(
    "OUTPUT :",
    OUTPUT_FILE,
)

print()

print(
    "STEP 14 OUTPUT : OK"
)

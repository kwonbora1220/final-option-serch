
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)

GREEKS_FILE = os.path.join(
    ANALYSIS_DIR,
    "options_greeks.csv"
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv"
)


def log(message):
    now = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    print(f"[08 STRUCTURE] {now} | {message}")


def find_column(df, candidates):

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
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


def prepare_greeks(df):

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol"
        ]
    )

    type_col = find_column(
        df,
        [
            "option_type",
            "type",
            "contract_type",
            "call_put"
        ]
    )

    strike_col = find_column(
        df,
        [
            "strike",
            "strike_price"
        ]
    )

    price_col = find_column(
        df,
        [
            "current_price",
            "underlying_price",
            "spot",
            "stock_price",
            "price"
        ]
    )

    volume_col = find_column(
        df,
        [
            "volume",
            "option_volume"
        ]
    )

    oi_col = find_column(
        df,
        [
            "open_interest",
            "oi"
        ]
    )

    gex_col = find_column(
        df,
        [
            "gex",
            "gamma_exposure"
        ]
    )

    result = pd.DataFrame(index=df.index)

    if ticker_col is None:
        raise ValueError(
            "Ticker column not found in options_greeks.csv"
        )

    if type_col is None:
        raise ValueError(
            "Option type column not found in options_greeks.csv"
        )

    if strike_col is None:
        raise ValueError(
            "Strike column not found in options_greeks.csv"
        )

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["option_type"] = (
        df[type_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({
            "C": "CALL",
            "P": "PUT"
        })
    )

    result["strike"] = numeric(
        df,
        strike_col
    )

    result["current_price"] = numeric(
        df,
        price_col
    )

    result["volume"] = numeric(
        df,
        volume_col
    )

    result["open_interest"] = numeric(
        df,
        oi_col
    )

    result["gex"] = numeric(
        df,
        gex_col
    )

    return result


def extract_top20_tickers(df):

    log("Detecting TOP20 ticker column")

    candidates = [
        "ticker",
        "symbol",
        "underlying",
        "underlying_symbol",
        "stock",
        "stock_symbol",
        "name"
    ]

    column = find_column(
        df,
        candidates
    )

    if column is None:

        # Last-resort automatic detection:
        # find a column containing common US ticker-like values.

        for col in df.columns:

            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
            )

            if len(values) == 0:
                continue

            sample = values.head(20)

            valid = sample.str.match(
                r"^[A-Z]{1,6}$"
            ).sum()

            if valid >= max(
                2,
                int(len(sample) * 0.5)
            ):

                column = col
                break

    if column is None:

        print()
        print("=" * 72)
        print("TOP20 AVAILABLE COLUMNS")
        print("=" * 72)

        for col in df.columns:
            print(repr(col))

        print("=" * 72)

        raise ValueError(
            "Unable to identify ticker column in top20.csv"
        )

    log(
        f"TOP20 TICKER COLUMN : {column}"
    )

    tickers = (
        df[column]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

    tickers = list(
        dict.fromkeys(tickers)
    )

    log(
        f"TOP20 TICKERS : {len(tickers)}"
    )

    return tickers


def calculate_wall(group, option_type):

    data = group[
        group["option_type"] == option_type
    ].copy()

    data = data.dropna(
        subset=["strike"]
    )

    if data.empty:
        return np.nan

    data["score"] = 0.0

    gex = data["gex"].abs().fillna(0)

    if gex.max() > 0:
        data["score"] += (
            gex / gex.max()
        )

    oi = data["open_interest"].fillna(0)

    if oi.max() > 0:
        data["score"] += (
            oi / oi.max()
        ) * 0.5

    volume = data["volume"].fillna(0)

    if volume.max() > 0:
        data["score"] += (
            volume / volume.max()
        ) * 0.25

    return float(
        data.sort_values(
            "score",
            ascending=False
        ).iloc[0]["strike"]
    )


def calculate_support_resistance(
    group,
    current_price
):

    if pd.isna(current_price):
        return np.nan, np.nan

    strikes = (
        group["strike"]
        .dropna()
        .unique()
    )

    strikes = sorted(
        float(x)
        for x in strikes
    )

    below = [
        x for x in strikes
        if x < current_price
    ]

    above = [
        x for x in strikes
        if x > current_price
    ]

    support = (
        max(below)
        if below
        else np.nan
    )

    resistance = (
        min(above)
        if above
        else np.nan
    )

    return support, resistance


def classify_structure(
    current_price,
    support,
    resistance,
    net_gex
):

    if pd.isna(current_price):
        return "UNAVAILABLE"

    if pd.isna(support) or pd.isna(resistance):
        return "MIXED STRUCTURE"

    if net_gex > 0:
        return "STABILIZED STRUCTURE"

    if net_gex < 0:
        return "HIGHER VOLATILITY STRUCTURE"

    return "MIXED STRUCTURE"


def main():

    log("START")

    if not os.path.exists(GREEKS_FILE):
        raise FileNotFoundError(
            GREEKS_FILE
        )

    if not os.path.exists(TOP20_FILE):
        raise FileNotFoundError(
            TOP20_FILE
        )

    log("Loading Greeks data")

    greeks_raw = pd.read_csv(
        GREEKS_FILE
    )

    log(
        f"GREEKS ROWS : {len(greeks_raw):,}"
    )

    log("Loading TOP20")

    top20 = pd.read_csv(
        TOP20_FILE
    )

    log(
        f"TOP20 ROWS : {len(top20)}"
    )

    top_tickers = extract_top20_tickers(
        top20
    )

    greeks = prepare_greeks(
        greeks_raw
    )

    greeks = greeks[
        greeks["ticker"].isin(
            top_tickers
        )
    ].copy()

    if greeks.empty:
        raise ValueError(
            "No TOP20 option data matched"
        )

    rows = []

    for ticker in top_tickers:

        group = greeks[
            greeks["ticker"] == ticker
        ].copy()

        if group.empty:
            log(
                f"{ticker} | NO OPTION DATA"
            )
            continue

        prices = (
            group["current_price"]
            .dropna()
        )

        current_price = (
            prices.iloc[0]
            if not prices.empty
            else np.nan
        )

        call_wall = calculate_wall(
            group,
            "CALL"
        )

        put_wall = calculate_wall(
            group,
            "PUT"
        )

        support, resistance = (
            calculate_support_resistance(
                group,
                current_price
            )
        )

        call_gex = (
            group.loc[
                group["option_type"] == "CALL",
                "gex"
            ]
            .sum(min_count=1)
        )

        put_gex = (
            group.loc[
                group["option_type"] == "PUT",
                "gex"
            ]
            .sum(min_count=1)
        )

        if pd.isna(call_gex):
            call_gex = 0.0

        if pd.isna(put_gex):
            put_gex = 0.0

        net_gex = (
            call_gex + put_gex
        )

        structure = classify_structure(
            current_price,
            support,
            resistance,
            net_gex
        )

        rows.append({

            "ticker": ticker,

            "current_price": current_price,

            "call_wall": call_wall,

            "put_wall": put_wall,

            "support": support,

            "resistance": resistance,

            "call_gex": call_gex,

            "put_gex": put_gex,

            "net_gex": net_gex,

            "structure": structure,

            "data_source": "CALCULATED"

        })

        log(
            f"{ticker} | "
            f"PRICE {current_price} | "
            f"CALL WALL {call_wall} | "
            f"PUT WALL {put_wall} | "
            f"NET GEX {net_gex:.4f} | "
            f"{structure}"
        )

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

    print()
    print("=" * 72)
    print("🔎 STEP 8 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT GREEKS ROWS : "
        f"{len(greeks_raw):,}"
    )

    print(
        f"TOP20 TICKERS     : "
        f"{len(top_tickers)}"
    )

    print(
        f"STRUCTURE ROWS    : "
        f"{len(output)}"
    )

    print(
        f"STRUCTURE TICKERS : "
        f"{output['ticker'].nunique()}"
    )

    print(
        "CALL WALL VALID   : "
        f"{output['call_wall'].notna().sum()}"
    )

    print(
        "PUT WALL VALID    : "
        f"{output['put_wall'].notna().sum()}"
    )

    print(
        "NET GEX VALID     : "
        f"{output['net_gex'].notna().sum()}"
    )

    print(
        "STRUCTURE VALID   : "
        f"{output['structure'].notna().sum()}"
    )

    print(
        "OUTPUT FILE       : "
        "data/analysis/structure.csv"
    )

    print("=" * 72)

    log(
        "STEP 8 STRUCTURE COMPLETE"
    )


if __name__ == "__main__":
    main()


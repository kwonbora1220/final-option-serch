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
        f"[08 STRUCTURE] {now} | {message}"
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
# PREPARE GREEKS
# ============================================================

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
            "underlying_last"
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

    if ticker_col is None:
        raise ValueError(
            "Ticker column not found in options_greeks.csv"
        )

    if type_col is None:
        raise ValueError(
            "Option type column not found"
        )

    if strike_col is None:
        raise ValueError(
            "Strike column not found"
        )

    result = pd.DataFrame(index=df.index)

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


# ============================================================
# TOP20
# ============================================================

def extract_top20_tickers(df):

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

        for col in df.columns:

            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
            )

            sample = values.head(20)

            if sample.empty:
                continue

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

        raise ValueError(
            "Unable to identify TOP20 ticker column"
        )

    tickers = (
        df[column]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    return tickers


# ============================================================
# WALL CALCULATION
# ============================================================

def calculate_wall(
    group,
    option_type,
    current_price
):

    data = group[
        group["option_type"] == option_type
    ].copy()

    data = data.dropna(
        subset=["strike"]
    )

    if data.empty:
        return np.nan

    # --------------------------------------------------------
    # Directional filter
    # --------------------------------------------------------

    if not pd.isna(current_price):

        if option_type == "CALL":

            directional = data[
                data["strike"] >= current_price
            ]

        else:

            directional = data[
                data["strike"] <= current_price
            ]

        if not directional.empty:
            data = directional

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    gex = data["gex"].abs().fillna(0)
    oi = data["open_interest"].fillna(0)
    volume = data["volume"].fillna(0)

    def normalize(series):

        maximum = series.max()

        if pd.isna(maximum) or maximum <= 0:

            return pd.Series(
                0.0,
                index=series.index
            )

        return series / maximum

    data["gex_score"] = normalize(gex)
    data["oi_score"] = normalize(oi)
    data["volume_score"] = normalize(volume)

    # GEX 중심
    data["wall_score"] = (
        data["gex_score"] * 0.50
        + data["oi_score"] * 0.30
        + data["volume_score"] * 0.20
    )

    data = data.sort_values(
        "wall_score",
        ascending=False
    )

    return float(
        data.iloc[0]["strike"]
    )


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(
    group,
    current_price
):

    if pd.isna(current_price):

        return np.nan, np.nan

    data = group.dropna(
        subset=["strike"]
    ).copy()

    if data.empty:

        return np.nan, np.nan

    # --------------------------------------------------------
    # Aggregate by strike
    # --------------------------------------------------------

    grouped = (
        data
        .groupby(
            ["option_type", "strike"],
            as_index=False
        )
        .agg({
            "gex": lambda x: x.abs().sum(),
            "open_interest": "sum",
            "volume": "sum"
        })
    )

    grouped["strength"] = (
        grouped["gex"].fillna(0)
        + grouped["open_interest"].fillna(0) * 0.01
        + grouped["volume"].fillna(0) * 0.10
    )

    below = grouped[
        grouped["strike"] < current_price
    ].copy()

    above = grouped[
        grouped["strike"] > current_price
    ].copy()

    support = np.nan
    resistance = np.nan

    if not below.empty:

        support = float(
            below.sort_values(
                "strength",
                ascending=False
            ).iloc[0]["strike"]
        )

    if not above.empty:

        resistance = float(
            above.sort_values(
                "strength",
                ascending=False
            ).iloc[0]["strike"]
        )

    return support, resistance


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def classify_structure(
    current_price,
    support,
    resistance,
    call_wall,
    put_wall,
    net_gex
):

    if pd.isna(current_price):

        return "UNAVAILABLE"

    if (
        pd.isna(call_wall)
        and pd.isna(put_wall)
    ):

        return "UNAVAILABLE"

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    if net_gex > 0:

        gex_state = "POSITIVE GEX"

    elif net_gex < 0:

        gex_state = "NEGATIVE GEX"

    else:

        gex_state = "NEUTRAL GEX"

    # --------------------------------------------------------
    # Price position
    # --------------------------------------------------------

    if (
        not pd.isna(support)
        and current_price > support
    ):

        support_state = "ABOVE SUPPORT"

    else:

        support_state = "SUPPORT RISK"

    if (
        not pd.isna(resistance)
        and current_price < resistance
    ):

        resistance_state = "BELOW RESISTANCE"

    else:

        resistance_state = "RESISTANCE RISK"

    # --------------------------------------------------------
    # Bullish / Bearish
    # --------------------------------------------------------

    bullish = 0
    bearish = 0

    if not pd.isna(put_wall):

        if current_price > put_wall:
            bullish += 1

        else:
            bearish += 1

    if not pd.isna(call_wall):

        if current_price < call_wall:
            bullish += 1

        else:
            bearish += 1

    if net_gex > 0:
        bullish += 1

    elif net_gex < 0:
        bearish += 1

    if bullish >= 2:

        direction = "BULLISH"

    elif bearish >= 2:

        direction = "BEARISH"

    else:

        direction = "NEUTRAL"

    return (
        f"{direction} | "
        f"{gex_state} | "
        f"{support_state} | "
        f"{resistance_state}"
    )


# ============================================================
# MAIN
# ============================================================

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

    greeks_raw = pd.read_csv(
        GREEKS_FILE
    )

    top20 = pd.read_csv(
        TOP20_FILE
    )

    log(
        f"GREEKS ROWS : {len(greeks_raw):,}"
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

    for rank, ticker in enumerate(
        top_tickers,
        start=1
    ):

        group = greeks[
            greeks["ticker"] == ticker
        ].copy()

        if group.empty:

            continue

        prices = (
            group["current_price"]
            .dropna()
        )

        current_price = (
            prices.median()
            if not prices.empty
            else np.nan
        )

        call_wall = calculate_wall(
            group,
            "CALL",
            current_price
        )

        put_wall = calculate_wall(
            group,
            "PUT",
            current_price
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

        call_gex = (
            0.0
            if pd.isna(call_gex)
            else float(call_gex)
        )

        put_gex = (
            0.0
            if pd.isna(put_gex)
            else float(put_gex)
        )

        net_gex = (
            call_gex + put_gex
        )

        structure = classify_structure(
            current_price,
            support,
            resistance,
            call_wall,
            put_wall,
            net_gex
        )

        rows.append({

            "rank": rank,

            "ticker": ticker,

            "current_price":
                current_price,

            "call_wall":
                call_wall,

            "put_wall":
                put_wall,

            "support":
                support,

            "resistance":
                resistance,

            "call_gex":
                call_gex,

            "put_gex":
                put_gex,

            "net_gex":
                net_gex,

            "structure":
                structure,

            "data_source":
                "CALCULATED"

        })

        log(
            f"{ticker} | "
            f"PRICE {current_price} | "
            f"CALL WALL {call_wall} | "
            f"PUT WALL {put_wall} | "
            f"GEX {net_gex:.4f} | "
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
        f"INPUT GREEKS ROWS : {len(greeks_raw):,}"
    )

    print(
        f"TOP20 TICKERS     : {len(top_tickers)}"
    )

    print(
        f"STRUCTURE ROWS    : {len(output)}"
    )

    print(
        f"CALL WALL VALID   : "
        f"{output['call_wall'].notna().sum()}"
    )

    print(
        f"PUT WALL VALID    : "
        f"{output['put_wall'].notna().sum()}"
    )

    print(
        f"NET GEX VALID     : "
        f"{output['net_gex'].notna().sum()}"
    )

    print(
        f"STRUCTURE VALID   : "
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

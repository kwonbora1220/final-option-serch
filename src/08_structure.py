from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

GREEKS_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "options_greeks.csv"
)

TOP20_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "top20.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "analysis"
    / "structure.csv"
)


# ============================================================
# HELPERS
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def find_col(df, names):

    normalized = {
        str(c)
        .strip()
        .lower()
        .replace(" ", "_"): c
        for c in df.columns
    }

    for name in names:

        key = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def normalize_option_type(series):

    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
        .replace(
            {
                "C": "CALL",
                "CALLS": "CALL",
                "P": "PUT",
                "PUTS": "PUT",
            }
        )
    )


def safe(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# OPTION STRENGTH
# ============================================================

def calculate_strength(df):

    gamma = df["gamma"].abs().fillna(0.0)
    oi = df["open_interest"].fillna(0.0)
    volume = df["volume"].fillna(0.0)

    return (
        np.log1p(gamma * oi)
        +
        0.25 * np.log1p(volume)
    )


# ============================================================
# CALL / PUT WALL
# ============================================================

def choose_wall(
    df,
    option_type,
    price,
):

    side = df[
        df["option_type"] == option_type
    ].copy()

    if side.empty:
        return np.nan

    if option_type == "CALL":

        side = side[
            side["strike"] >= price
        ]

    else:

        side = side[
            side["strike"] <= price
        ]

    if side.empty:
        return np.nan

    side["distance"] = (
        (side["strike"] - price).abs()
        / price
    )

    side = side[
        side["distance"] <= 0.20
    ].copy()

    if side.empty:
        return np.nan

    side["strength"] = calculate_strength(
        side
    )

    side["selection"] = (
        side["strength"]
        +
        3.0
        /
        (
            1.0
            +
            side["distance"] * 20.0
        )
    )

    best = (
        side
        .sort_values(
            [
                "selection",
                "strength",
                "open_interest",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return safe(
        best["strike"]
    )


# ============================================================
# GEX PROXY
#
# Gamma is calculated in STEP 4.
#
# GEX proxy:
#
# gamma
# × open interest
# × contract multiplier
# × spot²
# × 0.01
#
# CALL = positive
# PUT  = negative
#
# This is a MODELLED GEX PROXY.
# It is NOT exchange/dealer supplied GEX.
# ============================================================

def calculate_gex(
    df,
    option_type,
    price,
):

    side = df[
        df["option_type"] == option_type
    ].copy()

    side = side[
        side["gamma"].notna()
        &
        side["open_interest"].notna()
        &
        side["gamma"].gt(0)
        &
        side["open_interest"].gt(0)
    ].copy()

    if side.empty:
        return np.nan

    if not np.isfinite(price) or price <= 0:
        return np.nan

    gamma = (
        pd.to_numeric(
            side["gamma"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    oi = (
        pd.to_numeric(
            side["open_interest"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    contract_multiplier = 100.0

    # Standardized 1% underlying move scaling.
    gex = (
        gamma
        *
        oi
        *
        contract_multiplier
        *
        (price ** 2)
        *
        0.01
    )

    value = gex.sum()

    if not np.isfinite(value):
        return np.nan

    value = float(value)

    if option_type == "PUT":
        value *= -1.0

    return value


# ============================================================
# SUPPORT
# ============================================================

def choose_support(
    df,
    price,
):

    puts = df[
        (df["option_type"] == "PUT")
        &
        (df["strike"] < price)
    ].copy()

    if puts.empty:
        return np.nan

    puts["distance"] = (
        price - puts["strike"]
    ) / price

    puts = puts[
        puts["distance"] <= 0.15
    ].copy()

    if puts.empty:
        return np.nan

    puts["strength"] = calculate_strength(
        puts
    )

    puts["score"] = (
        puts["strength"]
        +
        3.0
        /
        (
            1.0
            +
            puts["distance"] * 20.0
        )
    )

    best = (
        puts
        .sort_values(
            ["score", "strength"],
            ascending=False,
        )
        .iloc[0]
    )

    return safe(
        best["strike"]
    )


# ============================================================
# RESISTANCE
# ============================================================

def choose_resistance(
    df,
    price,
):

    calls = df[
        (df["option_type"] == "CALL")
        &
        (df["strike"] > price)
    ].copy()

    if calls.empty:
        return np.nan

    calls["distance"] = (
        calls["strike"] - price
    ) / price

    calls = calls[
        calls["distance"] <= 0.15
    ].copy()

    if calls.empty:
        return np.nan

    calls["strength"] = calculate_strength(
        calls
    )

    calls["score"] = (
        calls["strength"]
        +
        3.0
        /
        (
            1.0
            +
            calls["distance"] * 20.0
        )
    )

    best = (
        calls
        .sort_values(
            ["score", "strength"],
            ascending=False,
        )
        .iloc[0]
    )

    return safe(
        best["strike"]
    )


# ============================================================
# WALL STRUCTURE
# ============================================================

def wall_structure(
    price,
    call_wall,
    put_wall,
):

    if not np.isfinite(price):
        return "WALL UNAVAILABLE"

    if (
        np.isfinite(call_wall)
        and
        np.isfinite(put_wall)
    ):

        if put_wall < call_wall:

            if price > call_wall:
                return "BULLISH BREAKOUT"

            if price < put_wall:
                return "BEARISH BREAKDOWN"

            return "RANGE"

        if put_wall == call_wall:
            return "SINGLE WALL"

    if np.isfinite(call_wall):

        if price > call_wall:
            return "ABOVE CALL WALL"

        return "BELOW CALL WALL"

    if np.isfinite(put_wall):

        if price < put_wall:
            return "BELOW PUT WALL"

        return "ABOVE PUT WALL"

    return "WALL UNAVAILABLE"


# ============================================================
# PRICE LOCATION
# ============================================================

def price_location(
    price,
    support,
    resistance,
):

    result = []

    if np.isfinite(support):

        d = (
            price - support
        ) / price

        if 0 <= d <= 0.02:

            result.append(
                "NEAR SUPPORT"
            )

        elif price > support:

            result.append(
                "ABOVE SUPPORT"
            )

        else:

            result.append(
                "BELOW SUPPORT"
            )

    else:

        result.append(
            "SUPPORT UNAVAILABLE"
        )

    if np.isfinite(resistance):

        d = (
            resistance - price
        ) / price

        if 0 <= d <= 0.02:

            result.append(
                "NEAR RESISTANCE"
            )

        elif price < resistance:

            result.append(
                "BELOW RESISTANCE"
            )

        else:

            result.append(
                "ABOVE RESISTANCE"
            )

    else:

        result.append(
            "RESISTANCE UNAVAILABLE"
        )

    return " | ".join(result)


# ============================================================
# GEX STRUCTURE
# ============================================================

def classify_gex(net_gex):

    if not np.isfinite(net_gex):
        return "GEX UNAVAILABLE"

    if net_gex > 0:
        return "POSITIVE GEX"

    if net_gex < 0:
        return "NEGATIVE GEX"

    return "NEUTRAL GEX"


# ============================================================
# OVERALL STRUCTURE
# ============================================================

def classify_structure(
    price,
    call_wall,
    put_wall,
    net_gex,
):

    wall = wall_structure(
        price,
        call_wall,
        put_wall,
    )

    # First priority:
    # actual price breaking a major wall.

    if wall in {
        "BULLISH BREAKOUT",
        "ABOVE CALL WALL",
    }:

        return "BULLISH"

    if wall in {
        "BEARISH BREAKDOWN",
        "BELOW PUT WALL",
    }:

        return "BEARISH"

    # Second priority:
    # modelled GEX.

    if np.isfinite(net_gex):

        if net_gex > 0:
            return "POSITIVE GEX STRUCTURE"

        if net_gex < 0:
            return "NEGATIVE GEX STRUCTURE"

    return "NEUTRAL"


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "RUN STEP 8 STRUCTURE"
    )

    print(
        "=========================================="
    )

    if not GREEKS_FILE.exists():
        raise FileNotFoundError(
            GREEKS_FILE
        )

    if not TOP20_FILE.exists():
        raise FileNotFoundError(
            TOP20_FILE
        )

    greeks = pd.read_csv(
        GREEKS_FILE
    )

    top20 = pd.read_csv(
        TOP20_FILE
    )

    # --------------------------------------------------------
    # TOP20 SYMBOL
    # --------------------------------------------------------

    ticker_col = find_col(
        top20,
        [
            "ticker",
            "symbol",
        ],
    )

    if ticker_col is None:
        raise RuntimeError(
            "TOP20 ticker/symbol column missing"
        )

    top_tickers = (
        top20[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    # --------------------------------------------------------
    # GREEKS COLUMNS
    # --------------------------------------------------------

    symbol_col = find_col(
        greeks,
        [
            "symbol",
            "ticker",
        ],
    )

    type_col = find_col(
        greeks,
        [
            "option_type",
            "type",
        ],
    )

    strike_col = find_col(
        greeks,
        [
            "strike",
            "strike_price",
        ],
    )

    price_col = find_col(
        greeks,
        [
            "underlying_price",
            "current_price",
            "spot_price",
        ],
    )

    gamma_col = find_col(
        greeks,
        [
            "gamma",
        ],
    )

    oi_col = find_col(
        greeks,
        [
            "openInterest",
            "open_interest",
            "oi",
        ],
    )

    volume_col = find_col(
        greeks,
        [
            "volume",
        ],
    )

    required = {
        "symbol": symbol_col,
        "option_type": type_col,
        "strike": strike_col,
        "price": price_col,
        "gamma": gamma_col,
        "open_interest": oi_col,
        "volume": volume_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:

        raise RuntimeError(
            "STEP 8 missing columns: "
            +
            ", ".join(missing)
        )

    # --------------------------------------------------------
    # INTERNAL STANDARDIZATION
    # --------------------------------------------------------

    df = pd.DataFrame()

    df["ticker"] = (
        greeks[symbol_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["option_type"] = (
        normalize_option_type(
            greeks[type_col]
        )
    )

    df["strike"] = numeric(
        greeks[strike_col]
    )

    df["current_price"] = numeric(
        greeks[price_col]
    )

    df["gamma"] = numeric(
        greeks[gamma_col]
    )

    df["open_interest"] = (
        numeric(
            greeks[oi_col]
        )
        .fillna(0)
        .clip(lower=0)
    )

    df["volume"] = (
        numeric(
            greeks[volume_col]
        )
        .fillna(0)
        .clip(lower=0)
    )

    df = df[
        df["ticker"].isin(
            top_tickers
        )
        &
        df["option_type"].isin(
            [
                "CALL",
                "PUT",
            ]
        )
        &
        df["strike"].notna()
        &
        df["current_price"].gt(0)
    ].copy()

    # --------------------------------------------------------
    # BUILD STRUCTURE
    # --------------------------------------------------------

    rows = []

    for ticker in top_tickers:

        group = df[
            df["ticker"] == ticker
        ].copy()

        if group.empty:

            raise RuntimeError(
                f"STEP 8 missing ticker: {ticker}"
            )

        price = safe(
            group["current_price"].median()
        )

        call_wall = choose_wall(
            group,
            "CALL",
            price,
        )

        put_wall = choose_wall(
            group,
            "PUT",
            price,
        )

        support = choose_support(
            group,
            price,
        )

        resistance = choose_resistance(
            group,
            price,
        )

        # ----------------------------------------------------
        # GEX
        # ----------------------------------------------------

        call_gex = calculate_gex(
            group,
            "CALL",
            price,
        )

        put_gex = calculate_gex(
            group,
            "PUT",
            price,
        )

        if (
            np.isfinite(call_gex)
            and
            np.isfinite(put_gex)
        ):

            net_gex = (
                call_gex
                +
                put_gex
            )

        else:

            net_gex = np.nan

        gex_structure = classify_gex(
            net_gex
        )

        structure = classify_structure(
            price,
            call_wall,
            put_wall,
            net_gex,
        )

        wall = wall_structure(
            price,
            call_wall,
            put_wall,
        )

        rows.append(
            {
                "ticker": ticker,

                "current_price": price,

                "call_wall": call_wall,
                "put_wall": put_wall,

                "support": support,
                "resistance": resistance,

                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex": net_gex,

                "structure": structure,

                "price_location": price_location(
                    price,
                    support,
                    resistance,
                ),

                "gex_structure": gex_structure,

                "wall_structure": wall,

                "data_source": (
                    "STEP4_GAMMA_OI_GEX_PROXY"
                ),
            }
        )

    output = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(output) != 20:
        raise RuntimeError(
            "STEP 8 must contain exactly 20 rows"
        )

    if output["ticker"].nunique() != 20:
        raise RuntimeError(
            "STEP 8 must contain 20 unique tickers"
        )

    if output["call_wall"].notna().sum() != 20:
        raise RuntimeError(
            "STEP 8 CALL WALL incomplete"
        )

    if output["put_wall"].notna().sum() != 20:
        raise RuntimeError(
            "STEP 8 PUT WALL incomplete"
        )

    valid_gex = (
        output["net_gex"]
        .notna()
        .sum()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        "STEP 8 STRUCTURE"
    )
    print(
        "=========================================="
    )

    print(
        f"ROWS       : {len(output)}"
    )

    print(
        f"TICKERS    : "
        f"{output['ticker'].nunique()}"
    )

    print(
        f"CALL WALL  : "
        f"{output['call_wall'].notna().sum()}"
    )

    print(
        f"PUT WALL   : "
        f"{output['put_wall'].notna().sum()}"
    )

    print(
        f"NET GEX    : "
        f"{valid_gex}"
    )

    print(
        f"POSITIVE GEX: "
        f"{(output['net_gex'] > 0).sum()}"
    )

    print(
        f"NEGATIVE GEX: "
        f"{(output['net_gex'] < 0).sum()}"
    )

    print(
        f"GEX UNAVAILABLE: "
        f"{output['net_gex'].isna().sum()}"
    )

    print(
        f"STRUCTURE  : "
        f"{output['structure'].notna().sum()}"
    )

    print()
    print(
        output[
            [
                "ticker",
                "call_gex",
                "put_gex",
                "net_gex",
                "gex_structure",
                "structure",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        f"OUTPUT FILE : {OUTPUT_FILE}"
    )

    print(
        "STEP 8 OUTPUT : OK"
    )


if __name__ == "__main__":
    main()

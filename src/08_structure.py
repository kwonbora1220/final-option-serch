from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

GREEKS_PATH = BASE_DIR / "data" / "analysis" / "options_greeks.csv"
TOP20_PATH = BASE_DIR / "data" / "analysis" / "top20.csv"
OUTPUT_PATH = BASE_DIR / "data" / "analysis" / "structure.csv"


# ============================================================
# BASIC HELPERS
# ============================================================

def norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    normalized = {
        norm_col(c): c
        for c in df.columns
    }

    for candidate in candidates:

        key = norm_col(candidate)

        if key in normalized:
            return normalized[key]

    return None


def numeric(series: pd.Series) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_float(value) -> float:

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not GREEKS_PATH.exists():

        raise FileNotFoundError(
            f"Missing Greeks file: {GREEKS_PATH}"
        )

    if not TOP20_PATH.exists():

        raise FileNotFoundError(
            f"Missing TOP20 file: {TOP20_PATH}"
        )

    greeks = pd.read_csv(
        GREEKS_PATH
    )

    top20 = pd.read_csv(
        TOP20_PATH
    )

    greeks.columns = [
        str(c).strip()
        for c in greeks.columns
    ]

    top20.columns = [
        str(c).strip()
        for c in top20.columns
    ]

    return greeks, top20


# ============================================================
# COLUMN DETECTION
# ============================================================

def prepare_columns(
    df: pd.DataFrame,
) -> dict[str, str | None]:

    return {

        "ticker": find_col(
            df,
            [
                "ticker",
                "symbol",
                "underlying",
                "underlying_symbol",
            ],
        ),

        "option_type": find_col(
            df,
            [
                "option_type",
                "type",
                "call_put",
                "cp",
            ],
        ),

        "strike": find_col(
            df,
            [
                "strike",
                "strike_price",
            ],
        ),

        "current_price": find_col(
            df,
            [
                "current_price",
                "underlying_price",
                "stock_price",
                "spot_price",
                "underlying_last",
                "last_underlying_price",
            ],
        ),

        "gamma": find_col(
            df,
            [
                "gamma",
            ],
        ),

        "open_interest": find_col(
            df,
            [
                "open_interest",
                "oi",
            ],
        ),

        "volume": find_col(
            df,
            [
                "volume",
                "option_volume",
            ],
        ),

        "dte": find_col(
            df,
            [
                "DTE",
                "dte",
                "days_to_expiration",
            ],
        ),
    }


# ============================================================
# OPTION STRENGTH
#
# Used for WALL / SUPPORT / RESISTANCE selection.
#
# Gamma x OI is the primary structural signal.
# Volume is only a secondary confirmation.
# ============================================================

def build_strength(
    df: pd.DataFrame,
) -> pd.Series:

    gamma = (
        numeric(
            df["gamma_value"]
        )
        .abs()
        .fillna(0)
    )

    oi = (
        numeric(
            df["open_interest_value"]
        )
        .fillna(0)
        .clip(lower=0)
    )

    volume = (
        numeric(
            df["volume_value"]
        )
        .fillna(0)
        .clip(lower=0)
    )

    gamma_oi = (
        gamma
        * oi
    )

    volume_component = np.log1p(
        volume
    )

    return (
        np.log1p(gamma_oi)
        + 0.15 * volume_component
    )


# ============================================================
# NEARBY LEVEL SELECTION
#
# Support:
#   PUT strikes below current price
#
# Resistance:
#   CALL strikes above current price
#
# The closest meaningful level is preferred.
# Strength is used as confirmation, not as the only factor.
# ============================================================

def choose_nearby_level(
    df: pd.DataFrame,
    current_price: float,
    direction: str,
    max_distance: float = 0.15,
) -> float:

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):
        return np.nan

    if direction == "support":

        candidates = df[
            (df["strike"] < current_price)
            & (
                df["strike"]
                >= current_price
                * (1.0 - max_distance)
            )
        ].copy()

    elif direction == "resistance":

        candidates = df[
            (df["strike"] > current_price)
            & (
                df["strike"]
                <= current_price
                * (1.0 + max_distance)
            )
        ].copy()

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    if candidates.empty:
        return np.nan

    candidates["distance"] = (
        (
            candidates["strike"]
            - current_price
        ).abs()
        / current_price
    )

    # Prefer nearby levels strongly.
    candidates["distance_score"] = (
        1.0
        /
        (
            1.0
            +
            candidates["distance"] * 20.0
        )
    )

    candidates["selection_score"] = (
        candidates["strength"]
        * candidates["distance_score"]
    )

    best = (
        candidates
        .sort_values(
            [
                "selection_score",
                "strength",
                "distance",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return safe_float(
        best["strike"]
    )


# ============================================================
# WALL SELECTION
#
# CALL WALL:
#   strongest CALL strike at / above spot
#
# PUT WALL:
#   strongest PUT strike at / below spot
#
# Unlike support/resistance, wall selection is based primarily
# on structural option positioning.
# ============================================================

def choose_call_wall(
    calls: pd.DataFrame,
    current_price: float,
) -> float:

    if calls.empty:
        return np.nan

    candidates = calls.copy()

    if np.isfinite(current_price):

        candidates = candidates[
            candidates["strike"]
            >= current_price
        ]

    if candidates.empty:
        return np.nan

    best = (
        candidates
        .sort_values(
            [
                "gex_strength",
                "strength",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return safe_float(
        best["strike"]
    )


def choose_put_wall(
    puts: pd.DataFrame,
    current_price: float,
) -> float:

    if puts.empty:
        return np.nan

    candidates = puts.copy()

    if np.isfinite(current_price):

        candidates = candidates[
            candidates["strike"]
            <= current_price
        ]

    if candidates.empty:
        return np.nan

    best = (
        candidates
        .sort_values(
            [
                "gex_strength",
                "strength",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return safe_float(
        best["strike"]
    )


# ============================================================
# GEX CLASSIFICATION
# ============================================================

def classify_gex(
    net_gex: float,
) -> str:

    if not np.isfinite(net_gex):

        return "GEX UNAVAILABLE"

    if net_gex > 0:

        return "POSITIVE GEX"

    if net_gex < 0:

        return "NEGATIVE GEX"

    return "NEUTRAL GEX"


# ============================================================
# WALL RELATIONSHIP
#
# IMPORTANT:
#
# Spot between PUT WALL and CALL WALL
#   -> RANGE
#
# Spot above CALL WALL
#   -> BULLISH BREAKOUT
#
# Spot below PUT WALL
#   -> BEARISH BREAKDOWN
#
# This fixes the previous logic where simply being between
# two walls was incorrectly classified as BULLISH.
# ============================================================

def classify_wall_structure(
    current_price: float,
    call_wall: float,
    put_wall: float,
) -> str:

    if not np.isfinite(current_price):

        return "WALL UNAVAILABLE"

    if (
        np.isfinite(put_wall)
        and np.isfinite(call_wall)
        and put_wall < call_wall
    ):

        if current_price > call_wall:

            return "BULLISH BREAKOUT"

        if current_price < put_wall:

            return "BEARISH BREAKDOWN"

        return "RANGE"

    if np.isfinite(call_wall):

        if current_price > call_wall:
            return "ABOVE CALL WALL"

        return "BELOW CALL WALL"

    if np.isfinite(put_wall):

        if current_price < put_wall:
            return "BELOW PUT WALL"

        return "ABOVE PUT WALL"

    return "WALL UNAVAILABLE"


# ============================================================
# PRICE LOCATION
# ============================================================

def classify_price_location(
    current_price: float,
    support: float,
    resistance: float,
) -> str:

    if not np.isfinite(current_price):

        return "PRICE UNAVAILABLE"

    locations = []

    if np.isfinite(support):

        if current_price > support:

            locations.append(
                "ABOVE SUPPORT"
            )

        elif current_price < support:

            locations.append(
                "BELOW SUPPORT"
            )

        else:

            locations.append(
                "AT SUPPORT"
            )

    else:

        locations.append(
            "SUPPORT UNAVAILABLE"
        )

    if np.isfinite(resistance):

        if current_price < resistance:

            locations.append(
                "BELOW RESISTANCE"
            )

        elif current_price > resistance:

            locations.append(
                "ABOVE RESISTANCE"
            )

        else:

            locations.append(
                "AT RESISTANCE"
            )

    else:

        locations.append(
            "RESISTANCE UNAVAILABLE"
        )

    return " | ".join(
        locations
    )


# ============================================================
# OVERALL STRUCTURE
#
# Structure is intentionally conservative.
#
# GEX does NOT directly make the structure bullish/bearish.
# Price/wall positioning is primary.
#
# RANGE + POSITIVE GEX is still RANGE.
# RANGE + NEGATIVE GEX is still RANGE.
#
# STEP 9 can score GEX separately.
# ============================================================

def classify_structure(
    current_price: float,
    call_wall: float,
    put_wall: float,
    net_gex: float,
    support: float,
    resistance: float,
) -> tuple[
    str,
    str,
    str,
    str,
]:

    gex_structure = classify_gex(
        net_gex
    )

    price_location = classify_price_location(
        current_price,
        support,
        resistance,
    )

    wall_structure = classify_wall_structure(
        current_price,
        call_wall,
        put_wall,
    )

    # --------------------------------------------------------
    # PRIMARY STRUCTURE
    # --------------------------------------------------------

    if wall_structure == "BULLISH BREAKOUT":

        structure = "BULLISH"

    elif wall_structure == "BEARISH BREAKDOWN":

        structure = "BEARISH"

    elif wall_structure == "RANGE":

        structure = "NEUTRAL"

    elif wall_structure == "ABOVE CALL WALL":

        structure = "BULLISH"

    elif wall_structure == "BELOW PUT WALL":

        structure = "BEARISH"

    else:

        structure = "NEUTRAL"

    return (
        structure,
        price_location,
        gex_structure,
        wall_structure,
    )


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(
    ticker: str,
    greeks: pd.DataFrame,
) -> dict:

    cols = prepare_columns(
        greeks
    )

    required = [
        "ticker",
        "option_type",
        "strike",
    ]

    missing = [
        c
        for c in required
        if cols[c] is None
    ]

    if missing:

        raise ValueError(
            "Missing required columns for "
            f"structure analysis: {missing}"
        )

    ticker_col = cols["ticker"]

    df = greeks[
        greeks[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .eq(ticker.upper())
    ].copy()

    if df.empty:

        return {
            "ticker": ticker,
            "current_price": np.nan,
            "call_wall": np.nan,
            "put_wall": np.nan,
            "call_gex": np.nan,
            "put_gex": np.nan,
            "net_gex": np.nan,
            "support": np.nan,
            "resistance": np.nan,
            "structure": "NO DATA",
            "price_location": "UNAVAILABLE",
            "gex_structure": "GEX UNAVAILABLE",
            "wall_structure": "UNAVAILABLE",
            "gex_valid_rows": 0,
            "data_source": "CALCULATED",
        }

    # --------------------------------------------------------
    # NORMALIZE NUMERIC COLUMNS
    # --------------------------------------------------------

    df["strike"] = numeric(
        df[cols["strike"]]
    )

    if cols["current_price"]:

        df["current_price"] = numeric(
            df[cols["current_price"]]
        )

    else:

        df["current_price"] = np.nan

    if cols["gamma"]:

        df["gamma_value"] = numeric(
            df[cols["gamma"]]
        )

    else:

        df["gamma_value"] = np.nan

    if cols["open_interest"]:

        df["open_interest_value"] = (
            numeric(
                df[cols["open_interest"]]
            )
            .fillna(0)
            .clip(lower=0)
        )

    else:

        df["open_interest_value"] = 0.0

    if cols["volume"]:

        df["volume_value"] = (
            numeric(
                df[cols["volume"]]
            )
            .fillna(0)
            .clip(lower=0)
        )

    else:

        df["volume_value"] = 0.0

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_prices = (
        df["current_price"]
        .dropna()
    )

    if current_prices.empty:

        current_price = np.nan

    else:

        current_price = safe_float(
            current_prices.iloc[0]
        )

    # --------------------------------------------------------
    # OPTION TYPE
    # --------------------------------------------------------

    option_type = (
        df[cols["option_type"]]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    calls = df[
        option_type.isin(
            [
                "CALL",
                "C",
            ]
        )
        & df["strike"].notna()
    ].copy()

    puts = df[
        option_type.isin(
            [
                "PUT",
                "P",
            ]
        )
        & df["strike"].notna()
    ].copy()

    # --------------------------------------------------------
    # STRUCTURAL STRENGTH
    # --------------------------------------------------------

    df["strength"] = build_strength(
        df
    )

    calls["strength"] = (
        df.loc[
            calls.index,
            "strength",
        ]
    )

    puts["strength"] = (
        df.loc[
            puts.index,
            "strength",
        ]
    )

    # --------------------------------------------------------
    # GEX
    #
    # IMPORTANT:
    #
    # CALL GEX = + abs(gamma) * OI
    # PUT GEX  = - abs(gamma) * OI
    #
    # This guarantees that PUT exposure is negative even if
    # the source gamma column itself is unsigned.
    #
    # We deliberately do not multiply by spot^2 here because
    # STEP 4 already provides the option Greek values and this
    # STEP is intended to compare structural exposure consistently.
    # --------------------------------------------------------

    calls["gex_value"] = (
        calls["gamma_value"].abs()
        * calls["open_interest_value"]
    )

    puts["gex_value"] = (
        -puts["gamma_value"].abs()
        * puts["open_interest_value"]
    )

    call_gex = (
        calls["gex_value"].sum()
        if not calls.empty
        else np.nan
    )

    put_gex = (
        puts["gex_value"].sum()
        if not puts.empty
        else np.nan
    )

    if (
        np.isfinite(call_gex)
        and np.isfinite(put_gex)
    ):

        net_gex = (
            call_gex
            + put_gex
        )

    else:

        net_gex = np.nan

    # --------------------------------------------------------
    # GEX STRENGTH
    # --------------------------------------------------------

    calls["gex_strength"] = (
        calls["gex_value"].abs()
    )

    puts["gex_strength"] = (
        puts["gex_value"].abs()
    )

    # --------------------------------------------------------
    # WALLS
    # --------------------------------------------------------

    call_wall = choose_call_wall(
        calls,
        current_price,
    )

    put_wall = choose_put_wall(
        puts,
        current_price,
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    #
    # Support MUST be below spot.
    # Resistance MUST be above spot.
    #
    # This prevents absurd levels such as support far below
    # the active trading range from being selected.
    # --------------------------------------------------------

    support = np.nan
    resistance = np.nan

    if np.isfinite(current_price):

        if not puts.empty:

            support = choose_nearby_level(
                puts,
                current_price,
                "support",
                max_distance=0.15,
            )

        if not calls.empty:

            resistance = choose_nearby_level(
                calls,
                current_price,
                "resistance",
                max_distance=0.15,
            )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    (
        structure,
        price_location,
        gex_structure,
        wall_structure,
    ) = classify_structure(
        current_price=current_price,
        call_wall=call_wall,
        put_wall=put_wall,
        net_gex=net_gex,
        support=support,
        resistance=resistance,
    )

    # --------------------------------------------------------
    # VALID GEX ROW COUNT
    # --------------------------------------------------------

    gex_valid_rows = int(
        (
            df["gamma_value"]
            .notna()
            &
            (
                df["open_interest_value"]
                >= 0
            )
        ).sum()
    )

    return {

        "ticker": ticker,

        "current_price": current_price,

        "call_wall": call_wall,

        "put_wall": put_wall,

        "call_gex": call_gex,

        "put_gex": put_gex,

        "net_gex": net_gex,

        "support": support,

        "resistance": resistance,

        "structure": structure,

        "price_location": price_location,

        "gex_structure": gex_structure,

        "wall_structure": wall_structure,

        "gex_valid_rows": gex_valid_rows,

        "data_source": "CALCULATED",
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.parse_args()

    greeks, top20 = load_data()

    # --------------------------------------------------------
    # TOP20 TICKER COLUMN
    # --------------------------------------------------------

    top20_ticker_col = find_col(
        top20,
        [
            "ticker",
            "symbol",
            "underlying",
        ],
    )

    if top20_ticker_col is None:

        raise ValueError(
            "Could not find ticker column "
            "in TOP20 file."
        )

    tickers = (
        top20[
            top20_ticker_col
        ]
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(
        f"TOP20 TICKERS : {len(tickers)}"
    )

    if len(tickers) != 20:

        raise RuntimeError(
            "Expected exactly 20 TOP20 tickers, "
            f"got {len(tickers)}"
        )

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    rows = []

    for ticker in tickers:

        try:

            result = process_ticker(
                ticker,
                greeks,
            )

            rows.append(
                result
            )

            print(
                f"{ticker:<8} "
                f"STRUCTURE="
                f"{result['structure']:<8} "
                f"GEX="
                f"{result['gex_structure']:<16} "
                f"WALL="
                f"{result['wall_structure']:<20} "
                f"SUPPORT="
                f"{result['support']} "
                f"RESISTANCE="
                f"{result['resistance']}"
            )

        except Exception as exc:

            print(
                f"{ticker} ERROR: {exc}"
            )

            rows.append(
                {
                    "ticker": ticker,
                    "current_price": np.nan,
                    "call_wall": np.nan,
                    "put_wall": np.nan,
                    "call_gex": np.nan,
                    "put_gex": np.nan,
                    "net_gex": np.nan,
                    "support": np.nan,
                    "resistance": np.nan,
                    "structure": "ERROR",
                    "price_location": "UNAVAILABLE",
                    "gex_structure": "GEX UNAVAILABLE",
                    "wall_structure": "UNAVAILABLE",
                    "gex_valid_rows": 0,
                    "data_source": "CALCULATED",
                }
            )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        "STEP 8 COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        f"ROWS       : {len(output)}"
    )

    print(
        f"STRUCTURES : "
        f"{output['structure'].value_counts().to_dict()}"
    )

    print(
        f"GEX        : "
        f"{output['gex_structure'].value_counts().to_dict()}"
    )

    print(
        f"WALLS      : "
        f"{output['wall_structure'].value_counts().to_dict()}"
    )

    print(
        f"OUTPUT     : {OUTPUT_PATH}"
    )

    if len(output) != 20:

        raise RuntimeError(
            "STEP 8 must contain exactly 20 rows."
        )

    if (
        output["ticker"]
        .nunique()
        != 20
    ):

        raise RuntimeError(
            "STEP 8 must contain exactly "
            "20 unique tickers."
        )

    # --------------------------------------------------------
    # DATA QUALITY CHECK
    # --------------------------------------------------------

    if output[
        "current_price"
    ].notna().sum() < 20:

        raise RuntimeError(
            "Some STEP 8 rows have no "
            "current price."
        )

    print()
    print(
        "STEP 8 OUTPUT : OK"
    )


if __name__ == "__main__":

    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

GREEKS_PATH = BASE_DIR / "data" / "analysis" / "options_greeks.csv"
TOP20_PATH = BASE_DIR / "data" / "analysis" / "top20.csv"
OUTPUT_PATH = BASE_DIR / "data" / "analysis" / "structure.csv"


def norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {norm_col(c): c for c in df.columns}

    for candidate in candidates:
        key = norm_col(candidate)
        if key in normalized:
            return normalized[key]

    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_float(value) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return np.nan


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GREEKS_PATH.exists():
        raise FileNotFoundError(f"Missing Greeks file: {GREEKS_PATH}")

    if not TOP20_PATH.exists():
        raise FileNotFoundError(f"Missing TOP20 file: {TOP20_PATH}")

    greeks = pd.read_csv(GREEKS_PATH)
    top20 = pd.read_csv(TOP20_PATH)

    greeks.columns = [str(c).strip() for c in greeks.columns]
    top20.columns = [str(c).strip() for c in top20.columns]

    return greeks, top20


def prepare_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "ticker": find_col(
            df,
            ["ticker", "symbol", "underlying", "underlying_symbol"],
        ),
        "option_type": find_col(
            df,
            ["option_type", "type", "call_put", "cp"],
        ),
        "strike": find_col(
            df,
            ["strike", "strike_price"],
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
            ["gamma"],
        ),
        "open_interest": find_col(
            df,
            ["open_interest", "oi"],
        ),
        "volume": find_col(
            df,
            ["volume", "option_volume"],
        ),
    }


def build_strength(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.Series:
    gamma = (
        numeric(df[cols["gamma"]]).abs()
        if cols["gamma"]
        else pd.Series(0.0, index=df.index)
    )

    oi = (
        numeric(df[cols["open_interest"]]).fillna(0)
        if cols["open_interest"]
        else pd.Series(0.0, index=df.index)
    )

    volume = (
        numeric(df[cols["volume"]]).fillna(0)
        if cols["volume"]
        else pd.Series(0.0, index=df.index)
    )

    # Log scaling prevents huge OI/volume outliers from dominating.
    return (
        np.log1p(gamma)
        + np.log1p(oi)
        + np.log1p(volume)
    )


def choose_nearby_level(
    df: pd.DataFrame,
    current_price: float,
    direction: str,
    max_distance: float = 0.20,
) -> float:
    """
    Choose a meaningful nearby PUT/CALL level.

    direction:
        support    -> PUT strike below current
        resistance -> CALL strike above current

    A level farther than 20% from the current price is ignored.
    """

    if not np.isfinite(current_price) or current_price <= 0:
        return np.nan

    if direction == "support":
        candidates = df[
            (df["strike"] < current_price)
            & (
                df["strike"]
                >= current_price * (1.0 - max_distance)
            )
        ].copy()

    else:
        candidates = df[
            (df["strike"] > current_price)
            & (
                df["strike"]
                <= current_price * (1.0 + max_distance)
            )
        ].copy()

    if candidates.empty:
        return np.nan

    # Prefer nearby levels, while still considering option strength.
    candidates["distance"] = (
        (candidates["strike"] - current_price).abs()
        / current_price
    )

    candidates["distance_score"] = 1.0 / (
        1.0 + candidates["distance"] * 10.0
    )

    candidates["selection_score"] = (
        candidates["strength"] * candidates["distance_score"]
    )

    best = candidates.sort_values(
        ["selection_score", "strength"],
        ascending=False,
    ).iloc[0]

    return safe_float(best["strike"])


def classify_structure(
    current_price: float,
    call_wall: float,
    put_wall: float,
    net_gex: float,
    support: float,
    resistance: float,
) -> tuple[str, str, str, str]:
    """
    Return:
        structure
        price_location
        gex_structure
        wall_structure
    """

    bullish_votes = 0
    bearish_votes = 0

    # ---------------------------------------------------------
    # WALL POSITION
    # ---------------------------------------------------------

    if np.isfinite(put_wall):
        if current_price > put_wall:
            bullish_votes += 1
        elif current_price < put_wall:
            bearish_votes += 1

    if np.isfinite(call_wall):
        if current_price < call_wall:
            bullish_votes += 1
        elif current_price > call_wall:
            bearish_votes += 1

    # ---------------------------------------------------------
    # PRICE LOCATION
    # ---------------------------------------------------------

    if np.isfinite(support):
        if current_price > support:
            price_location = "ABOVE SUPPORT"
        else:
            price_location = "BELOW SUPPORT"
    else:
        price_location = "SUPPORT UNAVAILABLE"

    if np.isfinite(resistance):
        if current_price < resistance:
            resistance_location = "BELOW RESISTANCE"
        else:
            resistance_location = "ABOVE RESISTANCE"
    else:
        resistance_location = "RESISTANCE UNAVAILABLE"

    # ---------------------------------------------------------
    # GEX
    # ---------------------------------------------------------

    if np.isfinite(net_gex):
        if net_gex > 0:
            gex_structure = "POSITIVE GEX"
        elif net_gex < 0:
            gex_structure = "NEGATIVE GEX"
        else:
            gex_structure = "NEUTRAL GEX"
    else:
        gex_structure = "GEX UNAVAILABLE"

    # GEX is deliberately NOT counted as a bullish/bearish vote.
    #
    # This prevents:
    #
    #   wall bullish + GEX bullish
    #
    # from automatically becoming a strong bullish structure.
    #
    # GEX will be scored independently in STEP 9.

    if bullish_votes >= 2 and bearish_votes == 0:
        structure = "BULLISH"
    elif bearish_votes >= 2 and bullish_votes == 0:
        structure = "BEARISH"
    else:
        structure = "NEUTRAL"

    return (
        structure,
        f"{price_location} | {resistance_location}",
        gex_structure,
        (
            f"BULLISH WALLS {bullish_votes}"
            if bullish_votes > bearish_votes
            else (
                f"BEARISH WALLS {bearish_votes}"
                if bearish_votes > bullish_votes
                else "MIXED WALLS"
            )
        ),
    )


def process_ticker(
    ticker: str,
    greeks: pd.DataFrame,
) -> dict:
    cols = prepare_columns(greeks)

    required = [
        "ticker",
        "option_type",
        "strike",
    ]

    missing = [c for c in required if cols[c] is None]

    if missing:
        raise ValueError(
            f"Missing required columns for structure analysis: {missing}"
        )

    df = greeks[
        greeks[cols["ticker"]].astype(str).str.upper().eq(ticker.upper())
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
        }

    df["strike"] = numeric(df[cols["strike"]])

    if cols["current_price"]:
        df["current_price"] = numeric(df[cols["current_price"]])
    else:
        df["current_price"] = np.nan

    # Use the first valid underlying price.
    current_prices = df["current_price"].dropna()

    if current_prices.empty:
        current_price = np.nan
    else:
        current_price = safe_float(current_prices.iloc[0])

    df["gamma_value"] = (
        numeric(df[cols["gamma"]])
        if cols["gamma"]
        else 0.0
    )

    df["open_interest_value"] = (
        numeric(df[cols["open_interest"]]).fillna(0)
        if cols["open_interest"]
        else 0.0
    )

    df["volume_value"] = (
        numeric(df[cols["volume"]]).fillna(0)
        if cols["volume"]
        else 0.0
    )

    df["strength"] = build_strength(df, cols)

    option_type = (
        df[cols["option_type"]]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    calls = df[
        option_type.isin(["CALL", "C"])
        & df["strike"].notna()
    ].copy()

    puts = df[
        option_type.isin(["PUT", "P"])
        & df["strike"].notna()
    ].copy()

    call_gex = (
        (calls["gamma_value"] * calls["open_interest_value"])
        .sum()
        if not calls.empty
        else np.nan
    )

    put_gex = (
        (puts["gamma_value"] * puts["open_interest_value"])
        .sum()
        if not puts.empty
        else np.nan
    )

    net_gex = (
        call_gex + put_gex
        if np.isfinite(call_gex) and np.isfinite(put_gex)
        else np.nan
    )

    # ---------------------------------------------------------
    # WALLS
    # ---------------------------------------------------------

    call_wall = np.nan
    put_wall = np.nan

    if not calls.empty:
        call_wall_candidates = calls.copy()

        if np.isfinite(current_price):
            call_wall_candidates = call_wall_candidates[
                call_wall_candidates["strike"] >= current_price
            ]

        if not call_wall_candidates.empty:
            call_wall = safe_float(
                call_wall_candidates.sort_values(
                    "strength",
                    ascending=False,
                ).iloc[0]["strike"]
            )

    if not puts.empty:
        put_wall_candidates = puts.copy()

        if np.isfinite(current_price):
            put_wall_candidates = put_wall_candidates[
                put_wall_candidates["strike"] <= current_price
            ]

        if not put_wall_candidates.empty:
            put_wall = safe_float(
                put_wall_candidates.sort_values(
                    "strength",
                    ascending=False,
                ).iloc[0]["strike"]
            )

    # ---------------------------------------------------------
    # SUPPORT / RESISTANCE
    # ---------------------------------------------------------

    support = np.nan
    resistance = np.nan

    if np.isfinite(current_price):

        if not puts.empty:
            support = choose_nearby_level(
                puts,
                current_price,
                "support",
                max_distance=0.20,
            )

        if not calls.empty:
            resistance = choose_nearby_level(
                calls,
                current_price,
                "resistance",
                max_distance=0.20,
            )

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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    greeks, top20 = load_data()

    top20_ticker_col = find_col(
        top20,
        ["ticker", "symbol", "underlying"],
    )

    if top20_ticker_col is None:
        raise ValueError(
            "Could not find ticker column in TOP20 file."
        )

    tickers = (
        top20[top20_ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    print(f"TOP20 TICKERS : {len(tickers)}")

    rows = []

    for ticker in tickers:
        try:
            result = process_ticker(ticker, greeks)
            rows.append(result)

            print(
                f"{ticker:<8} "
                f"STRUCTURE={result['structure']:<8} "
                f"GEX={result['gex_structure']:<16} "
                f"SUPPORT={result['support']} "
                f"RESISTANCE={result['resistance']}"
            )

        except Exception as exc:
            print(f"{ticker} ERROR: {exc}")

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
                }
            )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("STEP 8 COMPLETE")
    print(f"ROWS       : {len(output)}")
    print(
        f"STRUCTURES : "
        f"{output['structure'].value_counts().to_dict()}"
    )
    print(f"OUTPUT     : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/normalized/options_normalized.csv"
OUTPUT_DIR = "data/analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "options_greeks.csv")

# Annualized risk-free rate assumption.
# This is an explicit model assumption, not market-provided option data.
RISK_FREE_RATE = 0.04

# Standard equity option contract multiplier.
CONTRACT_MULTIPLIER = 100


# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[04 GREEKS] {now} | {message}")


# ============================================================
# NORMAL DISTRIBUTION
# ============================================================

def normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# BLACK-SCHOLES d1
# ============================================================

def calculate_d1(S, K, T, r, sigma):
    return (
        math.log(S / K)
        + (r + 0.5 * sigma * sigma) * T
    ) / (sigma * math.sqrt(T))


# ============================================================
# GREEKS
# ============================================================

def calculate_greeks(row):
    result = {
        "delta": np.nan,
        "gamma": np.nan,
        "vega": np.nan,
        "delta_source": "UNAVAILABLE",
        "gamma_source": "UNAVAILABLE",
        "vega_source": "UNAVAILABLE",
        "greeks_status": "UNAVAILABLE",
    }

    try:
        S = float(row["underlying_price"])
        K = float(row["strike"])
        dte = float(row["DTE"])
        iv = float(row["impliedVolatility"])

        option_type = str(
            row["option_type"]
        ).upper().strip()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not np.isfinite(S) or S <= 0:
            return result

        if not np.isfinite(K) or K <= 0:
            return result

        if not np.isfinite(dte) or dte <= 0:
            return result

        if not np.isfinite(iv) or iv <= 0:
            return result

        if option_type not in {"CALL", "PUT"}:
            return result

        # ----------------------------------------------------
        # Convert units
        # ----------------------------------------------------

        T = dte / 365.0
        sigma = iv

        if T <= 0 or sigma <= 0:
            return result

        d1 = calculate_d1(
            S,
            K,
            T,
            RISK_FREE_RATE,
            sigma,
        )

        pdf_d1 = normal_pdf(d1)

        # ----------------------------------------------------
        # Delta
        # ----------------------------------------------------

        if option_type == "CALL":
            delta = normal_cdf(d1)
        else:
            delta = normal_cdf(d1) - 1.0

        # ----------------------------------------------------
        # Gamma
        # ----------------------------------------------------

        gamma = (
            pdf_d1
            / (S * sigma * math.sqrt(T))
        )

        # ----------------------------------------------------
        # Vega
        #
        # Standard Black-Scholes vega per 1.00 change
        # in volatility.
        #
        # For a 1 percentage-point IV change,
        # divide by 100.
        # ----------------------------------------------------

        vega = (
            S
            * pdf_d1
            * math.sqrt(T)
            / 100.0
        )

        result["delta"] = delta
        result["gamma"] = gamma
        result["vega"] = vega

        result["delta_source"] = "CALCULATED"
        result["gamma_source"] = "CALCULATED"
        result["vega_source"] = "CALCULATED"
        result["greeks_status"] = "CALCULATED"

        return result

    except Exception:
        return result


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    log(f"INPUT : {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    input_rows = len(df)

    log(f"INPUT ROWS : {input_rows}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "symbol",
        "strike",
        "DTE",
        "impliedVolatility",
        "option_type",
    ]

    # underlying_price may have a different name depending
    # on the normalization implementation.
    possible_price_columns = [
        "underlying_price",
        "current_price",
        "stock_price",
        "underlyingPrice",
    ]

    missing_required = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    price_column = None

    for col in possible_price_columns:
        if col in df.columns:
            price_column = col
            break

    if price_column is None:
        raise ValueError(
            "No underlying price column found. "
            f"Expected one of: {possible_price_columns}"
        )

    # --------------------------------------------------------
    # Standardize price column internally
    # --------------------------------------------------------

    if price_column != "underlying_price":
        df["underlying_price"] = pd.to_numeric(
            df[price_column],
            errors="coerce",
        )
    else:
        df["underlying_price"] = pd.to_numeric(
            df["underlying_price"],
            errors="coerce",
        )

    df["strike"] = pd.to_numeric(
        df["strike"],
        errors="coerce",
    )

    df["DTE"] = pd.to_numeric(
        df["DTE"],
        errors="coerce",
    )

    df["impliedVolatility"] = pd.to_numeric(
        df["impliedVolatility"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Calculate Greeks
    # --------------------------------------------------------

    log("CALCULATING DELTA / GAMMA / VEGA")

    greek_results = df.apply(
        calculate_greeks,
        axis=1,
        result_type="expand",
    )

    df[
        [
            "delta",
            "gamma",
            "vega",
            "delta_source",
            "gamma_source",
            "vega_source",
            "greeks_status",
        ]
    ] = greek_results[
        [
            "delta",
            "gamma",
            "vega",
            "delta_source",
            "gamma_source",
            "vega_source",
            "greeks_status",
        ]
    ]

    # --------------------------------------------------------
    # Validation statistics
    # --------------------------------------------------------

    valid_delta = df["delta"].notna().sum()
    valid_gamma = df["gamma"].notna().sum()
    valid_vega = df["vega"].notna().sum()

    invalid_iv = (
        df["impliedVolatility"].isna()
        | (df["impliedVolatility"] <= 0)
    ).sum()

    invalid_dte = (
        df["DTE"].isna()
        | (df["DTE"] <= 0)
    ).sum()

    invalid_price = (
        df["underlying_price"].isna()
        | (df["underlying_price"] <= 0)
    ).sum()

    invalid_strike = (
        df["strike"].isna()
        | (df["strike"] <= 0)
    ).sum()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    output_rows = len(df)

    # --------------------------------------------------------
    # Validation output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("🔎 STEP 4 VALIDATION")
    print("=" * 72)

    print(f"INPUT ROWS       : {input_rows:,}")
    print(f"OUTPUT ROWS      : {output_rows:,}")
    print()

    print(f"VALID DELTA      : {valid_delta:,}")
    print(f"VALID GAMMA      : {valid_gamma:,}")
    print(f"VALID VEGA       : {valid_vega:,}")
    print()

    print(f"INVALID IV       : {invalid_iv:,}")
    print(f"INVALID DTE      : {invalid_dte:,}")
    print(f"INVALID PRICE    : {invalid_price:,}")
    print(f"INVALID STRIKE   : {invalid_strike:,}")
    print()

    print("DELTA SOURCE     : CALCULATED")
    print("GAMMA SOURCE     : CALCULATED")
    print("VEGA SOURCE      : CALCULATED")
    print()

    if input_rows == output_rows:
        print("ROW COUNT CHECK  : OK")
    else:
        print("ROW COUNT CHECK  : ERROR")
        raise RuntimeError(
            "Input/output row count mismatch."
        )

    print(f"OUTPUT FILE      : {OUTPUT_FILE}")
    print("=" * 72)

    log("STEP 4 GREEKS COMPLETE")


if __name__ == "__main__":
    main()

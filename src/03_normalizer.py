import os
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# OPTION FLOW SCANNER V3
# STEP 3 - DATA NORMALIZATION
# ============================================================

INPUT_FILE = "data/raw/options_raw.csv"
OUTPUT_DIR = "data/normalized"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "options_normalized.csv"
)


# ============================================================
# LOG
# ============================================================

def log(message):

    now = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[03 NORMALIZER] {now} | {message}"
    )


# ============================================================
# DIRECTORY
# ============================================================

def prepare_directory():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    log(
        f"OUTPUT DIRECTORY READY : "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    log(
        f"LOADING : {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    log(
        f"ROWS LOADED : {len(df):,}"
    )

    log(
        f"COLUMNS FOUND : {len(df.columns)}"
    )

    return df


# ============================================================
# REQUIRED COLUMNS
# ============================================================

def validate_input_columns(df):

    required_columns = [

        "symbol",
        "option_type",
        "current_price",
        "strike",
        "expiration",
        "DTE",

        "bid",
        "ask",
        "lastPrice",

        "volume",
        "openInterest",

        "impliedVolatility",

        "collected_at",
    ]

    missing = [

        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    log(
        "REQUIRED COLUMNS : OK"
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_numeric_columns(df):

    numeric_columns = [

        "current_price",
        "strike",

        "bid",
        "ask",
        "lastPrice",

        "change",
        "percentChange",

        "volume",
        "openInterest",

        "impliedVolatility",

        "DTE",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    log(
        "NUMERIC CONVERSION : OK"
    )

    return df


# ============================================================
# OPTION TYPE NORMALIZATION
# ============================================================

def normalize_option_type(df):

    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    valid_types = [
        "CALL",
        "PUT"
    ]

    invalid = (
        ~df["option_type"].isin(
            valid_types
        )
    )

    invalid_count = int(
        invalid.sum()
    )

    if invalid_count > 0:

        log(
            f"REMOVING INVALID "
            f"OPTION TYPES : "
            f"{invalid_count}"
        )

        df = df.loc[
            ~invalid
        ].copy()

    log(
        "OPTION TYPE NORMALIZATION : OK"
    )

    return df


# ============================================================
# EXPIRATION NORMALIZATION
# ============================================================

def normalize_expiration(df):

    df["expiration"] = pd.to_datetime(
        df["expiration"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "expiration"
        ]
    ).copy()

    removed = before - len(df)

    if removed > 0:

        log(
            f"REMOVED INVALID "
            f"EXPIRATIONS : {removed}"
        )

    df["expiration"] = (
        df["expiration"]
        .dt.strftime("%Y-%m-%d")
    )

    log(
        "EXPIRATION NORMALIZATION : OK"
    )

    return df


# ============================================================
# DTE NORMALIZATION
# ============================================================

def normalize_dte(df):

    df["DTE"] = pd.to_numeric(
        df["DTE"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(
        subset=[
            "DTE"
        ]
    ).copy()

    removed = before - len(df)

    if removed > 0:

        log(
            f"REMOVED INVALID DTE : "
            f"{removed}"
        )

    df["DTE"] = (
        df["DTE"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Safety filter
    # --------------------------------------------------------

    df = df.loc[
        (df["DTE"] >= 0)
        &
        (df["DTE"] <= 180)
    ].copy()

    log(
        f"DTE RANGE AFTER NORMALIZATION : "
        f"{df['DTE'].min()} ~ "
        f"{df['DTE'].max()}"
    )

    return df


# ============================================================
# MID PRICE
# ============================================================

def calculate_mid_price(df):

    bid = df["bid"]
    ask = df["ask"]

    # --------------------------------------------------------
    # Valid bid / ask
    # --------------------------------------------------------

    valid_bid_ask = (
        bid.notna()
        &
        ask.notna()
        &
        (bid >= 0)
        &
        (ask >= 0)
        &
        (ask >= bid)
    )

    df["mid_price"] = np.where(

        valid_bid_ask,

        (bid + ask) / 2,

        np.nan
    )

    # --------------------------------------------------------
    # Fallback to last price
    # --------------------------------------------------------

    fallback = (
        df["mid_price"].isna()
        &
        df["lastPrice"].notna()
        &
        (df["lastPrice"] >= 0)
    )

    df.loc[
        fallback,
        "mid_price"
    ] = df.loc[
        fallback,
        "lastPrice"
    ]

    log(
        "MID PRICE : CALCULATED"
    )

    return df


# ============================================================
# BID / ASK SPREAD
# ============================================================

def calculate_spread(df):

    df["bid_ask_spread"] = (
        df["ask"]
        -
        df["bid"]
    )

    # --------------------------------------------------------
    # Relative spread
    # --------------------------------------------------------

    df["relative_spread"] = np.where(

        df["mid_price"] > 0,

        df["bid_ask_spread"]
        /
        df["mid_price"],

        np.nan
    )

    log(
        "BID/ASK SPREAD : CALCULATED"
    )

    return df


# ============================================================
# VOLUME / OI
# ============================================================

def calculate_volume_oi(df):

    df["volume_oi_ratio"] = np.where(

        df["openInterest"] > 0,

        df["volume"]
        /
        df["openInterest"],

        np.nan
    )

    log(
        "VOLUME/OI : CALCULATED"
    )

    return df


# ============================================================
# MONEINESS
# ============================================================

def calculate_moneyness(df):

    df["moneyness"] = np.where(

        df["current_price"] > 0,

        df["strike"]
        /
        df["current_price"],

        np.nan
    )

    df["distance_pct"] = np.where(

        df["current_price"] > 0,

        (
            df["strike"]
            -
            df["current_price"]
        )
        /
        df["current_price"]
        *
        100,

        np.nan
    )

    log(
        "MONEYNESS : CALCULATED"
    )

    return df


# ============================================================
# PREMIUM ESTIMATE
# ============================================================

def calculate_premium(df):

    # --------------------------------------------------------
    # Contract multiplier
    # --------------------------------------------------------

    df["contract_multiplier"] = 100

    # --------------------------------------------------------
    # Estimated traded premium
    #
    # Volume × Mid Price × 100
    # --------------------------------------------------------

    df["estimated_traded_premium"] = (

        df["volume"].fillna(0)

        *
        
        df["mid_price"].fillna(0)

        *

        df["contract_multiplier"]
    )

    log(
        "ESTIMATED TRADED PREMIUM : CALCULATED"
    )

    return df


# ============================================================
# DATA QUALITY FLAGS
# ============================================================

def create_quality_flags(df):

    # --------------------------------------------------------
    # Price source
    # --------------------------------------------------------

    df["price_source"] = np.where(

        (
            df["bid"].notna()
            &
            df["ask"].notna()
            &
            (df["ask"] >= df["bid"])
        ),

        "REAL",

        np.where(

            df["lastPrice"].notna(),

            "REAL",

            "UNAVAILABLE"
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_source"] = np.where(

        df["volume"].notna(),

        "REAL",

        "UNAVAILABLE"
    )

    # --------------------------------------------------------
    # Open Interest
    # --------------------------------------------------------

    df["oi_source"] = np.where(

        df["openInterest"].notna(),

        "REAL",

        "UNAVAILABLE"
    )

    # --------------------------------------------------------
    # IV
    # --------------------------------------------------------

    df["iv_source"] = np.where(

        df["impliedVolatility"].notna(),

        "REAL",

        "UNAVAILABLE"
    )

    # --------------------------------------------------------
    # Mid price
    # --------------------------------------------------------

    df["mid_price_source"] = np.where(

        (
            df["bid"].notna()
            &
            df["ask"].notna()
            &
            (df["ask"] >= df["bid"])
        ),

        "CALCULATED",

        np.where(

            df["lastPrice"].notna(),

            "ESTIMATED",

            "UNAVAILABLE"
        )
    )

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    df["premium_source"] = "CALCULATED"

    log(
        "DATA QUALITY FLAGS : CREATED"
    )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    key_columns = [

        "symbol",
        "option_type",
        "strike",
        "expiration",
    ]

    before = len(df)

    df = df.drop_duplicates(
        subset=key_columns,
        keep="last"
    ).copy()

    removed = before - len(df)

    log(
        f"DUPLICATES REMOVED : "
        f"{removed:,}"
    )

    return df


# ============================================================
# FINAL SORT
# ============================================================

def sort_data(df):

    df = df.sort_values(

        by=[
            "symbol",
            "expiration",
            "option_type",
            "strike",
        ]

    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# SAVE
# ============================================================

def save_data(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    log(
        f"NORMALIZED FILE SAVED : "
        f"{OUTPUT_FILE}"
    )

    log(
        f"ROWS SAVED : "
        f"{len(df):,}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_output(df):

    print()
    print("=" * 72)
    print("🔎 STEP 3 VALIDATION")
    print("=" * 72)

    print(
        f"ROWS              : "
        f"{len(df):,}"
    )

    print(
        f"TICKERS           : "
        f"{df['symbol'].nunique()}"
    )

    print(
        f"EXPIRATIONS       : "
        f"{df['expiration'].nunique()}"
    )

    print(
        f"DTE MIN           : "
        f"{df['DTE'].min()}"
    )

    print(
        f"DTE MAX           : "
        f"{df['DTE'].max()}"
    )

    print(
        f"CALL ROWS         : "
        f"{(df['option_type'] == 'CALL').sum():,}"
    )

    print(
        f"PUT ROWS          : "
        f"{(df['option_type'] == 'PUT').sum():,}"
    )

    print(
        f"MID PRICE         : "
        f"{df['mid_price'].notna().sum():,}"
    )

    print(
        f"PREMIUM ESTIMATE  : "
        f"{df['estimated_traded_premium'].notna().sum():,}"
    )

    # --------------------------------------------------------
    # Required normalized columns
    # --------------------------------------------------------

    required = [

        "symbol",
        "option_type",
        "current_price",
        "strike",
        "expiration",
        "DTE",

        "bid",
        "ask",
        "lastPrice",

        "volume",
        "openInterest",
        "impliedVolatility",

        "mid_price",
        "bid_ask_spread",
        "relative_spread",

        "volume_oi_ratio",

        "moneyness",
        "distance_pct",

        "contract_multiplier",
        "estimated_traded_premium",

        "price_source",
        "volume_source",
        "oi_source",
        "iv_source",

        "mid_price_source",
        "premium_source",

        "collected_at",
    ]

    missing = [

        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            f"❌ MISSING COLUMNS : "
            f"{missing}"
        )

        raise RuntimeError(
            "STEP 3 validation failed."
        )

    print(
        "REQUIRED COLUMNS   : OK"
    )

    # --------------------------------------------------------
    # DTE
    # --------------------------------------------------------

    if df["DTE"].min() < 0:

        raise RuntimeError(
            "Invalid negative DTE found."
        )

    if df["DTE"].max() > 180:

        raise RuntimeError(
            "DTE greater than 180 found."
        )

    print(
        "DTE RANGE          : OK"
    )

    # --------------------------------------------------------
    # Option types
    # --------------------------------------------------------

    option_types = set(
        df["option_type"]
        .dropna()
        .unique()
    )

    if not {
        "CALL",
        "PUT"
    }.issubset(option_types):

        raise RuntimeError(
            "CALL / PUT validation failed."
        )

    print(
        "CALL / PUT         : OK"
    )

    # --------------------------------------------------------
    # Numeric calculations
    # --------------------------------------------------------

    calculated_columns = [

        "mid_price",
        "bid_ask_spread",
        "volume_oi_ratio",
        "moneyness",
        "distance_pct",
        "estimated_traded_premium",
    ]

    for column in calculated_columns:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing calculated column: "
                f"{column}"
            )

    print(
        "CALCULATED FIELDS  : OK"
    )

    print("=" * 72)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    prepare_directory()

    df = load_data()

    validate_input_columns(
        df
    )

    df = convert_numeric_columns(
        df
    )

    df = normalize_option_type(
        df
    )

    df = normalize_expiration(
        df
    )

    df = normalize_dte(
        df
    )

    df = calculate_mid_price(
        df
    )

    df = calculate_spread(
        df
    )

    df = calculate_volume_oi(
        df
    )

    df = calculate_moneyness(
        df
    )

    df = calculate_premium(
        df
    )

    df = create_quality_flags(
        df
    )

    df = remove_duplicates(
        df
    )

    df = sort_data(
        df
    )

    validate_output(
        df
    )

    save_data(
        df
    )

    log(
        "STEP 3 NORMALIZATION COMPLETE"
    )


if __name__ == "__main__":

    main()

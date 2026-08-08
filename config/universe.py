# ============================================================
# OPTION FLOW SCANNER V3
# STOCK / ETF UNIVERSE
#
# STEP 1 : Market Regime
# STEP 2+ : Option Universe
# ============================================================

import io
import pandas as pd
import requests


# ============================================================
# STEP 1 MARKET UNIVERSE
# ============================================================

MARKET_UNIVERSE = {
    "NDX": "^NDX",
    "SPY": "SPY",
    "SOXX": "SOXX",
    "DIA": "DIA",
}


# ============================================================
# MAJOR ETF UNIVERSE
# ============================================================

MAJOR_ETFS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "SOXX",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLC",
    "XLU",
    "XLB",
    "ARKK",
    "SMH",
    "SOXL",
    "SOXS",
    "TQQQ",
    "SQQQ",
    "TLT",
    "HYG",
    "LQD",
    "GLD",
    "SLV",
    "USO",
    "UNG",
    "XBI",
    "XOP",
    "KRE",
]


# ============================================================
# HTTP SETTINGS
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131.0 Safari/537.36"
    )
}


# ============================================================
# LOAD S&P 500
# ============================================================

def load_sp500():

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(response.text)
    )

    if not tables:
        raise RuntimeError(
            "S&P 500 table was not found."
        )

    df = tables[0]

    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    return symbols


# ============================================================
# LOAD NASDAQ 100
# ============================================================

def load_nasdaq100():

    url = (
        "https://en.wikipedia.org/wiki/"
        "Nasdaq-100"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(response.text)
    )

    symbols = []

    for table in tables:

        for column in [
            "Ticker",
            "Ticker symbol",
            "Symbol",
        ]:

            if column in table.columns:

                values = (
                    table[column]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                symbols.extend(values)

                break

    if not symbols:
        raise RuntimeError(
            "Nasdaq 100 ticker table "
            "was not found."
        )

    return symbols


# ============================================================
# SYMBOL CLEANING
# ============================================================

def clean_symbol(symbol):

    symbol = str(symbol).strip().upper()

    # Yahoo Finance uses '-' instead of '.'
    # for symbols such as BRK.B / BF.B.
    symbol = symbol.replace(".", "-")

    return symbol


# ============================================================
# BUILD OPTION UNIVERSE
# ============================================================

def build_option_universe():

    print(
        "[UNIVERSE] Loading S&P 500..."
    )

    sp500 = load_sp500()

    print(
        f"[UNIVERSE] S&P 500 : "
        f"{len(sp500)}"
    )

    print(
        "[UNIVERSE] Loading Nasdaq 100..."
    )

    nasdaq100 = load_nasdaq100()

    print(
        f"[UNIVERSE] Nasdaq 100 : "
        f"{len(nasdaq100)}"
    )

    universe = []

    universe.extend(sp500)
    universe.extend(nasdaq100)
    universe.extend(MAJOR_ETFS)

    cleaned = []

    for symbol in universe:

        symbol = clean_symbol(symbol)

        if symbol:
            cleaned.append(symbol)

    # Remove duplicates while preserving order
    unique_symbols = list(
        dict.fromkeys(cleaned)
    )

    print(
        f"[UNIVERSE] UNIQUE OPTION TICKERS : "
        f"{len(unique_symbols)}"
    )

    return unique_symbols


# ============================================================
# DEFAULT OPTION UNIVERSE
# ============================================================

DEFAULT_OPTION_UNIVERSE = (
    build_option_universe()
)

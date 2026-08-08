# ============================================================
# OPTION FLOW SCANNER V3
# STOCK / ETF UNIVERSE
#
# STEP 1 : MARKET REGIME
# STEP 2+ : OPTION UNIVERSE
# ============================================================

import io
import requests
import pandas as pd


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
# NASDAQ-100 OPTION UNIVERSE
#
# Current validated public universe.
#
# Nasdaq-100 contains 100 companies but can contain more than
# 100 securities/tickers because some companies have multiple
# listed securities.
#
# Therefore validation below does NOT require exactly 100.
# ============================================================

NASDAQ_100_UNIVERSE = [

    "NVDA",
    "AAPL",
    "GOOGL",
    "GOOG",
    "MSFT",
    "AMZN",
    "AVGO",
    "SPCX",
    "META",
    "TSLA",
    "MU",
    "WMT",
    "AMD",
    "ASML",
    "INTC",
    "CSCO",
    "AMAT",
    "COST",
    "PLTR",
    "LRCX",
    "NFLX",
    "ARM",
    "PANW",
    "TXN",
    "KLAC",
    "LIN",
    "AMGN",
    "CRWD",
    "SHOP",
    "MRVL",
    "TMUS",
    "ADI",
    "PEP",
    "STX",
    "SNDK",
    "QCOM",
    "GILD",
    "BKNG",
    "WDC",
    "ISRG",
    "PDD",
    "VRTX",
    "SBUX",
    "FTNT",
    "APP",
    "ADP",
    "ADBE",
    "ABNB",
    "CEG",
    "DASH",
    "CDNS",
    "CSX",
    "MELI",
    "MAR",
    "CMCSA",
    "INTU",
    "MNST",
    "DDOG",
    "ROST",
    "CTAS",
    "MDLZ",
    "SNPS",
    "REGN",
    "HON",
    "ORLY",
    "PCAR",
    "LITE",
    "MPWR",
    "AEP",
    "WBD",
    "BKR",
    "NXPI",
    "FAST",
    "TER",
    "ALAB",
    "HONA",
    "FANG",
    "ADSK",
    "PYPL",
    "RKLB",
    "CRWV",
    "XEL",
    "NBIS",
    "FER",
    "CCEP",
    "EXC",
    "AXON",
    "IDXX",
    "TTWO",
    "MCHP",
    "ODFL",
    "WDAY",
    "TRI",
    "PAYX",
    "KDP",
    "ROP",
    "MSTR",
    "GEHC",
    "DXCM",
    "KHC",
    "ALNY",
    "CPRT",
]


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
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    )
}


# ============================================================
# SYMBOL CLEANING
# ============================================================

def clean_symbol(symbol):

    if symbol is None:
        return ""

    symbol = str(
        symbol
    ).strip().upper()

    if symbol in {
        "",
        "NAN",
        "NONE",
        "N/A",
        "NA",
    }:
        return ""

    # Yahoo Finance convention
    # BRK.B -> BRK-B
    symbol = symbol.replace(
        ".",
        "-"
    )

    return symbol


# ============================================================
# LOAD S&P 500
#
# This part is retained because the previous GitHub Actions
# execution already proved that the S&P 500 table is readable.
# ============================================================

def load_sp500():

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    print(
        "[UNIVERSE] Requesting S&P 500..."
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.StringIO(
            response.text
        )
    )

    if not tables:

        raise RuntimeError(
            "S&P 500 tables were not found."
        )

    selected_table = None
    selected_column = None

    for table in tables:

        for column in table.columns:

            column_name = str(
                column
            ).strip().lower()

            if column_name == "symbol":

                selected_table = table
                selected_column = column

                break

        if selected_table is not None:
            break

    if selected_table is None:

        raise RuntimeError(
            "S&P 500 Symbol column "
            "was not found."
        )

    symbols = []

    for value in selected_table[
        selected_column
    ]:

        symbol = clean_symbol(
            value
        )

        if symbol:

            symbols.append(
                symbol
            )

    symbols = list(
        dict.fromkeys(
            symbols
        )
    )

    if len(symbols) < 450:

        raise RuntimeError(
            "S&P 500 extraction returned "
            f"too few symbols: {len(symbols)}"
        )

    return symbols


# ============================================================
# BUILD OPTION UNIVERSE
# ============================================================

def build_option_universe():

    # --------------------------------------------------------
    # S&P 500
    # --------------------------------------------------------

    print(
        "[UNIVERSE] Loading S&P 500..."
    )

    sp500 = load_sp500()

    print(
        "[UNIVERSE] S&P 500 : "
        f"{len(sp500)}"
    )

    # --------------------------------------------------------
    # NASDAQ-100
    # --------------------------------------------------------

    nasdaq100 = [
        clean_symbol(symbol)
        for symbol in NASDAQ_100_UNIVERSE
        if clean_symbol(symbol)
    ]

    print(
        "[UNIVERSE] NASDAQ-100 : "
        f"{len(nasdaq100)}"
    )

    # --------------------------------------------------------
    # MAJOR ETFs
    # --------------------------------------------------------

    etfs = [
        clean_symbol(symbol)
        for symbol in MAJOR_ETFS
        if clean_symbol(symbol)
    ]

    print(
        "[UNIVERSE] MAJOR ETFs : "
        f"{len(etfs)}"
    )

    # --------------------------------------------------------
    # Validate Nasdaq-100
    #
    # Do not require exactly 100 because multiple securities
    # can represent companies in the index.
    # --------------------------------------------------------

    if len(nasdaq100) < 100:

        raise RuntimeError(
            "NASDAQ-100 universe is "
            "unexpectedly small: "
            f"{len(nasdaq100)}"
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = []

    combined.extend(
        sp500
    )

    combined.extend(
        nasdaq100
    )

    combined.extend(
        etfs
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_symbols = list(
        dict.fromkeys(
            combined
        )
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(unique_symbols) < 500:

        raise RuntimeError(
            "Final option universe is "
            "unexpectedly small: "
            f"{len(unique_symbols)}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    overlap_sp500_ndx = len(
        set(sp500)
        &
        set(nasdaq100)
    )

    print(
        "[UNIVERSE] S&P 500 + "
        "NASDAQ-100 + ETFs"
    )

    print(
        "[UNIVERSE] S&P 500 TICKERS : "
        f"{len(sp500)}"
    )

    print(
        "[UNIVERSE] NASDAQ-100 TICKERS : "
        f"{len(nasdaq100)}"
    )

    print(
        "[UNIVERSE] ETF TICKERS : "
        f"{len(etfs)}"
    )

    print(
        "[UNIVERSE] S&P 500 / "
        "NASDAQ-100 OVERLAP : "
        f"{overlap_sp500_ndx}"
    )

    print(
        "[UNIVERSE] UNIQUE OPTION "
        "TICKERS : "
        f"{len(unique_symbols)}"
    )

    return unique_symbols


# ============================================================
# DEFAULT OPTION UNIVERSE
# ============================================================

DEFAULT_OPTION_UNIVERSE = (
    build_option_universe()
)

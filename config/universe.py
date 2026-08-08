# ============================================================
# OPTION FLOW SCANNER V3
# STOCK / ETF UNIVERSE
#
# STEP 1 : MARKET REGIME
# STEP 2+ : OPTION UNIVERSE
# ============================================================

import io
import re

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
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": (
        "en-US,en;q=0.9"
    ),
}


# ============================================================
# CLEAN SYMBOL
# ============================================================

def clean_symbol(symbol):

    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if symbol in {
        "",
        "NAN",
        "NONE",
        "N/A",
        "NA",
        "TICKER",
        "SYMBOL",
    }:
        return ""

    # Remove Wikipedia-style footnotes
    symbol = re.sub(
        r"\[[^\]]*\]",
        "",
        symbol,
    )

    symbol = symbol.strip()

    # Yahoo Finance convention
    # BRK.B -> BRK-B
    symbol = symbol.replace(
        ".",
        "-",
    )

    symbol = symbol.replace(
        " ",
        "",
    )

    return symbol


# ============================================================
# VALID TICKER
# ============================================================

def is_valid_ticker(symbol):

    if not symbol:
        return False

    return bool(
        re.fullmatch(
            r"[A-Z0-9]{1,5}(?:-[A-Z0-9]{1,2})?",
            symbol,
        )
    )


# ============================================================
# LOAD S&P 500
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

        if is_valid_ticker(
            symbol
        ):

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
# LOAD NASDAQ 100
#
# Official Nasdaq page
# ============================================================

def load_nasdaq100():

    url = (
        "https://www.nasdaq.com/"
        "solutions/nasdaq-100/companies"
    )

    print(
        "[UNIVERSE] Requesting "
        "Nasdaq-100 official page..."
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=45,
    )

    response.raise_for_status()

    html = response.text

    print(
        "[UNIVERSE] Nasdaq page size : "
        f"{len(html):,} bytes"
    )

    if len(html) < 10000:

        raise RuntimeError(
            "Nasdaq page response is "
            "unexpectedly small."
        )

    symbols = []

    # --------------------------------------------------------
    # Method 1
    #
    # Parse HTML tables if the page exposes
    # the company table directly.
    # --------------------------------------------------------

    try:

        tables = pd.read_html(
            io.StringIO(html)
        )

    except Exception:

        tables = []

    print(
        "[UNIVERSE] Nasdaq HTML tables : "
        f"{len(tables)}"
    )

    for table_index, table in enumerate(
        tables
    ):

        print(
            "[UNIVERSE] NASDAQ TABLE "
            f"{table_index} COLUMNS : "
            f"{list(table.columns)}"
        )

        for column in table.columns:

            column_name = str(
                column
            ).strip().lower()

            if column_name not in {
                "symbol",
                "ticker",
                "ticker symbol",
            }:
                continue

            for value in table[
                column
            ]:

                symbol = clean_symbol(
                    value
                )

                if is_valid_ticker(
                    symbol
                ):

                    symbols.append(
                        symbol
                    )

    # --------------------------------------------------------
    # Method 2
    #
    # Nasdaq pages can embed the component
    # data in JSON / JavaScript.
    #
    # Search for quoted ticker-like values.
    # --------------------------------------------------------

    if len(
        list(
            dict.fromkeys(symbols)
        )
    ) < 80:

        print(
            "[UNIVERSE] HTML table extraction "
            "was insufficient."
        )

        # Look for common JSON fields.
        patterns = [
            r'"symbol"\s*:\s*"([A-Z][A-Z0-9.-]{0,5})"',
            r'"ticker"\s*:\s*"([A-Z][A-Z0-9.-]{0,5})"',
            r'"Symbol"\s*:\s*"([A-Z][A-Z0-9.-]{0,5})"',
            r'"Ticker"\s*:\s*"([A-Z][A-Z0-9.-]{0,5})"',
        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                html,
            )

            for value in matches:

                symbol = clean_symbol(
                    value
                )

                if is_valid_ticker(
                    symbol
                ):

                    symbols.append(
                        symbol
                    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    symbols = list(
        dict.fromkeys(
            symbols
        )
    )

    # --------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    #
    # Never allow a partial/random extraction
    # to silently become the Nasdaq-100 universe.
    # --------------------------------------------------------

    if len(symbols) < 80:

        raise RuntimeError(
            "Nasdaq-100 extraction failed. "
            "Only "
            f"{len(symbols)} valid symbols "
            "were found."
        )

    if len(symbols) > 130:

        raise RuntimeError(
            "Nasdaq-100 extraction returned "
            "an unexpectedly large universe: "
            f"{len(symbols)} symbols."
        )

    print(
        "[UNIVERSE] Nasdaq 100 : "
        f"{len(symbols)}"
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
    # Nasdaq 100
    # --------------------------------------------------------

    print(
        "[UNIVERSE] Loading Nasdaq 100..."
    )

    nasdaq100 = load_nasdaq100()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    universe = []

    universe.extend(
        sp500
    )

    universe.extend(
        nasdaq100
    )

    universe.extend(
        MAJOR_ETFS
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    cleaned = []

    for symbol in universe:

        symbol = clean_symbol(
            symbol
        )

        if is_valid_ticker(
            symbol
        ):

            cleaned.append(
                symbol
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_symbols = list(
        dict.fromkeys(
            cleaned
        )
    )

    print(
        "[UNIVERSE] S&P 500 + "
        "Nasdaq 100 + ETF"
    )

    print(
        "[UNIVERSE] UNIQUE OPTION "
        "TICKERS : "
        f"{len(unique_symbols)}"
    )

    if len(unique_symbols) < 500:

        raise RuntimeError(
            "Final option universe is "
            "unexpectedly small: "
            f"{len(unique_symbols)}"
        )

    return unique_symbols


# ============================================================
# DEFAULT OPTION UNIVERSE
# ============================================================

DEFAULT_OPTION_UNIVERSE = (
    build_option_universe()
)

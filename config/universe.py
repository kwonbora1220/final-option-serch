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

    # Wikipedia can contain footnote markers.
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

    # Remove spaces
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

    # Normal US equity / ETF ticker.
    #
    # Allow:
    # AAPL
    # BRK-B
    # BF-B
    # etc.
    #
    # Reject long text such as company names.

    if not re.fullmatch(
        r"[A-Z0-9]{1,5}(?:-[A-Z0-9]{1,2})?",
        symbol,
    ):
        return False

    return True


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

    # --------------------------------------------------------
    # Find table containing Symbol column
    # --------------------------------------------------------

    selected_table = None
    selected_column = None

    for table in tables:

        columns = [
            str(column)
            .strip()
            .lower()
            for column in table.columns
        ]

        for index, column in enumerate(
            columns
        ):

            if column == "symbol":

                selected_table = table
                selected_column = (
                    table.columns[index]
                )

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

        if is_valid_ticker(symbol):

            symbols.append(
                symbol
            )

    symbols = list(
        dict.fromkeys(symbols)
    )

    if not symbols:

        raise RuntimeError(
            "S&P 500 ticker list is empty."
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

    print(
        "[UNIVERSE] Requesting Nasdaq 100..."
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
            "Nasdaq 100 tables were not found."
        )

    symbols = []

    # --------------------------------------------------------
    # Examine every table.
    #
    # Wikipedia has changed table layouts over time.
    # Do not depend on one exact column name.
    # --------------------------------------------------------

    for table_index, table in enumerate(
        tables
    ):

        print(
            "[UNIVERSE] NASDAQ TABLE "
            f"{table_index} COLUMNS : "
            f"{list(table.columns)}"
        )

        # ----------------------------------------------------
        # Flatten MultiIndex columns if necessary.
        # ----------------------------------------------------

        column_map = {}

        for column in table.columns:

            if isinstance(
                column,
                tuple,
            ):

                parts = [
                    str(part).strip()
                    for part in column
                ]

                column_name = " ".join(
                    part
                    for part in parts
                    if part
                    and part.lower()
                    != "nan"
                )

            else:

                column_name = str(
                    column
                ).strip()

            column_map[
                column_name.lower()
            ] = column

        # ----------------------------------------------------
        # Candidate ticker columns.
        # ----------------------------------------------------

        candidate_names = [
            "ticker",
            "ticker symbol",
            "symbol",
            "ticker/symbol",
            "ticker symbol (nasdaq)",
        ]

        selected_column = None

        for candidate in candidate_names:

            if candidate in column_map:

                selected_column = (
                    column_map[candidate]
                )

                break

        # ----------------------------------------------------
        # If exact name not found, inspect
        # every column for ticker-like content.
        # ----------------------------------------------------

        if selected_column is None:

            best_column = None
            best_score = 0

            for column in table.columns:

                values = table[column]

                score = 0
                checked = 0

                for value in values.head(
                    min(len(values), 150)
                ):

                    symbol = clean_symbol(
                        value
                    )

                    if is_valid_ticker(
                        symbol
                    ):

                        score += 1

                    checked += 1

                if checked > 0:

                    ratio = (
                        score / checked
                    )

                    if (
                        score >= 10
                        and ratio >= 0.40
                        and score > best_score
                    ):

                        best_column = column
                        best_score = score

            selected_column = best_column

        if selected_column is None:
            continue

        print(
            "[UNIVERSE] NASDAQ TICKER "
            "COLUMN FOUND : "
            f"{selected_column}"
        )

        # ----------------------------------------------------
        # Extract ticker values.
        # ----------------------------------------------------

        for value in table[
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

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    symbols = list(
        dict.fromkeys(symbols)
    )

    # --------------------------------------------------------
    # Safety check
    #
    # We expect a Nasdaq-100 table to contain
    # approximately 100 constituents.
    # If we accidentally extracted a random
    # column, do not silently continue.
    # --------------------------------------------------------

    if len(symbols) < 80:

        raise RuntimeError(
            "Nasdaq 100 ticker extraction "
            "returned too few symbols: "
            f"{len(symbols)}"
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

    if not unique_symbols:

        raise RuntimeError(
            "Option universe is empty."
        )

    return unique_symbols


# ============================================================
# DEFAULT OPTION UNIVERSE
# ============================================================

DEFAULT_OPTION_UNIVERSE = (
    build_option_universe()
)

# ============================================================
# OPTION FLOW SCANNER V3
# STOCK / ETF UNIVERSE
#
# STEP 1 : MARKET REGIME
# STEP 2+ : OPTION UNIVERSE
# ============================================================


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
# Maintained as a static validated universe so that
# GitHub Actions does not depend on web-page HTML parsing.
#
# This list contains the current public Nasdaq-100 stock
# universe used by the scanner.
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
# SYMBOL CLEANING
# ============================================================

def clean_symbol(symbol):

    if symbol is None:
        return ""

    symbol = str(
        symbol
    ).strip().upper()

    # Yahoo Finance convention
    # BRK.B -> BRK-B
    symbol = symbol.replace(
        ".",
        "-"
    )

    return symbol


# ============================================================
# BUILD OPTION UNIVERSE
# ============================================================

def build_option_universe():

    # --------------------------------------------------------
    # NASDAQ-100
    # --------------------------------------------------------

    nasdaq100 = [
        clean_symbol(symbol)
        for symbol in NASDAQ_100_UNIVERSE
        if clean_symbol(symbol)
    ]

    # --------------------------------------------------------
    # MAJOR ETFs
    # --------------------------------------------------------

    etfs = [
        clean_symbol(symbol)
        for symbol in MAJOR_ETFS
        if clean_symbol(symbol)
    ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = []

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
    # Validation
    # --------------------------------------------------------

    if len(nasdaq100) < 90:

        raise RuntimeError(
            "NASDAQ-100 universe is "
            "unexpectedly small: "
            f"{len(nasdaq100)}"
        )

    if len(unique_symbols) < 100:

        raise RuntimeError(
            "Final option universe is "
            "unexpectedly small: "
            f"{len(unique_symbols)}"
        )

    print(
        "[UNIVERSE] NASDAQ-100 TICKERS : "
        f"{len(nasdaq100)}"
    )

    print(
        "[UNIVERSE] MAJOR ETF TICKERS : "
        f"{len(etfs)}"
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

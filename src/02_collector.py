import os
import time
from datetime import datetime, date

import pandas as pd
import yfinance as yf


# ============================================================
# OPTION FLOW SCANNER V3
# STEP 2 - OPTION DATA COLLECTOR
# ============================================================

TEST_UNIVERSE = [
    "SPY",
    "QQQ",
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "AMD",
    "META",
    "GOOG",
    "TSLA",
]

MIN_DTE = 0
MAX_DTE = 180

OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "options_raw.csv"
)


# ============================================================
# LOG
# ============================================================

def log(message):
    now = datetime.utcnow().strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    print(
        f"[02 COLLECTOR] {now} | {message}"
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
# DTE
# ============================================================

def calculate_dte(expiration):

    today = date.today()

    expiration_date = datetime.strptime(
        expiration,
        "%Y-%m-%d"
    ).date()

    return (
        expiration_date - today
    ).days


# ============================================================
# COLLECT ONE TICKER
# ============================================================

def collect_ticker(ticker):

    log(
        f"{ticker} START"
    )

    try:

        stock = yf.Ticker(ticker)

        # ----------------------------------------------------
        # Current price
        # ----------------------------------------------------

        history = stock.history(
            period="5d",
            auto_adjust=False
        )

        if history.empty:

            log(
                f"{ticker} FAILED - "
                f"NO PRICE DATA"
            )

            return []

        current_price = float(
            history["Close"].iloc[-1]
        )

        log(
            f"{ticker} PRICE : "
            f"${current_price:.2f}"
        )

        # ----------------------------------------------------
        # Expirations
        # ----------------------------------------------------

        expirations = stock.options

        if not expirations:

            log(
                f"{ticker} FAILED - "
                f"NO EXPIRATIONS"
            )

            return []

        log(
            f"{ticker} TOTAL EXPIRATIONS : "
            f"{len(expirations)}"
        )

        rows = []

        expiration_count = 0

        call_count = 0
        put_count = 0

        min_dte_found = None
        max_dte_found = None

        # ----------------------------------------------------
        # Process expirations
        # ----------------------------------------------------

        for expiration in expirations:

            try:

                dte = calculate_dte(
                    expiration
                )

            except Exception:

                log(
                    f"{ticker} INVALID "
                    f"EXPIRATION : "
                    f"{expiration}"
                )

                continue

            # ------------------------------------------------
            # DTE filter
            # ------------------------------------------------

            if dte < MIN_DTE:
                continue

            if dte > MAX_DTE:
                continue

            expiration_count += 1

            if (
                min_dte_found is None
                or dte < min_dte_found
            ):
                min_dte_found = dte

            if (
                max_dte_found is None
                or dte > max_dte_found
            ):
                max_dte_found = dte

            log(
                f"{ticker} EXPIRATION "
                f"{expiration} | "
                f"DTE {dte}"
            )

            # ------------------------------------------------
            # Option chain
            # ------------------------------------------------

            try:

                chain = stock.option_chain(
                    expiration
                )

            except Exception as e:

                log(
                    f"{ticker} CHAIN FAILED "
                    f"{expiration} | "
                    f"{type(e).__name__}: {e}"
                )

                continue

            # =================================================
            # CALLS
            # =================================================

            calls = chain.calls.copy()

            if not calls.empty:

                calls["option_type"] = "CALL"

                call_count += len(calls)

                for _, row in calls.iterrows():

                    rows.append(
                        build_row(
                            ticker=ticker,
                            current_price=current_price,
                            expiration=expiration,
                            dte=dte,
                            option_type="CALL",
                            row=row
                        )
                    )

            # =================================================
            # PUTS
            # =================================================

            puts = chain.puts.copy()

            if not puts.empty:

                puts["option_type"] = "PUT"

                put_count += len(puts)

                for _, row in puts.iterrows():

                    rows.append(
                        build_row(
                            ticker=ticker,
                            current_price=current_price,
                            expiration=expiration,
                            dte=dte,
                            option_type="PUT",
                            row=row
                        )
                    )

            # ------------------------------------------------
            # Small pause
            # ------------------------------------------------

            time.sleep(0.2)

        # =====================================================
        # SUMMARY
        # =====================================================

        log(
            f"{ticker} COMPLETE | "
            f"EXPIRATIONS={expiration_count} | "
            f"CALLS={call_count} | "
            f"PUTS={put_count}"
        )

        if min_dte_found is not None:

            log(
                f"{ticker} DTE RANGE : "
                f"{min_dte_found} ~ "
                f"{max_dte_found}"
            )

        else:

            log(
                f"{ticker} DTE RANGE : "
                f"NO DATA"
            )

        return rows

    except Exception as e:

        log(
            f"{ticker} FAILED - "
            f"{type(e).__name__}: {e}"
        )

        return []


# ============================================================
# BUILD ROW
# ============================================================

def build_row(
    ticker,
    current_price,
    expiration,
    dte,
    option_type,
    row
):

    strike = safe_float(
        row.get("strike")
    )

    bid = safe_float(
        row.get("bid")
    )

    ask = safe_float(
        row.get("ask")
    )

    last_price = safe_float(
        row.get("lastPrice")
    )

    volume = safe_int(
        row.get("volume")
    )

    open_interest = safe_int(
        row.get("openInterest")
    )

    implied_volatility = safe_float(
        row.get(
            "impliedVolatility"
        )
    )

    change = safe_float(
        row.get("change")
    )

    percent_change = safe_float(
        row.get("percentChange")
    )

    in_the_money = row.get(
        "inTheMoney"
    )

    contract_size = row.get(
        "contractSize"
    )

    currency = row.get(
        "currency"
    )

    return {

        "symbol": ticker,

        "option_type": option_type,

        "current_price": current_price,

        "strike": strike,

        "expiration": expiration,

        "DTE": dte,

        "bid": bid,

        "ask": ask,

        "lastPrice": last_price,

        "change": change,

        "percentChange": percent_change,

        "volume": volume,

        "openInterest": open_interest,

        "impliedVolatility":
            implied_volatility,

        "inTheMoney":
            in_the_money,

        "contractSize":
            contract_size,

        "currency":
            currency,

        "collected_at":
            datetime.utcnow().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
    }


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(value):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        return float(value)

    except Exception:

        return None


def safe_int(value):

    if value is None:
        return 0

    try:

        if pd.isna(value):
            return 0

        return int(value)

    except Exception:

        return 0


# ============================================================
# SAVE
# ============================================================

def save_data(rows):

    if not rows:

        log(
            "NO OPTION DATA TO SAVE"
        )

        return

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "symbol",
            "expiration",
            "option_type",
            "strike",
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    log(
        f"RAW FILE SAVED : "
        f"{OUTPUT_FILE}"
    )

    log(
        f"TOTAL ROWS : "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("🔥 OPTION COLLECTION SUMMARY")
    print("=" * 72)

    print(
        f"TOTAL ROWS       : "
        f"{len(df):,}"
    )

    print(
        f"TICKERS          : "
        f"{df['symbol'].nunique()}"
    )

    print(
        f"EXPIRATIONS      : "
        f"{df['expiration'].nunique()}"
    )

    print(
        f"CALL CONTRACTS   : "
        f"{(df['option_type'] == 'CALL').sum():,}"
    )

    print(
        f"PUT CONTRACTS    : "
        f"{(df['option_type'] == 'PUT').sum():,}"
    )

    print(
        f"DTE MIN          : "
        f"{df['DTE'].min()}"
    )

    print(
        f"DTE MAX          : "
        f"{df['DTE'].max()}"
    )

    print(
        f"DATE MIN         : "
        f"{df['expiration'].min()}"
    )

    print(
        f"DATE MAX         : "
        f"{df['expiration'].max()}"
    )

    print("=" * 72)


# ============================================================
# VALIDATION
# ============================================================

def validate_data(rows):

    if not rows:

        log(
            "VALIDATION FAILED - "
            "NO ROWS"
        )

        return False

    df = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("🔎 STEP 2 VALIDATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Ticker count
    # --------------------------------------------------------

    ticker_count = df[
        "symbol"
    ].nunique()

    print(
        f"TICKERS FOUND : "
        f"{ticker_count}"
    )

    # --------------------------------------------------------
    # DTE
    # --------------------------------------------------------

    min_dte = df["DTE"].min()
    max_dte = df["DTE"].max()

    print(
        f"DTE RANGE     : "
        f"{min_dte} ~ {max_dte}"
    )

    # --------------------------------------------------------
    # Option types
    # --------------------------------------------------------

    call_count = (
        df["option_type"] == "CALL"
    ).sum()

    put_count = (
        df["option_type"] == "PUT"
    ).sum()

    print(
        f"CALL ROWS     : "
        f"{call_count:,}"
    )

    print(
        f"PUT ROWS      : "
        f"{put_count:,}"
    )

    # --------------------------------------------------------
    # Required columns
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
        "collected_at",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print(
            f"MISSING COLUMNS : "
            f"{missing}"
        )

        return False

    print(
        "REQUIRED COLUMNS : OK"
    )

    # --------------------------------------------------------
    # DTE validation
    # --------------------------------------------------------

    if min_dte < MIN_DTE:

        print(
            "❌ DTE BELOW MINIMUM"
        )

        return False

    if max_dte > MAX_DTE:

        print(
            "❌ DTE ABOVE MAXIMUM"
        )

        return False

    # --------------------------------------------------------
    # Call / Put validation
    # --------------------------------------------------------

    if call_count == 0:

        print(
            "❌ NO CALL DATA"
        )

        return False

    if put_count == 0:

        print(
            "❌ NO PUT DATA"
        )

        return False

    print(
        "DTE VALIDATION : OK"
    )

    print(
        "CALL / PUT VALIDATION : OK"
    )

    print("=" * 72)

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    prepare_directory()

    all_rows = []

    for ticker in TEST_UNIVERSE:

        rows = collect_ticker(
            ticker
        )

        all_rows.extend(rows)

    log(
        f"ALL TICKERS COMPLETE | "
        f"ROWS={len(all_rows):,}"
    )

    if not all_rows:

        log(
            "FATAL - "
            "NO OPTION DATA COLLECTED"
        )

        raise RuntimeError(
            "No option data collected."
        )

    valid = validate_data(
        all_rows
    )

    if not valid:

        raise RuntimeError(
            "STEP 2 validation failed."
        )

    save_data(
        all_rows
    )

    log(
        "STEP 2 COLLECTION COMPLETE"
    )


if __name__ == "__main__":
    main()

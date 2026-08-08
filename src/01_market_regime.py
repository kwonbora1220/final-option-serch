import math
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ============================================================
# OPTION FLOW SCANNER V3
# STEP 1 - MARKET REGIME ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv"
)


TICKERS = {
    "NDX": "^NDX",
    "SPY": "SPY",
    "SOXX": "SOXX",
    "DIA": "DIA",
}

PERIOD = "6mo"
INTERVAL = "1d"


# ============================================================
# LOG
# ============================================================

def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[01 MARKET] {now} | {message}"
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        math.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# TREND
# ============================================================

def calculate_trend(df):

    close = df["Close"]

    sma20 = close.rolling(20).mean()

    sma50 = close.rolling(50).mean()

    latest = float(
        close.iloc[-1]
    )

    latest_sma20 = float(
        sma20.iloc[-1]
    )

    latest_sma50 = float(
        sma50.iloc[-1]
    )

    if (
        pd.isna(latest_sma20)
        or pd.isna(latest_sma50)
    ):
        return "UNAVAILABLE"

    if (
        latest
        > latest_sma20
        > latest_sma50
    ):
        return "BULLISH"

    if (
        latest
        < latest_sma20
        < latest_sma50
    ):
        return "BEARISH"

    return "MIXED"


# ============================================================
# MOMENTUM
# ============================================================

def calculate_momentum(df):

    close = df["Close"]

    if len(close) < 21:
        return None

    current = float(
        close.iloc[-1]
    )

    previous = float(
        close.iloc[-21]
    )

    return (
        (current / previous) - 1
    ) * 100


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_levels(df):

    recent = df.tail(20)

    support = float(
        recent["Low"].min()
    )

    resistance = float(
        recent["High"].max()
    )

    return support, resistance


# ============================================================
# DIRECTION
# ============================================================

def determine_direction(result):

    if result is None:
        return "UNAVAILABLE"

    score = 0

    # --------------------------------------------------------
    # DAILY PRICE
    # --------------------------------------------------------

    if result["daily_change_pct"] > 0:
        score += 1

    elif result["daily_change_pct"] < 0:
        score -= 1

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if result["trend"] == "BULLISH":
        score += 2

    elif result["trend"] == "BEARISH":
        score -= 2

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = result["rsi"]

    if rsi is not None:

        if rsi >= 55:
            score += 1

        elif rsi <= 45:
            score -= 1

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if score >= 2:
        return "BULLISH"

    if score <= -2:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# SINGLE MARKET ANALYSIS
# ============================================================

def analyze_ticker(name, ticker):

    log(
        f"{name} DATA REQUEST"
    )

    try:

        df = yf.download(
            ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:

            log(
                f"{name} FAILED - NO DATA"
            )

            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required_columns:

            if column not in df.columns:

                log(
                    f"{name} FAILED - "
                    f"MISSING {column}"
                )

                return None

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 50:

            log(
                f"{name} FAILED - "
                f"NOT ENOUGH DATA"
            )

            return None

        close = df["Close"]

        current_price = float(
            close.iloc[-1]
        )

        previous_close = float(
            close.iloc[-2]
        )

        daily_change = (
            current_price
            - previous_close
        )

        daily_change_pct = (
            daily_change
            / previous_close
        ) * 100

        day_high = float(
            df["High"].iloc[-1]
        )

        day_low = float(
            df["Low"].iloc[-1]
        )

        volume = int(
            df["Volume"].iloc[-1]
        )

        rsi_series = calculate_rsi(
            close
        )

        rsi = rsi_series.iloc[-1]

        if pd.isna(rsi):
            rsi_value = None
        else:
            rsi_value = float(rsi)

        trend = calculate_trend(
            df
        )

        momentum = calculate_momentum(
            df
        )

        support, resistance = (
            calculate_levels(df)
        )

        result = {

            "name": name,

            "ticker": ticker,

            "price": current_price,

            "daily_change": daily_change,

            "daily_change_pct":
                daily_change_pct,

            "high": day_high,

            "low": day_low,

            "volume": volume,

            "rsi": rsi_value,

            "trend": trend,

            "momentum_pct_20d":
                momentum,

            "support": support,

            "resistance": resistance,
        }

        result["direction"] = (
            determine_direction(
                result
            )
        )

        log(
            f"{name} OK | "
            f"{result['direction']}"
        )

        return result

    except Exception as e:

        log(
            f"{name} FAILED - "
            f"{type(e).__name__}: {e}"
        )

        return None


# ============================================================
# MARKET SCORE
# ============================================================

def calculate_market_score(results):

    weights = {

        "NDX": 1.25,

        "SPY": 1.25,

        "SOXX": 1.00,

        "DIA": 0.75,
    }

    direction_scores = {

        "BULLISH": 1,

        "NEUTRAL": 0,

        "BEARISH": -1,

        "UNAVAILABLE": 0,
    }

    weighted_total = 0.0

    total_weight = 0.0

    for name, result in results.items():

        if result is None:
            continue

        direction = result[
            "direction"
        ]

        weight = weights[name]

        weighted_total += (
            direction_scores[
                direction
            ] * weight
        )

        total_weight += weight

    if total_weight == 0:

        return None

    raw_score = (
        weighted_total
        / total_weight
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Raw score : -1 ~ +1
    #
    # Final score : 0 ~ 100
    #
    # -1.0 -> 0
    # -0.5 -> 25
    #  0.0 -> 50
    # +0.5 -> 75
    # +1.0 -> 100
    # --------------------------------------------------------

    score_100 = (
        (raw_score + 1)
        * 50
    )

    score_100 = max(
        0,
        min(
            100,
            score_100
        )
    )

    return score_100


# ============================================================
# REGIME
# ============================================================

def determine_regime(score):

    if score is None:
        return "NO DATA"

    if score >= 70:
        return "RISK-ON"

    if score >= 55:
        return "MILD RISK-ON"

    if score >= 45:
        return "NEUTRAL"

    if score >= 30:
        return "RISK-OFF WARNING"

    return "RISK-OFF"


# ============================================================
# DIVERGENCE
# ============================================================

def detect_divergence(results):

    ndx = results.get("NDX")

    spy = results.get("SPY")

    soxx = results.get("SOXX")

    dia = results.get("DIA")

    messages = []

    ndx_direction = (
        ndx["direction"]
        if ndx
        else "UNAVAILABLE"
    )

    spy_direction = (
        spy["direction"]
        if spy
        else "UNAVAILABLE"
    )

    soxx_direction = (
        soxx["direction"]
        if soxx
        else "UNAVAILABLE"
    )

    dia_direction = (
        dia["direction"]
        if dia
        else "UNAVAILABLE"
    )

    if (
        ndx_direction == "BULLISH"
        and spy_direction == "BEARISH"
    ):
        messages.append(
            "Technology strong / "
            "Broad market weak"
        )

    if (
        ndx_direction == "BEARISH"
        and spy_direction == "BULLISH"
    ):
        messages.append(
            "Technology weak / "
            "Broad market strong"
        )

    if (
        soxx_direction == "BULLISH"
        and ndx_direction != "BULLISH"
    ):
        messages.append(
            "Semiconductors stronger "
            "than Nasdaq"
        )

    if (
        soxx_direction == "BEARISH"
        and ndx_direction != "BEARISH"
    ):
        messages.append(
            "Semiconductors weaker "
            "than Nasdaq"
        )

    if (
        dia_direction == "BULLISH"
        and ndx_direction == "BEARISH"
    ):
        messages.append(
            "Traditional sectors strong / "
            "Technology weak"
        )

    if (
        dia_direction == "BEARISH"
        and ndx_direction == "BULLISH"
    ):
        messages.append(
            "Technology strong / "
            "Traditional sectors weak"
        )

    return messages


# ============================================================
# SAVE MARKET REGIME
# ============================================================

def save_market_regime(
    results,
    market_score,
    regime
):

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True
    )

    divergence = detect_divergence(
        results
    )

    rows = []

    for name in [
        "NDX",
        "SPY",
        "SOXX",
        "DIA"
    ]:

        result = results.get(name)

        if result is None:

            rows.append({

                "ticker": name,

                "direction":
                    "UNAVAILABLE",

                "price": None,

                "daily_change_pct":
                    None,

                "rsi": None,

                "momentum_pct_20d":
                    None,

                "trend":
                    "UNAVAILABLE",

                "support": None,

                "resistance": None,

                "market_score":
                    market_score,

                "market_regime":
                    regime,

                "divergence":
                    " | ".join(
                        divergence
                    ),

                "data_source":
                    "CALCULATED",
            })

            continue

        rows.append({

            "ticker": name,

            "direction":
                result["direction"],

            "price":
                result["price"],

            "daily_change_pct":
                result["daily_change_pct"],

            "rsi":
                result["rsi"],

            "momentum_pct_20d":
                result[
                    "momentum_pct_20d"
                ],

            "trend":
                result["trend"],

            "support":
                result["support"],

            "resistance":
                result["resistance"],

            "market_score":
                market_score,

            "market_regime":
                regime,

            "divergence":
                " | ".join(
                    divergence
                ),

            "data_source":
                "CALCULATED",
        })

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    log(
        f"MARKET REGIME SAVED : "
        f"{OUTPUT_FILE}"
    )

    return output


# ============================================================
# REPORT
# ============================================================

def print_report(
    results,
    market_score,
    regime
):

    print()

    print("=" * 72)

    print(
        "🔥 MARKET REGIME ANALYSIS"
    )

    print("=" * 72)

    print(
        "UTC TIME : "
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    print()

    for name in [
        "NDX",
        "SPY",
        "SOXX",
        "DIA"
    ]:

        result = results.get(name)

        print()

        print("-" * 72)

        print(
            f"📈 {name}"
        )

        print("-" * 72)

        if result is None:

            print(
                "STATUS : FAILED"
            )

            continue

        print(
            f"PRICE          : "
            f"${result['price']:.2f}"
        )

        print(
            f"DAILY CHANGE   : "
            f"{result['daily_change_pct']:+.2f}%"
        )

        print(
            f"RSI(14)        : "
            f"{result['rsi']:.2f}"
            if result["rsi"] is not None
            else "RSI(14)        : UNAVAILABLE"
        )

        print(
            f"MOMENTUM(20D)  : "
            f"{result['momentum_pct_20d']:+.2f}%"
            if result["momentum_pct_20d"]
            is not None
            else
            "MOMENTUM(20D)  : UNAVAILABLE"
        )

        print(
            f"TREND          : "
            f"{result['trend']}"
        )

        print(
            f"DIRECTION      : "
            f"{result['direction']}"
        )

        print(
            f"SUPPORT        : "
            f"${result['support']:.2f}"
        )

        print(
            f"RESISTANCE     : "
            f"${result['resistance']:.2f}"
        )

    print()

    print("=" * 72)

    print(
        "🔥 MARKET REGIME SCORE"
    )

    print("=" * 72)

    if market_score is None:

        print(
            "SCORE  : UNAVAILABLE"
        )

        print(
            "REGIME : NO DATA"
        )

    else:

        print(
            f"SCORE  : "
            f"{market_score:.1f} / 100"
        )

        print(
            f"REGIME : "
            f"{regime}"
        )

    divergence = detect_divergence(
        results
    )

    print()

    print("=" * 72)

    print(
        "⚠️ MARKET DIVERGENCE"
    )

    print("=" * 72)

    if divergence:

        for item in divergence:
            print(
                f"⚠️ {item}"
            )

    else:

        print(
            "No major divergence detected."
        )

    print()

    print("=" * 72)

    print(
        "📌 DATA RELIABILITY"
    )

    print("=" * 72)

    print(
        "Price / OHLC / Volume : REAL"
    )

    print(
        "RSI                   : CALCULATED"
    )

    print(
        "Trend                 : CALCULATED"
    )

    print(
        "Momentum              : CALCULATED"
    )

    print(
        "Support / Resistance  : CALCULATED"
    )

    print(
        "Market Score          : CALCULATED 0-100"
    )

    print(
        "Market Regime         : CALCULATED"
    )

    print(
        "Option data           : STEP 2"
    )

    print(
        "Greeks                : STEP 4"
    )

    print(
        "GEX                   : STEP 4"
    )

    print(
        "Call Wall / Put Wall  : STEP 8"
    )

    print(
        "HIRO                  : UNAVAILABLE"
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    results = {}

    for name, ticker in TICKERS.items():

        results[name] = analyze_ticker(
            name,
            ticker
        )

    successful = sum(
        1
        for value in results.values()
        if value is not None
    )

    log(
        f"DATA CHECK "
        f"{successful}/"
        f"{len(TICKERS)} SUCCESS"
    )

    if successful == 0:

        raise RuntimeError(
            "All market datasets failed"
        )

    if successful != len(TICKERS):

        log(
            "WARNING - "
            "ONE OR MORE MARKET DATASETS FAILED"
        )

    market_score = (
        calculate_market_score(
            results
        )
    )

    regime = determine_regime(
        market_score
    )

    print_report(
        results,
        market_score,
        regime
    )

    output = save_market_regime(
        results,
        market_score,
        regime
    )

    print()

    print("=" * 72)

    print(
        "🔎 STEP 1 VALIDATION"
    )

    print("=" * 72)

    print(
        f"MARKET ROWS       : "
        f"{len(output)}"
    )

    print(
        f"MARKET SCORE      : "
        f"{market_score:.2f}"
        if market_score is not None
        else
        "MARKET SCORE      : UNAVAILABLE"
    )

    print(
        f"MARKET REGIME     : "
        f"{regime}"
    )

    print(
        f"OUTPUT FILE       : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 72)

    log(
        "STEP 1 REGIME COMPLETE"
    )


if __name__ == "__main__":
    main()

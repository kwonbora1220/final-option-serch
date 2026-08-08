import math
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

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
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[01 MARKET] {now} | {message}")


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

    rs = avg_gain / avg_loss.replace(0, math.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# TREND
# ============================================================

def calculate_trend(df):
    close = df["Close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    latest = close.iloc[-1]
    latest_sma20 = sma20.iloc[-1]
    latest_sma50 = sma50.iloc[-1]

    if pd.isna(latest_sma20) or pd.isna(latest_sma50):
        return "UNAVAILABLE"

    if latest > latest_sma20 > latest_sma50:
        return "BULLISH"

    if latest < latest_sma20 < latest_sma50:
        return "BEARISH"

    return "MIXED"


# ============================================================
# MOMENTUM
# ============================================================

def calculate_momentum(df):
    close = df["Close"]

    if len(close) < 21:
        return None

    current = float(close.iloc[-1])
    previous = float(close.iloc[-21])

    return ((current / previous) - 1) * 100


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_levels(df):
    recent = df.tail(20)

    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())

    return support, resistance


# ============================================================
# SINGLE MARKET ANALYSIS
# ============================================================

def analyze_ticker(name, ticker):
    log(f"{name} DATA REQUEST")

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
            log(f"{name} FAILED - NO DATA")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required_columns:
            if column not in df.columns:
                log(f"{name} FAILED - MISSING {column}")
                return None

        df = df.dropna(subset=["Close"])

        if len(df) < 50:
            log(f"{name} FAILED - NOT ENOUGH DATA")
            return None

        close = df["Close"]

        current_price = float(close.iloc[-1])
        previous_close = float(close.iloc[-2])

        daily_change = current_price - previous_close
        daily_change_pct = (
            daily_change / previous_close
        ) * 100

        day_high = float(df["High"].iloc[-1])
        day_low = float(df["Low"].iloc[-1])
        volume = int(df["Volume"].iloc[-1])

        rsi = calculate_rsi(close).iloc[-1]

        trend = calculate_trend(df)

        momentum = calculate_momentum(df)

        support, resistance = calculate_levels(df)

        if pd.isna(rsi):
            rsi_value = None
        else:
            rsi_value = float(rsi)

        result = {
            "name": name,
            "ticker": ticker,
            "price": current_price,
            "daily_change": daily_change,
            "daily_change_pct": daily_change_pct,
            "high": day_high,
            "low": day_low,
            "volume": volume,
            "rsi": rsi_value,
            "trend": trend,
            "momentum_pct_20d": momentum,
            "support": support,
            "resistance": resistance,
        }

        log(f"{name} OK")

        return result

    except Exception as e:
        log(f"{name} FAILED - {type(e).__name__}: {e}")
        return None


# ============================================================
# MARKET ALIGNMENT
# ============================================================

def determine_direction(result):
    if result is None:
        return "UNAVAILABLE"

    score = 0

    if result["daily_change_pct"] > 0:
        score += 1
    elif result["daily_change_pct"] < 0:
        score -= 1

    if result["trend"] == "BULLISH":
        score += 2
    elif result["trend"] == "BEARISH":
        score -= 2

    if result["rsi"] is not None:
        if result["rsi"] >= 55:
            score += 1
        elif result["rsi"] <= 45:
            score -= 1

    if score >= 2:
        return "BULLISH"

    if score <= -2:
        return "BEARISH"

    return "NEUTRAL"


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

        direction = determine_direction(result)

        weight = weights.get(name, 1.0)

        weighted_total += (
            direction_scores[direction] * weight
        )

        total_weight += weight

    if total_weight == 0:
        return None

    normalized = weighted_total / total_weight

    return normalized


# ============================================================
# REGIME
# ============================================================

def determine_regime(score):
    if score is None:
        return "NO DATA"

    if score >= 0.70:
        return "RISK-ON"

    if score >= 0.30:
        return "MILD RISK-ON"

    if score > -0.30:
        return "NEUTRAL"

    if score > -0.70:
        return "RISK-OFF WARNING"

    return "RISK-OFF"


# ============================================================
# DIVERGENCE
# ============================================================

def detect_divergence(results):
    ndx = determine_direction(results.get("NDX"))
    spy = determine_direction(results.get("SPY"))
    soxx = determine_direction(results.get("SOXX"))
    dia = determine_direction(results.get("DIA"))

    messages = []

    if ndx == "BULLISH" and spy == "BEARISH":
        messages.append(
            "Technology strong / Broad market weak"
        )

    if ndx == "BEARISH" and spy == "BULLISH":
        messages.append(
            "Technology weak / Broad market strong"
        )

    if soxx == "BULLISH" and ndx != "BULLISH":
        messages.append(
            "Semiconductors stronger than Nasdaq"
        )

    if soxx == "BEARISH" and ndx != "BEARISH":
        messages.append(
            "Semiconductors weaker than Nasdaq"
        )

    if dia == "BULLISH" and ndx == "BEARISH":
        messages.append(
            "Traditional sectors strong / Technology weak"
        )

    if dia == "BEARISH" and ndx == "BULLISH":
        messages.append(
            "Technology strong / Traditional sectors weak"
        )

    return messages


# ============================================================
# REPORT
# ============================================================

def print_report(results, market_score, regime):
    print()
    print("=" * 70)
    print("🔥 MARKET REGIME ANALYSIS")
    print("=" * 70)

    print(
        f"DATE : "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print()

    for name in ["NDX", "SPY", "SOXX", "DIA"]:
        result = results.get(name)

        print("-" * 70)
        print(name)

        if result is None:
            print("STATUS : FAILED")
            continue

        direction = determine_direction(result)

        print(f"PRICE          : ${result['price']:.2f}")
        print(
            f"DAILY CHANGE   : "
            f"{result['daily_change_pct']:+.2f}%"
        )
        print(f"DAY HIGH       : ${result['high']:.2f}")
        print(f"DAY LOW        : ${result['low']:.2f}")
        print(f"VOLUME         : {result['volume']:,}")

        if result["rsi"] is not None:
            print(f"RSI(14)        : {result['rsi']:.2f}")
        else:
            print("RSI(14)        : UNAVAILABLE")

        print(
            f"MOMENTUM(20D)  : "
            f"{result['momentum_pct_20d']:+.2f}%"
        )

        print(f"TREND          : {result['trend']}")
        print(f"DIRECTION      : {direction}")

        print(
            f"SUPPORT        : "
            f"${result['support']:.2f}"
        )

        print(
            f"RESISTANCE     : "
            f"${result['resistance']:.2f}"
        )

    print()
    print("=" * 70)
    print("🔥 MARKET REGIME")
    print("=" * 70)

    if market_score is None:
        print("SCORE : UNAVAILABLE")
    else:
        print(
            f"SCORE : {market_score * 100:+.1f}"
        )

    print(f"REGIME : {regime}")

    print()
    print("📊 MARKET DIVERGENCE")
    print("-" * 70)

    divergence = detect_divergence(results)

    if divergence:
        for item in divergence:
            print(f"⚠️ {item}")
    else:
        print("No major divergence detected.")

    print()
    print("=" * 70)
    print("DATA SOURCE")
    print("=" * 70)
    print("Price / OHLC / Volume : REAL")
    print("RSI                   : CALCULATED")
    print("Trend                 : CALCULATED")
    print("Momentum              : CALCULATED")
    print("Support / Resistance  : CALCULATED")
    print("Options data          : STEP 2")
    print("GEX                   : STEP 4")
    print("Call Wall / Put Wall  : STEP 4")
    print("HIRO                  : UNAVAILABLE")
    print("=" * 70)


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
        1 for value in results.values()
        if value is not None
    )

    log(
        f"DATA CHECK "
        f"{successful}/{len(TICKERS)} SUCCESS"
    )

    market_score = calculate_market_score(
        results
    )

    regime = determine_regime(
        market_score
    )

    print_report(
        results,
        market_score,
        regime
    )

    log("REGIME COMPLETE")


if __name__ == "__main__":
    main()

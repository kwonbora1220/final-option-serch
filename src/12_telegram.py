
# ============================================================
# FINAL OPTION SEARCH
#
# STEP 12 - TELEGRAM REPORTER
#
# IMPORTANT
# ------------------------------------------------------------
# STEP 1 ~ STEP 11 결과를 변경하지 않는다.
# data/analysis/*.csv 만 읽어서 Telegram 보고서를 생성한다.
#
# DATA CLASSIFICATION
# ------------------------------------------------------------
# REAL
# CALCULATED
# ESTIMATED
# UNAVAILABLE
#
# TELEGRAM ORDER
# ------------------------------------------------------------
# 1. FINAL TRADING LIST
# 2. CALL BUY + PUT SELL
# 3. MARKET REGIME
# 4. MARKET REGIME SCORE
# 5. UNUSUAL OPTION FLOW
# 6. TOP 20 OPTION SEARCH
# 7. TOP STOCK DETAIL
# 8. DATA DISCLAIMER
# 9. SCAN COMPLETE
#
# Telegram limit:
# approximately 4096 characters
# automatically split into multiple messages
# ============================================================

import csv
import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis"
)


# ============================================================
# ANALYSIS FILES
# ============================================================

FILES = {
    "market": os.path.join(
        DATA_DIR,
        "market_regime.csv"
    ),
    "unusual": os.path.join(
        DATA_DIR,
        "unusual_flow.csv"
    ),
    "top20": os.path.join(
        DATA_DIR,
        "top20.csv"
    ),
    "option_search": os.path.join(
        DATA_DIR,
        "option_search.csv"
    ),
    "greeks": os.path.join(
        DATA_DIR,
        "options_greeks.csv"
    ),
    "structure": os.path.join(
        DATA_DIR,
        "structure.csv"
    ),
    "decision": os.path.join(
        DATA_DIR,
        "decision.csv"
    ),
    "final": os.path.join(
        DATA_DIR,
        "final_report.csv"
    ),
    "special": os.path.join(
        DATA_DIR,
        "special_list.csv"
    ),
}


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID"
)

TELEGRAM_LIMIT = 3900


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
        f"[12 TELEGRAM] {now} | {message}"
    )


# ============================================================
# CSV
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        log(
            "FILE NOT FOUND : "
            + os.path.relpath(
                path,
                BASE_DIR
            )
        )

        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

        log(
            "LOADED : "
            + os.path.relpath(
                path,
                BASE_DIR
            )
            + f" | ROWS={len(rows):,}"
        )

        return rows

    except Exception as exc:

        log(
            "CSV READ ERROR : "
            + str(exc)
        )

        return []


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_key(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace(".", "")
    )


def get_value(row, *names, default=""):

    if not row:
        return default

    normalized = {}

    for key, value in row.items():

        normalized[
            normalize_key(key)
        ] = value

    for name in names:

        key = normalize_key(name)

        if key not in normalized:
            continue

        value = normalized[key]

        if value is None:
            continue

        value = str(value).strip()

        if value:
            return value

    return default


def text_value(value):

    if value is None:
        return ""

    return str(value).strip()


def number_value(value):

    if value is None:
        return None

    try:

        text = (
            str(value)
            .strip()
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
        )

        if not text:
            return None

        return float(text)

    except Exception:

        return None


def fmt_number(value):

    number = number_value(value)

    if number is None:
        return text_value(value) or "N/A"

    if number.is_integer():

        return f"{int(number):,}"

    return f"{number:,.2f}"


def fmt_money(value):

    number = number_value(value)

    if number is None:
        return text_value(value) or "N/A"

    absolute = abs(number)

    if absolute >= 1_000_000_000:

        return (
            f"${number / 1_000_000_000:.2f}B"
        )

    if absolute >= 1_000_000:

        return (
            f"${number / 1_000_000:.2f}M"
        )

    if absolute >= 1_000:

        return (
            f"${number / 1_000:.1f}K"
        )

    return f"${number:,.0f}"


# ============================================================
# DATA CLASSIFICATION
# ============================================================

def REAL(value):

    value = text_value(value)

    return value if value else "N/A"


def CALCULATED(value):

    value = text_value(value)

    if not value:
        return "N/A"

    return f"{value} [CALCULATED]"


def ESTIMATED(value):

    value = text_value(value)

    if not value:
        return "N/A"

    return f"{value} [ESTIMATED]"


def UNAVAILABLE():

    return "UNAVAILABLE"


# ============================================================
# SYMBOL
# ============================================================

def symbol_of(row):

    return get_value(
        row,
        "symbol",
        "ticker",
        "underlying",
        "underlying_symbol",
        "stock",
        "name"
    ).upper()


# ============================================================
# SCORE
# ============================================================

def score_of(row):

    value = get_value(
        row,
        "score",
        "final_score",
        "decision_score",
        "flow_score",
        "unusual_flow_score",
        "option_search_score",
        "structure_score",
        default="0"
    )

    number = number_value(value)

    if number is None:
        return -999999

    return number


def sorted_rows(rows):

    return sorted(
        rows,
        key=score_of,
        reverse=True
    )


# ============================================================
# STATUS
# ============================================================

def status_emoji(value):

    text = text_value(value).upper()

    if any(
        word in text
        for word in (
            "ENTRY",
            "ENTER",
            "BUY",
            "LONG",
            "BULLISH",
            "진입"
        )
    ):

        return "🟢"

    if any(
        word in text
        for word in (
            "WATCH",
            "HOLD",
            "NEUTRAL",
            "관망"
        )
    ):

        return "🟡"

    if any(
        word in text
        for word in (
            "AVOID",
            "SELL",
            "BEARISH",
            "회피"
        )
    ):

        return "🔴"

    return "⚪"


# ============================================================
# FIND ROW
# ============================================================

def find_symbol(rows, symbol):

    symbol = symbol.upper()

    for row in rows:

        if symbol_of(row) == symbol:
            return row

    return None


# ============================================================
# MARKET REGIME
# ============================================================

def build_market_regime(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🌎 MARKET REGIME",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not rows:

        lines.append(
            "⚪ MARKET REGIME : "
            + UNAVAILABLE()
        )

        return lines

    preferred = [
        "NDX",
        "SPY",
        "SOXX",
        "DIA"
    ]

    ordered = []

    used = set()

    for ticker in preferred:

        for row in rows:

            if symbol_of(row) == ticker:

                ordered.append(row)
                used.add(id(row))
                break

    for row in rows:

        if id(row) not in used:

            ordered.append(row)

    for row in ordered:

        symbol = symbol_of(row)

        if not symbol:
            continue

        price = get_value(
            row,
            "current_price",
            "price",
            "close",
            "current"
        )

        change = get_value(
            row,
            "change_percent",
            "percent_change",
            "change_pct",
            "pct_change"
        )

        trend = get_value(
            row,
            "trend",
            "regime",
            "direction"
        )

        momentum = get_value(
            row,
            "momentum"
        )

        rsi = get_value(
            row,
            "rsi"
        )

        support = get_value(
            row,
            "support",
            "support_level"
        )

        resistance = get_value(
            row,
            "resistance",
            "resistance_level"
        )

        lines.append("")
        lines.append(
            f"📊 {symbol}"
        )

        if price:
            lines.append(
                f"현재가 {REAL(price)}"
            )

        if change:
            lines.append(
                f"등락 {REAL(change)}"
            )

        if trend:
            lines.append(
                f"Trend       {REAL(trend)}"
            )

        if momentum:
            lines.append(
                f"Momentum    {REAL(momentum)}"
            )

        if rsi:
            lines.append(
                f"RSI         {REAL(rsi)}"
            )

        if support:
            lines.append(
                f"Support     {CALCULATED(support)}"
            )

        if resistance:
            lines.append(
                f"Resistance  {CALCULATED(resistance)}"
            )

    return lines


# ============================================================
# MARKET SCORE
# ============================================================

def build_market_score(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 MARKET REGIME SCORE",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not rows:

        lines.append(
            "⚪ UNAVAILABLE"
        )

        return lines

    for row in rows:

        symbol = symbol_of(row)

        score = get_value(
            row,
            "score",
            "regime_score",
            "market_score"
        )

        if not symbol or not score:
            continue

        trend = get_value(
            row,
            "trend",
            "regime"
        )

        lines.append(
            f"{symbol:<8} "
            f"{score}/100 "
            f"{status_emoji(trend)}"
        )

    overall_score = ""

    overall_regime = ""

    for row in rows:

        if not overall_score:

            overall_score = get_value(
                row,
                "overall_score",
                "total_score"
            )

        if not overall_regime:

            overall_regime = get_value(
                row,
                "overall_regime",
                "market_regime",
                "regime"
            )

    if overall_score:

        lines.append("")
        lines.append("OVERALL")
        lines.append(
            f"🔥 {overall_score}/100"
        )

    if overall_regime:

        lines.append(
            f"{status_emoji(overall_regime)} "
            f"{overall_regime}"
        )

    return lines


# ============================================================
# UNUSUAL FLOW
# ============================================================

def build_unusual_flow(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 UNUSUAL OPTION FLOW",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    symbols = set()

    for row in rows:

        symbol = symbol_of(row)

        if symbol:
            symbols.add(symbol)

    lines.append(
        f"전체 분석 종목       {len(symbols):,}"
    )

    lines.append(
        f"옵션 Flow 분석       {len(rows):,}"
    )

    lines.append(
        "DTE                   0~180 ALL"
    )

    lines.append(
        "CALL / PUT            ALL"
    )

    lines.append("")
    lines.append(
        "오늘 비정상 Flow TOP"
    )

    for index, row in enumerate(
        sorted_rows(rows)[:20],
        start=1
    ):

        symbol = symbol_of(row)

        if not symbol:
            continue

        score = get_value(
            row,
            "score",
            "unusual_flow_score",
            "flow_score"
        )

        if index == 1:
            rank = "🥇"
        elif index == 2:
            rank = "🥈"
        elif index == 3:
            rank = "🥉"
        else:
            rank = f"{index}."

        lines.append(
            f"{rank} {symbol:<8} "
            f"{score or 'N/A'}"
        )

    return lines


# ============================================================
# TOP20
# ============================================================

def build_top20(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 TOP 20 OPTION SEARCH",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not rows:

        lines.append(
            "⚪ UNAVAILABLE"
        )

        return lines

    for index, row in enumerate(
        sorted_rows(rows)[:20],
        start=1
    ):

        symbol = symbol_of(row)

        if not symbol:
            continue

        score = get_value(
            row,
            "score",
            "option_search_score",
            "flow_score",
            "unusual_flow_score"
        )

        decision = get_value(
            row,
            "decision",
            "action",
            "status",
            "signal"
        )

        lines.append(
            f"{index}. {symbol} "
            f"{status_emoji(decision)} "
            f"Score {score or 'N/A'}"
        )

    return lines


# ============================================================
# FINAL TRADING LIST
# ============================================================

def build_final_list(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥🔥 FINAL TRADING LIST",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not rows:

        lines.append(
            "⚪ FINAL REPORT : "
            + UNAVAILABLE()
        )

        return lines

    for index, row in enumerate(
        sorted_rows(rows)[:20],
        start=1
    ):

        symbol = symbol_of(row)

        if not symbol:
            continue

        decision = get_value(
            row,
            "decision",
            "action",
            "status",
            "signal"
        )

        score = get_value(
            row,
            "score",
            "final_score",
            "decision_score",
            "structure_score"
        )

        reason = get_value(
            row,
            "reason",
            "reasons",
            "summary"
        )

        call = get_value(
            row,
            "call",
            "call_contract",
            "call_symbol"
        )

        put = get_value(
            row,
            "put",
            "put_contract",
            "put_symbol"
        )

        support = get_value(
            row,
            "support",
            "support_level"
        )

        resistance = get_value(
            row,
            "resistance",
            "resistance_level"
        )

        if index == 1:
            rank = "🥇"
        elif index == 2:
            rank = "🥈"
        elif index == 3:
            rank = "🥉"
        else:
            rank = f"{index}️⃣"

        lines.append("")
        lines.append(
            f"{rank} {symbol}"
        )

        lines.append(
            f"{status_emoji(decision)} "
            f"{decision or 'UNAVAILABLE'}"
        )

        if score:

            lines.append(
                f"Score {score}"
            )

        if reason:

            lines.append(
                "📌 이유"
            )

            for part in reason.split("|"):

                part = part.strip()

                if part:
                    lines.append(
                        f"• {part}"
                    )

        if call:

            lines.append(
                f"🎯 Call {call}"
            )

        if put:

            lines.append(
                f"🛡 Put {put}"
            )

        if support:

            lines.append(
                f"📐 Support "
                f"{CALCULATED(support)}"
            )

        if resistance:

            lines.append(
                f"📐 Resistance "
                f"{CALCULATED(resistance)}"
            )

    return lines


# ============================================================
# SPECIAL STRUCTURE
# ============================================================

def build_special(rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 CALL BUY + PUT SELL",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not rows:

        lines.append(
            "해당 구조 없음"
        )

        return lines

    for index, row in enumerate(
        sorted_rows(rows)[:20],
        start=1
    ):

        symbol = symbol_of(row)

        if not symbol:
            continue

        call = get_value(
            row,
            "call",
            "call_contract",
            "call_symbol",
            "call_strike"
        )

        put = get_value(
            row,
            "put",
            "put_contract",
            "put_symbol",
            "put_strike"
        )

        call_dte = get_value(
            row,
            "call_dte",
            "dte"
        )

        put_dte = get_value(
            row,
            "put_dte",
            "dte"
        )

        call_premium = get_value(
            row,
            "call_premium",
            "call_premium_flow"
        )

        put_premium = get_value(
            row,
            "put_premium",
            "put_premium_flow"
        )

        strength = get_value(
            row,
            "strength",
            "score",
            "structure_score"
        )

        current_price = get_value(
            row,
            "current_price",
            "price"
        )

        lines.append("")
        lines.append(
            f"{index}. {symbol}"
        )

        if strength:

            lines.append(
                f"Strength: {strength}"
            )

        lines.append("")
        lines.append(
            "CALL BUY EST. 🟢"
        )

        if call:
            lines.append(call)

        if call_dte:
            lines.append(
                f"DTE {call_dte}"
            )

        if call_premium:
            lines.append(
                "Premium Flow "
                + fmt_money(call_premium)
                + " [CALCULATED]"
            )

        lines.append("")
        lines.append(
            "PUT SELL EST. 🔴"
        )

        if put:
            lines.append(put)

        if put_dte:
            lines.append(
                f"DTE {put_dte}"
            )

        if put_premium:
            lines.append(
                "Premium Flow "
                + fmt_money(put_premium)
                + " [CALCULATED]"
            )

        lines.append("")
        lines.append(
            "Structure"
        )

        lines.append(
            "🔥 BULLISH RISK-REVERSAL"
        )

        if current_price:

            lines.append(
                "Current Price "
                + REAL(current_price)
            )

    return lines


# ============================================================
# TOP STOCK DETAIL
# ============================================================

def build_top_detail(
    top20_rows,
    option_rows,
    greek_rows,
    structure_rows,
    decision_rows
):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🥇 TOP 종목 상세 분석",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not top20_rows:

        lines.append(
            "⚪ TOP20 : "
            + UNAVAILABLE()
        )

        return lines

    top = sorted_rows(top20_rows)

    if not top:
        return lines

    base = top[0]

    symbol = symbol_of(base)

    if not symbol:
        return lines

    current_price = get_value(
        base,
        "current_price",
        "price"
    )

    score = get_value(
        base,
        "score",
        "option_search_score",
        "flow_score",
        "unusual_flow_score"
    )

    lines.append(
        f"🥇 {symbol}"
    )

    if current_price:

        lines.append(
            f"현재가 {REAL(current_price)}"
        )

    if score:

        lines.append(
            f"UNUSUAL FLOW SCORE {score}"
        )

    # --------------------------------------------------------
    # 선정 이유
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "📌 TOP20 선정 이유"
    )

    reason = get_value(
        base,
        "reason",
        "reasons",
        "selection_reason"
    )

    if reason:

        for part in reason.split("|"):

            part = part.strip()

            if part:
                lines.append(
                    f"• {part}"
                )

    else:

        lines.append(
            "⚪ 선정 이유 : "
            + UNAVAILABLE()
        )

    # --------------------------------------------------------
    # MARKET ALIGNMENT
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🌎 MARKET ALIGNMENT"
    )

    alignment = get_value(
        base,
        "market_alignment",
        "alignment"
    )

    if alignment:

        lines.append(alignment)

    else:

        lines.append(
            "⚪ MARKET ALIGNMENT : "
            + UNAVAILABLE()
        )

    # --------------------------------------------------------
    # OPTION STRUCTURE
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )
    lines.append(
        "🎯 OPTION STRUCTURE"
    )
    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    structure = find_symbol(
        structure_rows,
        symbol
    )

    if structure:

        call_wall = get_value(
            structure,
            "call_wall"
        )

        put_wall = get_value(
            structure,
            "put_wall"
        )

        support = get_value(
            structure,
            "support",
            "support_level"
        )

        resistance = get_value(
            structure,
            "resistance",
            "resistance_level"
        )

        if call_wall:
            lines.append(
                f"CALL WALL "
                f"{CALCULATED(call_wall)}"
            )

        if put_wall:
            lines.append(
                f"PUT WALL "
                f"{CALCULATED(put_wall)}"
            )

        if support:
            lines.append(
                f"주요 지지 "
                f"{CALCULATED(support)}"
            )

        if resistance:
            lines.append(
                f"주요 저항 "
                f"{CALCULATED(resistance)}"
            )

    else:

        lines.append(
            "⚪ STRUCTURE : "
            + UNAVAILABLE()
        )

    # --------------------------------------------------------
    # GREEKS / EXPOSURE
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "📊 GREEKS / EXPOSURE"
    )

    greek = find_symbol(
        greek_rows,
        symbol
    )

    if greek:

        iv = get_value(
            greek,
            "implied_volatility",
            "impliedVolatility",
            "iv"
        )

        delta = get_value(
            greek,
            "delta"
        )

        gamma = get_value(
            greek,
            "gamma"
        )

        vanna = get_value(
            greek,
            "vanna"
        )

        charm = get_value(
            greek,
            "charm"
        )

        gex = get_value(
            greek,
            "gex"
        )

        if iv:
            lines.append(
                f"IV          {REAL(iv)} [REAL]"
            )

        if delta:
            lines.append(
                f"Delta       {REAL(delta)} [REAL]"
            )

        if gamma:
            lines.append(
                f"Gamma       {CALCULATED(gamma)}"
            )

        if vanna:
            lines.append(
                f"Vanna       {CALCULATED(vanna)}"
            )

        if charm:
            lines.append(
                f"Charm       {CALCULATED(charm)}"
            )

        if gex:
            lines.append(
                f"GEX         {CALCULATED(gex)}"
            )

    else:

        lines.append(
            "Greeks : "
            + UNAVAILABLE()
        )

    lines.append(
        "HIRO        UNAVAILABLE"
    )

    # --------------------------------------------------------
    # CALL FLOW
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🟢 TOP CALL FLOW"
    )

    calls = []

    for row in option_rows:

        if symbol_of(row) != symbol:
            continue

        option_type = get_value(
            row,
            "option_type",
            "type",
            "contract_type"
        ).upper()

        if option_type == "CALL":

            calls.append(row)

    calls = sorted_rows(calls)

    for call in calls[:2]:

        strike = get_value(
            call,
            "strike"
        )

        dte = get_value(
            call,
            "dte"
        )

        volume = get_value(
            call,
            "volume"
        )

        oi = get_value(
            call,
            "open_interest",
            "openInterest",
            "oi"
        )

        iv = get_value(
            call,
            "implied_volatility",
            "impliedVolatility",
            "iv"
        )

        delta = get_value(
            call,
            "delta"
        )

        gamma = get_value(
            call,
            "gamma"
        )

        premium = get_value(
            call,
            "premium_flow",
            "premium"
        )

        lines.append("")
        lines.append(
            f"💚 ${strike or 'N/A'}C "
            f"| DTE {dte or 'N/A'}"
        )

        if volume or oi:

            lines.append(
                "Vol "
                + fmt_number(volume)
                + " | OI "
                + fmt_number(oi)
            )

        if iv:

            lines.append(
                f"IV {REAL(iv)} [REAL]"
            )

        if delta:

            lines.append(
                f"Delta {REAL(delta)} [REAL]"
            )

        if gamma:

            lines.append(
                f"Gamma {CALCULATED(gamma)}"
            )

        if premium:

            lines.append(
                "Premium Flow "
                + fmt_money(premium)
                + " [CALCULATED]"
            )

        lines.append(
            "BUY EST. 🟢 [ESTIMATED]"
        )

        lines.append(
            "OPEN EST. 🟢 [ESTIMATED]"
        )

    # --------------------------------------------------------
    # PUT FLOW
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🔴 TOP PUT FLOW"
    )

    puts = []

    for row in option_rows:

        if symbol_of(row) != symbol:
            continue

        option_type = get_value(
            row,
            "option_type",
            "type",
            "contract_type"
        ).upper()

        if option_type == "PUT":

            puts.append(row)

    puts = sorted_rows(puts)

    for put in puts[:2]:

        strike = get_value(
            put,
            "strike"
        )

        dte = get_value(
            put,
            "dte"
        )

        volume = get_value(
            put,
            "volume"
        )

        oi = get_value(
            put,
            "open_interest",
            "openInterest",
            "oi"
        )

        iv = get_value(
            put,
            "implied_volatility",
            "impliedVolatility",
            "iv"
        )

        delta = get_value(
            put,
            "delta"
        )

        gamma = get_value(
            put,
            "gamma"
        )

        premium = get_value(
            put,
            "premium_flow",
            "premium"
        )

        lines.append("")
        lines.append(
            f"❤️ ${strike or 'N/A'}P "
            f"| DTE {dte or 'N/A'}"
        )

        if volume or oi:

            lines.append(
                "Vol "
                + fmt_number(volume)
                + " | OI "
                + fmt_number(oi)
            )

        if iv:

            lines.append(
                f"IV {REAL(iv)} [REAL]"
            )

        if delta:

            lines.append(
                f"Delta {REAL(delta)} [REAL]"
            )

        if gamma:

            lines.append(
                f"Gamma {CALCULATED(gamma)}"
            )

        if premium:

            lines.append(
                "Premium Flow "
                + fmt_money(premium)
                + " [CALCULATED]"
            )

        lines.append(
            "SELL EST. 🔴 [ESTIMATED]"
        )

        lines.append(
            "OPEN EST. 🟢 [ESTIMATED]"
        )

    # --------------------------------------------------------
    # RISK REVERSAL
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🔥 SPECIAL STRUCTURE"
    )

    special = find_symbol(
        decision_rows,
        symbol
    )

    if special:

        structure_text = get_value(
            special,
            "structure",
            "special_structure",
            "signal"
        )

        if structure_text:

            lines.append(
                structure_text
            )

        else:

            lines.append(
                "CALL BUY EST. + PUT SELL EST."
            )

    else:

        lines.append(
            "CALL BUY EST. + PUT SELL EST."
        )

    lines.append(
        "※ BUY/SELL 방향은 ESTIMATED"
    )

    # --------------------------------------------------------
    # AI / DECISION
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )
    lines.append(
        "🧠 종합 판단"
    )
    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    decision = find_symbol(
        decision_rows,
        symbol
    )

    if decision:

        summary = get_value(
            decision,
            "summary",
            "analysis",
            "comment",
            "reason"
        )

        if summary:

            for part in summary.split("|"):

                part = part.strip()

                if part:
                    lines.append(part)

        decision_value = get_value(
            decision,
            "decision",
            "action",
            "status",
            "signal"
        )

        decision_score = get_value(
            decision,
            "score",
            "decision_score",
            "final_score"
        )

        if decision_value:

            lines.append("")
            lines.append(
                f"{status_emoji(decision_value)} "
                f"{decision_value}"
            )

        if decision_score:

            lines.append(
                f"Score {decision_score}/100 "
                "[CALCULATED]"
            )

    else:

        lines.append(
            "Decision : "
            + UNAVAILABLE()
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🎯 FINAL DECISION"
    )

    if decision:

        decision_value = get_value(
            decision,
            "decision",
            "action",
            "status",
            "signal"
        )

        decision_score = get_value(
            decision,
            "score",
            "decision_score",
            "final_score"
        )

        if decision_value:

            lines.append(
                f"{status_emoji(decision_value)} "
                f"{decision_value}"
            )

        if decision_score:

            lines.append(
                f"Score {decision_score}/100 "
                "[CALCULATED]"
            )

    else:

        lines.append(
            "⚪ "
            + UNAVAILABLE()
        )

    return lines


# ============================================================
# DISCLAIMER
# ============================================================

def build_disclaimer():

    return [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ DATA DISCLAIMER",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "무료 데이터 기반 분석입니다.",
        "",
        "🟢 REAL",
        "실제 제공 데이터",
        "",
        "🔵 CALCULATED",
        "시장 데이터로 자체 계산",
        "",
        "🟡 ESTIMATED",
        "BUY / SELL / OPEN / CLOSE 등",
        "무료 데이터 기반 간접 추정",
        "",
        "⚪ UNAVAILABLE",
        "무료 데이터에서 확인 불가",
        "",
        "실제 옵션 체결 방향 및",
        "기관 포지션을 확정적으로",
        "의미하지 않습니다.",
    ]


# ============================================================
# SCAN COMPLETE
# ============================================================

def build_complete(
    final_rows,
    special_rows,
    market_rows
):

    entries = 0
    watches = 0
    avoids = 0

    for row in final_rows:

        decision = get_value(
            row,
            "decision",
            "action",
            "status",
            "signal"
        ).upper()

        if any(
            word in decision
            for word in (
                "ENTRY",
                "ENTER",
                "BUY",
                "LONG",
                "진입"
            )
        ):

            entries += 1

        elif any(
            word in decision
            for word in (
                "WATCH",
                "HOLD",
                "NEUTRAL",
                "관망"
            )
        ):

            watches += 1

        elif any(
            word in decision
            for word in (
                "AVOID",
                "SELL",
                "회피"
            )
        ):

            avoids += 1

    market_status = (
        "CHECKED"
        if market_rows
        else UNAVAILABLE()
    )

    return [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🏁 SCAN COMPLETE",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "Market Regime",
        market_status,
        "",
        "TOP20",
        f"{min(20, len(final_rows))} 종목",
        "",
        "Final Entry",
        f"{entries} 종목",
        "",
        "Watch",
        f"{watches} 종목",
        "",
        "Avoid",
        f"{avoids} 종목",
        "",
        "Bullish Risk-Reversal",
        f"{len(special_rows)} 종목",
        "",
        "⏰ Next Scan",
        "다음 미국장 종료 후",
        "한국시간 오전 6:00",
    ]


# ============================================================
# REPORT
# ============================================================

def build_report():

    log(
        "LOADING data/analysis RESULTS"
    )

    market_rows = read_csv(
        FILES["market"]
    )

    unusual_rows = read_csv(
        FILES["unusual"]
    )

    top20_rows = read_csv(
        FILES["top20"]
    )

    option_rows = read_csv(
        FILES["option_search"]
    )

    greek_rows = read_csv(
        FILES["greeks"]
    )

    structure_rows = read_csv(
        FILES["structure"]
    )

    decision_rows = read_csv(
        FILES["decision"]
    )

    final_rows = read_csv(
        FILES["final"]
    )

    special_rows = read_csv(
        FILES["special"]
    )

    report = []

    # 1
    report.extend(
        build_final_list(
            final_rows
        )
    )

    # 2
    report.extend(
        build_special(
            special_rows
        )
    )

    # 3
    report.extend(
        build_market_regime(
            market_rows
        )
    )

    # 4
    report.extend(
        build_market_score(
            market_rows
        )
    )

    # 5
    report.extend(
        build_unusual_flow(
            unusual_rows
        )
    )

    # 6
    report.extend(
        build_top20(
            top20_rows
        )
    )

    # 7
    report.extend(
        build_top_detail(
            top20_rows,
            option_rows,
            greek_rows,
            structure_rows,
            decision_rows
        )
    )

    # 8
    report.extend(
        build_disclaimer()
    )

    # 9
    report.extend(
        build_complete(
            final_rows,
            special_rows,
            market_rows
        )
    )

    return "\n".join(report)


# ============================================================
# SPLIT TELEGRAM MESSAGE
# ============================================================

def split_message(
    text,
    limit=TELEGRAM_LIMIT
):

    if len(text) <= limit:

        return [text]

    separator = (
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    sections = text.split(
        separator
    )

    messages = []

    current = ""

    for section in sections:

        section = section.strip()

        if not section:
            continue

        section = (
            separator
            + "\n"
            + section
        )

        if (
            len(current)
            + len(section)
            + 2
            <= limit
        ):

            if current:

                current += "\n\n"

            current += section

        else:

            if current:

                messages.append(
                    current
                )

            if len(section) <= limit:

                current = section

            else:

                start = 0

                while start < len(section):

                    chunk = section[
                        start:
                        start + limit
                    ]

                    messages.append(
                        chunk
                    )

                    start += limit

                current = ""

    if current:

        messages.append(
            current
        )

    return messages


# ============================================================
# TELEGRAM REQUEST
# ============================================================

def telegram_request(
    method,
    payload
):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN "
            "GitHub Secret is missing."
        )

    url = (
        "https://api.telegram.org/bot"
        + TELEGRAM_BOT_TOKEN
        + "/"
        + method
    )

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            body = response.read().decode(
                "utf-8"
            )

        result = json.loads(body)

        if not result.get("ok"):

            raise RuntimeError(
                "Telegram API error: "
                + str(result)
            )

        return result

    except urllib.error.HTTPError as exc:

        body = ""

        try:
            body = exc.read().decode(
                "utf-8"
            )
        except Exception:
            pass

        raise RuntimeError(
            "Telegram HTTP error "
            + str(exc.code)
            + ": "
            + body
        )


# ============================================================
# TELEGRAM VALIDATION
# ============================================================

def validate_telegram_config():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing."
        )

    log(
        "TELEGRAM CONFIG : OK"
    )

    result = telegram_request(
        "getMe",
        {}
    )

    bot = result.get(
        "result",
        {}
    )

    username = bot.get(
        "username",
        "UNKNOWN"
    )

    log(
        "BOT AUTHENTICATION : OK | "
        + username
    )


# ============================================================
# SEND
# ============================================================

def send_telegram(text):

    validate_telegram_config()

    messages = split_message(text)

    total = len(messages)

    log(
        f"TELEGRAM MESSAGE COUNT : {total}"
    )

    for index, message in enumerate(
        messages,
        start=1
    ):

        if total > 1:

            prefix = (
                f"📨 MESSAGE {index}/{total}\n\n"
            )

        else:

            prefix = ""

        payload = {
            "chat_id":
                TELEGRAM_CHAT_ID,
            "text":
                prefix + message,
        }

        telegram_request(
            "sendMessage",
            payload
        )

        log(
            f"TELEGRAM SENT "
            f"{index}/{total}"
        )

    log(
        "TELEGRAM DELIVERY COMPLETE"
    )


# ============================================================
# REPORT VALIDATION
# ============================================================

def validate_report(report):

    if not report.strip():

        raise RuntimeError(
            "Generated report is empty."
        )

    required = [
        "FINAL TRADING LIST",
        "CALL BUY + PUT SELL",
        "MARKET REGIME",
        "MARKET REGIME SCORE",
        "UNUSUAL OPTION FLOW",
        "TOP 20 OPTION SEARCH",
        "DATA DISCLAIMER",
        "SCAN COMPLETE",
    ]

    missing = []

    for section in required:

        if section not in report:

            missing.append(section)

    if missing:

        raise RuntimeError(
            "Missing report sections: "
            + str(missing)
        )

    log(
        "REPORT STRUCTURE VALIDATION : OK"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log(
        "STEP 12 TELEGRAM REPORTER START"
    )

    if not os.path.isdir(DATA_DIR):

        raise RuntimeError(
            "DATA DIRECTORY NOT FOUND: "
            + DATA_DIR
        )

    report = build_report()

    validate_report(report)

    print()
    print("=" * 72)
    print("TELEGRAM REPORT PREVIEW")
    print("=" * 72)
    print(report)
    print("=" * 72)
    print()

    send_telegram(report)

    log(
        "STEP 12 TELEGRAM COMPLETE"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()


# ============================================================
# FINAL OPTION SEARCH
#
# STEP 12 - TELEGRAM REPORTER
#
# IMPORTANT
# ------------------------------------------------------------
# STEP 1 ~ STEP 11 결과를 변경하지 않습니다.
# 기존 CSV 결과만 읽어서 Telegram 보고서를 생성합니다.
#
# DATA CLASSIFICATION
# REAL
# CALCULATED
# ESTIMATED
# UNAVAILABLE
#
# Telegram 출력 순서
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
# ============================================================

import csv
import json
import os
import sys
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

MARKET_FILE = os.path.join(
    DATA_DIR,
    "market_regime.csv"
)

UNUSUAL_FILE = os.path.join(
    DATA_DIR,
    "unusual_flow.csv"
)

TOP20_FILE = os.path.join(
    DATA_DIR,
    "top20.csv"
)

OPTION_SEARCH_FILE = os.path.join(
    DATA_DIR,
    "option_search.csv"
)

STRUCTURE_FILE = os.path.join(
    DATA_DIR,
    "structure.csv"
)

DECISION_FILE = os.path.join(
    DATA_DIR,
    "decision.csv"
)

FINAL_REPORT_FILE = os.path.join(
    DATA_DIR,
    "final_report.csv"
)

SPECIAL_FILE = os.path.join(
    DATA_DIR,
    "special_list.csv"
)


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID"
)


# Telegram maximum message length is approximately 4096.
# Keep a safety margin.
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
# FILE READER
# ============================================================

def read_csv_file(path):

    if not os.path.exists(path):

        log(
            f"FILE NOT FOUND : "
            f"{os.path.relpath(path, BASE_DIR)}"
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
            f"LOADED : "
            f"{os.path.relpath(path, BASE_DIR)} "
            f"| ROWS={len(rows):,}"
        )

        return rows

    except Exception as e:

        log(
            f"FILE READ FAILED : "
            f"{path} | "
            f"{type(e).__name__}: {e}"
        )

        return []


# ============================================================
# VALUE HELPERS
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
    )


def get_value(
    row,
    *names,
    default=""
):

    if not row:
        return default

    normalized = {
        normalize_key(k): v
        for k, v in row.items()
    }

    for name in names:

        key = normalize_key(name)

        if key in normalized:

            value = normalized[key]

            if value is None:
                continue

            if str(value).strip() == "":
                continue

            return str(value).strip()

    return default


def float_value(value):

    try:

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        text = (
            text
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
        )

        return float(text)

    except Exception:

        return None


def format_number(value):

    number = float_value(value)

    if number is None:
        return str(value) if value else "N/A"

    if number.is_integer():
        return f"{int(number):,}"

    return f"{number:,.2f}"


def format_money(value):

    number = float_value(value)

    if number is None:
        return str(value) if value else "N/A"

    absolute = abs(number)

    if absolute >= 1_000_000_000:

        return f"${number / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:

        return f"${number / 1_000_000:.2f}M"

    if absolute >= 1_000:

        return f"${number / 1_000:.1f}K"

    return f"${number:,.0f}"


def display(value, fallback="N/A"):

    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    return text


# ============================================================
# DATA CLASSIFICATION
# ============================================================

def real(value):

    return f"{display(value)}"


def calculated(value):

    return f"{display(value)} [CALCULATED]"


def estimated(value):

    return f"{display(value)} [ESTIMATED]"


# ============================================================
# ROW SORTING
# ============================================================

def score_value(row):

    value = get_value(
        row,
        "score",
        "final_score",
        "flow_score",
        "unusual_flow_score",
        "structure_score",
        default="0"
    )

    number = float_value(value)

    if number is None:
        return -999999

    return number


def sorted_by_score(rows):

    return sorted(
        rows,
        key=score_value,
        reverse=True
    )


# ============================================================
# STATUS
# ============================================================

def status_emoji(value):

    text = display(value).upper()

    if any(
        word in text
        for word in [
            "BUY",
            "ENTRY",
            "ENTER",
            "LONG",
            "BULLISH"
        ]
    ):

        return "🟢"

    if any(
        word in text
        for word in [
            "WATCH",
            "NEUTRAL",
            "HOLD"
        ]
    ):

        return "🟡"

    if any(
        word in text
        for word in [
            "AVOID",
            "SELL",
            "BEARISH"
        ]
    ):

        return "🔴"

    return "⚪"


# ============================================================
# MARKET REGIME
# ============================================================

def build_market_regime(
    market_rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🌎 MARKET REGIME"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    if not market_rows:

        lines.append(
            "⚠️ MARKET REGIME DATA UNAVAILABLE"
        )

        return lines

    preferred_order = [
        "NDX",
        "SPY",
        "SOXX",
        "DIA"
    ]

    used = set()

    ordered = []

    for ticker in preferred_order:

        for row in market_rows:

            symbol = get_value(
                row,
                "symbol",
                "ticker",
                "index",
                "name"
            ).upper()

            if symbol == ticker:

                ordered.append(row)
                used.add(id(row))

                break

    for row in market_rows:

        if id(row) not in used:

            ordered.append(row)

    for row in ordered:

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "index",
            "name"
        )

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
                f"현재가 {real(price)}"
            )

        if change:
            lines.append(
                f"등락 {real(change)}"
            )

        if trend:
            lines.append(
                f"Trend       {real(trend)}"
            )

        if momentum:
            lines.append(
                f"Momentum    {real(momentum)}"
            )

        if rsi:
            lines.append(
                f"RSI         {real(rsi)}"
            )

        if support:
            lines.append(
                f"Support     {calculated(support)}"
            )

        if resistance:
            lines.append(
                f"Resistance  {calculated(resistance)}"
            )

    return lines


# ============================================================
# MARKET SCORE
# ============================================================

def build_market_score(
    market_rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 MARKET REGIME SCORE"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    for row in market_rows:

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "index",
            "name"
        )

        score = get_value(
            row,
            "score",
            "regime_score",
            "market_score"
        )

        if not symbol:
            continue

        if score:

            emoji = status_emoji(
                get_value(
                    row,
                    "trend",
                    "regime"
                )
            )

            lines.append(
                f"{symbol:<8} "
                f"{score}/100 {emoji}"
            )

    overall_score = ""

    for row in market_rows:

        candidate = get_value(
            row,
            "overall_score",
            "total_score"
        )

        if candidate:

            overall_score = candidate
            break

    regime = ""

    for row in market_rows:

        candidate = get_value(
            row,
            "overall_regime",
            "market_regime",
            "regime"
        )

        if candidate:

            regime = candidate
            break

    if overall_score:

        lines.append("")
        lines.append(
            f"OVERALL"
        )
        lines.append(
            f"🔥 {overall_score}/100"
        )

    if regime:

        lines.append(
            f"{status_emoji(regime)} {regime}"
        )

    return lines


# ============================================================
# UNUSUAL FLOW
# ============================================================

def build_unusual_flow(
    rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥 UNUSUAL OPTION FLOW"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"전체 분석 종목       {len(set("
        "get_value(r, 'symbol', 'ticker') "
        "for r in rows "
        "if get_value(r, 'symbol', 'ticker')"
        "))}"
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
        sorted_by_score(rows)[:20],
        start=1
    ):

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "underlying"
        )

        score = get_value(
            row,
            "score",
            "unusual_flow_score",
            "flow_score"
        )

        if not symbol:
            continue

        if index <= 3:

            rank = [
                "🥇",
                "🥈",
                "🥉"
            ][index - 1]

        elif index == 10:

            rank = "🔟"

        else:

            rank = f"{index}️⃣"

        lines.append(
            f"{rank} {symbol:<8} "
            f"{display(score)}"
        )

    return lines


# ============================================================
# TOP 20
# ============================================================

def build_top20(
    rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥 TOP 20 OPTION SEARCH"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    for index, row in enumerate(
        sorted_by_score(rows)[:20],
        start=1
    ):

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "underlying"
        )

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

        if not symbol:
            continue

        emoji = status_emoji(
            decision
        )

        lines.append(
            f"{index}. {symbol} "
            f"{emoji} "
            f"Score {display(score)}"
        )

    return lines


# ============================================================
# FINAL TRADING LIST
# ============================================================

def build_final_trading_list(
    rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥🔥 FINAL TRADING LIST"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    if not rows:

        lines.append(
            "⚠️ FINAL REPORT DATA UNAVAILABLE"
        )

        return lines

    ordered = sorted_by_score(
        rows
    )

    for index, row in enumerate(
        ordered[:20],
        start=1
    ):

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "underlying"
        )

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

        emoji = status_emoji(
            decision
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
            f"{emoji} {display(decision)}"
        )

        if score:
            lines.append(
                f"Score {score}"
            )

        if reason:
            lines.append(
                f"📌 이유"
            )
            lines.append(
                reason
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
                f"{calculated(support)}"
            )

        if resistance:
            lines.append(
                f"📐 Resistance "
                f"{calculated(resistance)}"
            )

    return lines


# ============================================================
# SPECIAL LIST
# ============================================================

def build_special_list(
    rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥 CALL BUY + PUT SELL"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    if not rows:

        lines.append(
            "해당 구조 없음"
        )

        return lines

    ordered = sorted_by_score(
        rows
    )

    for index, row in enumerate(
        ordered[:20],
        start=1
    ):

        symbol = get_value(
            row,
            "symbol",
            "ticker",
            "underlying"
        )

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
                f"Strength : "
                f"{strength}"
            )

        lines.append("")
        lines.append(
            "CALL BUY EST."
        )

        if call:
            lines.append(
                f"{call}"
            )

        if call_dte:
            lines.append(
                f"DTE {call_dte}"
            )

        if call_premium:
            lines.append(
                f"Premium Flow "
                f"{format_money(call_premium)} "
                f"[CALCULATED]"
            )

        lines.append("")
        lines.append(
            "PUT SELL EST."
        )

        if put:
            lines.append(
                f"{put}"
            )

        if put_dte:
            lines.append(
                f"DTE {put_dte}"
            )

        if put_premium:
            lines.append(
                f"Premium Flow "
                f"{format_money(put_premium)} "
                f"[CALCULATED]"
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
                f"Current Price "
                f"{real(current_price)}"
            )

    return lines


# ============================================================
# TOP STOCK DETAIL
# ============================================================

def build_stock_detail(
    top20_rows,
    option_rows,
    structure_rows,
    decision_rows
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🥇 TOP 종목 상세 분석"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    if not top20_rows:

        lines.append(
            "⚠️ TOP20 DATA UNAVAILABLE"
        )

        return lines

    top = sorted_by_score(
        top20_rows
    )[:1]

    if not top:
        return lines

    row = top[0]

    symbol = get_value(
        row,
        "symbol",
        "ticker",
        "underlying"
    )

    current_price = get_value(
        row,
        "current_price",
        "price"
    )

    score = get_value(
        row,
        "score",
        "option_search_score",
        "flow_score"
    )

    lines.append(
        f"🥇 {symbol}"
    )

    if current_price:
        lines.append(
            f"현재가 {real(current_price)}"
        )

    if score:
        lines.append(
            f"UNUSUAL FLOW SCORE "
            f"{score}"
        )

    # --------------------------------------------------------
    # 선정 이유
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "📌 TOP20 선정 이유"
    )

    reasons = [
        get_value(
            row,
            "reason",
            "reasons",
            "selection_reason"
        )
    ]

    for reason in reasons:

        if reason:

            for part in str(
                reason
            ).split("|"):

                part = part.strip()

                if part:
                    lines.append(
                        f"• {part}"
                    )

    # --------------------------------------------------------
    # Market Alignment
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🌎 MARKET ALIGNMENT"
    )

    alignment = get_value(
        row,
        "market_alignment",
        "alignment"
    )

    if alignment:

        lines.append(
            alignment
        )

    else:

        lines.append(
            "시장 방향과 종목 방향의"
        )
        lines.append(
            "일치 여부를 STEP 9 결과에서 확인"
        )

    # --------------------------------------------------------
    # Option Structure
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

    structure = None

    for candidate in structure_rows:

        candidate_symbol = get_value(
            candidate,
            "symbol",
            "ticker",
            "underlying"
        )

        if candidate_symbol.upper() == symbol.upper():

            structure = candidate
            break

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
                f"{calculated(call_wall)}"
            )

        if put_wall:
            lines.append(
                f"PUT WALL "
                f"{calculated(put_wall)}"
            )

        if support:
            lines.append(
                f"주요 지지 "
                f"{calculated(support)}"
            )

        if resistance:
            lines.append(
                f"주요 저항 "
                f"{calculated(resistance)}"
            )

    # --------------------------------------------------------
    # Greeks
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "📊 GREEKS / EXPOSURE"
    )

    option_detail = None

    for candidate in option_rows:

        candidate_symbol = get_value(
            candidate,
            "symbol",
            "ticker",
            "underlying"
        )

        if candidate_symbol.upper() == symbol.upper():

            option_detail = candidate
            break

    if option_detail:

        iv = get_value(
            option_detail,
            "impliedVolatility",
            "iv"
        )

        delta = get_value(
            option_detail,
            "delta"
        )

        gamma = get_value(
            option_detail,
            "gamma"
        )

        vanna = get_value(
            option_detail,
            "vanna"
        )

        charm = get_value(
            option_detail,
            "charm"
        )

        gex = get_value(
            option_detail,
            "gex"
        )

        if iv:
            lines.append(
                f"IV          "
                f"{real(iv)} [REAL]"
            )

        if delta:
            lines.append(
                f"Delta       "
                f"{real(delta)} [REAL]"
            )

        if gamma:
            lines.append(
                f"Gamma       "
                f"{calculated(gamma)}"
            )

        if vanna:
            lines.append(
                f"Vanna       "
                f"{calculated(vanna)}"
            )

        if charm:
            lines.append(
                f"Charm       "
                f"{calculated(charm)}"
            )

        if gex:
            lines.append(
                f"GEX         "
                f"{calculated(gex)}"
            )

    lines.append(
        "HIRO        UNAVAILABLE"
    )

    # --------------------------------------------------------
    # Call Flow
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🟢 TOP CALL FLOW"
    )

    call_rows = []

    for candidate in option_rows:

        candidate_symbol = get_value(
            candidate,
            "symbol",
            "ticker",
            "underlying"
        )

        option_type = get_value(
            candidate,
            "option_type",
            "type"
        ).upper()

        if (
            candidate_symbol.upper()
            == symbol.upper()
            and option_type == "CALL"
        ):

            call_rows.append(
                candidate
            )

    call_rows = sorted(
        call_rows,
        key=score_value,
        reverse=True
    )

    for call in call_rows[:2]:

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
            "openInterest",
            "open_interest"
        )

        iv = get_value(
            call,
            "impliedVolatility",
            "iv"
        )

        premium = get_value(
            call,
            "premium_flow",
            "premium"
        )

        delta = get_value(
            call,
            "delta"
        )

        gamma = get_value(
            call,
            "gamma"
        )

        lines.append("")
        lines.append(
            f"💚 ${display(strike)}C "
            f"| DTE {display(dte)}"
        )

        if volume or oi:

            lines.append(
                f"Vol {format_number(volume)} "
                f"| OI {format_number(oi)}"
            )

        if iv:
            lines.append(
                f"IV {real(iv)}"
            )

        if delta:
            lines.append(
                f"Delta {real(delta)}"
            )

        if gamma:
            lines.append(
                f"Gamma {calculated(gamma)}"
            )

        if premium:
            lines.append(
                f"Premium Flow "
                f"{format_money(premium)} "
                f"[CALCULATED]"
            )

        lines.append(
            "BUY EST. 🟢 [ESTIMATED]"
        )

        lines.append(
            "OPEN EST. 🟢 [ESTIMATED]"
        )

    # --------------------------------------------------------
    # Put Flow
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🔴 TOP PUT FLOW"
    )

    put_rows = []

    for candidate in option_rows:

        candidate_symbol = get_value(
            candidate,
            "symbol",
            "ticker",
            "underlying"
        )

        option_type = get_value(
            candidate,
            "option_type",
            "type"
        ).upper()

        if (
            candidate_symbol.upper()
            == symbol.upper()
            and option_type == "PUT"
        ):

            put_rows.append(
                candidate
            )

    put_rows = sorted(
        put_rows,
        key=score_value,
        reverse=True
    )

    for put in put_rows[:2]:

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
            "openInterest",
            "open_interest"
        )

        iv = get_value(
            put,
            "impliedVolatility",
            "iv"
        )

        premium = get_value(
            put,
            "premium_flow",
            "premium"
        )

        delta = get_value(
            put,
            "delta"
        )

        gamma = get_value(
            put,
            "gamma"
        )

        lines.append("")
        lines.append(
            f"❤️ ${display(strike)}P "
            f"| DTE {display(dte)}"
        )

        if volume or oi:

            lines.append(
                f"Vol {format_number(volume)} "
                f"| OI {format_number(oi)}"
            )

        if iv:
            lines.append(
                f"IV {real(iv)}"
            )

        if delta:
            lines.append(
                f"Delta {real(delta)}"
            )

        if gamma:
            lines.append(
                f"Gamma {calculated(gamma)}"
            )

        if premium:
            lines.append(
                f"Premium Flow "
                f"{format_money(premium)} "
                f"[CALCULATED]"
            )

        lines.append(
            "SELL EST. 🔴 [ESTIMATED]"
        )

        lines.append(
            "OPEN EST. 🟢 [ESTIMATED]"
        )

    # --------------------------------------------------------
    # Risk Reversal
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "🔥 SPECIAL STRUCTURE"
    )

    lines.append(
        "CALL BUY EST. + PUT SELL EST."
    )

    lines.append(
        "🔥 BULLISH RISK-REVERSAL"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = None

    for candidate in decision_rows:

        candidate_symbol = get_value(
            candidate,
            "symbol",
            "ticker",
            "underlying"
        )

        if candidate_symbol.upper() == symbol.upper():

            decision = candidate
            break

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

    if decision:

        decision_text = get_value(
            decision,
            "summary",
            "analysis",
            "comment",
            "reason"
        )

        if decision_text:

            for sentence in str(
                decision_text
            ).split("|"):

                sentence = sentence.strip()

                if sentence:
                    lines.append(
                        sentence
                    )

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
                f"Score "
                f"{decision_score}/100 "
                f"[CALCULATED]"
            )

    else:

        lines.append(
            "STEP 9 decision data unavailable."
        )

    # --------------------------------------------------------
    # Final Decision
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
                f"Score "
                f"{decision_score}/100 "
                f"[CALCULATED]"
            )

    return lines


# ============================================================
# DATA DISCLAIMER
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

def build_scan_complete(
    final_rows,
    special_rows
):

    entry_count = 0
    watch_count = 0
    avoid_count = 0

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
            for word in [
                "BUY",
                "ENTRY",
                "ENTER",
                "LONG"
            ]
        ):

            entry_count += 1

        elif any(
            word in decision
            for word in [
                "WATCH",
                "NEUTRAL",
                "HOLD"
            ]
        ):

            watch_count += 1

        elif any(
            word in decision
            for word in [
                "AVOID",
                "SELL"
            ]
        ):

            avoid_count += 1

    risk_reversal_count = len(
        special_rows
    )

    return [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🏁 SCAN COMPLETE",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "Market Regime",
        "🟢 CHECK STEP 1",
        "",
        f"TOP20",
        f"{min(20, len(final_rows))} 종목",
        "",
        f"Final Entry",
        f"{entry_count} 종목",
        "",
        f"Watch",
        f"{watch_count} 종목",
        "",
        f"Avoid",
        f"{avoid_count} 종목",
        "",
        "Bullish Risk-Reversal",
        f"{risk_reversal_count} 종목",
        "",
        "⏰ Next Scan",
        "다음 미국장 종료 후",
        "한국시간 오전 6:00",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 OPTION FLOW SCANNER",
        "FINAL OPTION SEARCH",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]


# ============================================================
# REPORT BUILDER
# ============================================================

def build_report():

    log("LOADING STEP 1~11 RESULTS")

    market_rows = read_csv_file(
        MARKET_FILE
    )

    unusual_rows = read_csv_file(
        UNUSUAL_FILE
    )

    top20_rows = read_csv_file(
        TOP20_FILE
    )

    option_rows = read_csv_file(
        OPTION_SEARCH_FILE
    )

    structure_rows = read_csv_file(
        STRUCTURE_FILE
    )

    decision_rows = read_csv_file(
        DECISION_FILE
    )

    final_rows = read_csv_file(
        FINAL_REPORT_FILE
    )

    special_rows = read_csv_file(
        SPECIAL_FILE
    )

    report = []

    # ========================================================
    # 1. FINAL TRADING LIST
    # ========================================================

    report.extend(
        build_final_trading_list(
            final_rows
        )
    )

    # ========================================================
    # 2. CALL BUY + PUT SELL
    # ========================================================

    report.extend(
        build_special_list(
            special_rows
        )
    )

    # ========================================================
    # 3. MARKET REGIME
    # ========================================================

    report.extend(
        build_market_regime(
            market_rows
        )
    )

    # ========================================================
    # 4. MARKET SCORE
    # ========================================================

    report.extend(
        build_market_score(
            market_rows
        )
    )

    # ========================================================
    # 5. UNUSUAL FLOW
    # ========================================================

    report.extend(
        build_unusual_flow(
            unusual_rows
        )
    )

    # ========================================================
    # 6. TOP20
    # ========================================================

    report.extend(
        build_top20(
            top20_rows
        )
    )

    # ========================================================
    # 7. TOP STOCK DETAIL
    # ========================================================

    report.extend(
        build_stock_detail(
            top20_rows,
            option_rows,
            structure_rows,
            decision_rows
        )
    )

    # ========================================================
    # 8. DISCLAIMER
    # ========================================================

    report.extend(
        build_disclaimer()
    )

    # ========================================================
    # 9. COMPLETE
    # ========================================================

    report.extend(
        build_scan_complete(
            final_rows,
            special_rows
        )
    )

    return "\n".join(
        report
    )


# ============================================================
# TELEGRAM API
# ============================================================

def telegram_request(
    method,
    payload
):

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/"
        f"{method}"
    )

    data = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type":
                "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

    return json.loads(
        body
    )


# ============================================================
# MESSAGE SPLITTER
# ============================================================

def split_message(
    text,
    limit=TELEGRAM_LIMIT
):

    if len(text) <= limit:

        return [text]

    sections = text.split(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    messages = []

    current = ""

    for section in sections:

        if not section.strip():
            continue

        section = (
            "━━━━━━━━━━━━━━━━━━━━━━"
            + section
        )

        if len(
            current
        ) + len(section) + 1 <= limit:

            current += section + "\n"

        else:

            if current.strip():

                messages.append(
                    current.strip()
                )

            # Extremely long individual
            # section fallback
            if len(section) > limit:

                start = 0

                while start < len(section):

                    messages.append(
                        section[
                            start:
                            start + limit
                        ]
                    )

                    start += limit

                current = ""

            else:

                current = section + "\n"

    if current.strip():

        messages.append(
            current.strip()
        )

    return messages


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram(
    text
):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN "
            "environment variable is missing."
        )

    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID "
            "environment variable is missing."
        )

    messages = split_message(
        text
    )

    total = len(
        messages
    )

    log(
        f"TELEGRAM MESSAGE COUNT : "
        f"{total}"
    )

    for index, message in enumerate(
        messages,
        start=1
    ):

        prefix = (
            f"📨 MESSAGE {index}/{total}\n\n"
            if total > 1
            else ""
        )

        payload = {
            "chat_id":
                TELEGRAM_CHAT_ID,
            "text":
                prefix + message,
        }

        try:

            result = telegram_request(
                "sendMessage",
                payload
            )

            if not result.get(
                "ok",
                False
            ):

                raise RuntimeError(
                    str(result)
                )

            log(
                f"TELEGRAM SENT "
                f"{index}/{total}"
            )

        except (
            urllib.error.URLError,
            urllib.error.HTTPError
        ) as e:

            raise RuntimeError(
                "Telegram API request failed: "
                f"{e}"
            )

    log(
        "TELEGRAM DELIVERY COMPLETE"
    )


# ============================================================
# LOCAL VALIDATION
# ============================================================

def validate_report(
    report
):

    if not report.strip():

        raise RuntimeError(
            "Generated Telegram report "
            "is empty."
        )

    required_sections = [
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

    for section in required_sections:

        if section not in report:

            missing.append(
                section
            )

    if missing:

        raise RuntimeError(
            "Telegram report missing "
            f"sections: {missing}"
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

    report = build_report()

    validate_report(
        report
    )

    print()
    print(
        "=" * 72
    )
    print(
        "TELEGRAM REPORT PREVIEW"
    )
    print(
        "=" * 72
    )
    print(
        report
    )
    print(
        "=" * 72
    )

    send_telegram(
        report
    )

    log(
        "STEP 12 TELEGRAM COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

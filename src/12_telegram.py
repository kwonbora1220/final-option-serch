
# ============================================================
# OPTION FLOW SCANNER
#
# STEP 12 - TELEGRAM REPORTER
#
# IMPORTANT
# ------------------------------------------------------------
# STEP 1 ~ STEP 11 결과를 변경하지 않는다.
# data/analysis/*.csv 만 읽는다.
#
# TELEGRAM ORDER
# ------------------------------------------------------------
# 1. 🔥🔥 FINAL TRADING LIST
# 2. 🔥 CALL BUY + PUT SELL
# 3. 🌎 MARKET REGIME
# 4. 🎯 MARKET REGIME SCORE
# 5. 🔥 UNUSUAL OPTION FLOW
# 6. 🔥 TOP 20 OPTION SEARCH
# 7. 🥇 TOP 종목 상세 분석
# 8. ⚠️ DATA DISCLAIMER
# 9. 🏁 SCAN COMPLETE
#
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

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)

TELEGRAM_LIMIT = 3900


# ============================================================
# FILES
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
# LOG
# ============================================================

def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[STEP 12] {now} | {message}"
    )


# ============================================================
# CSV
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        log(
            "FILE NOT FOUND | "
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
                "LOADED | "
                + os.path.relpath(
                    path,
                    BASE_DIR
                )
                + f" | ROWS={len(rows):,}"
            )

            if reader.fieldnames:

                log(
                    "COLUMNS | "
                    + ", ".join(
                        reader.fieldnames
                    )
                )

            return rows

    except Exception as exc:

        log(
            "CSV READ ERROR | "
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
# CLASSIFICATION
# ============================================================

def REAL(value):

    value = text_value(value)

    return value if value else "UNAVAILABLE"


def CALCULATED(value):

    value = text_value(value)

    if not value:
        return "UNAVAILABLE"

    return f"{value} [CALCULATED]"


def ESTIMATED(value):

    value = text_value(value)

    if not value:
        return "UNAVAILABLE"

    return f"{value} [ESTIMATED]"


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
        "stock"
    ).upper()


# ============================================================
# RANK
# ============================================================

def rank_value(row):

    value = get_value(
        row,
        "rank",
        "final_rank",
        "top_rank"
    )

    number = number_value(value)

    if number is None:
        return 999999

    return number


# ============================================================
# SCORE
# ============================================================

def score_of(row):

    value = get_value(
        row,
        "top20_score",
        "score",
        "final_score",
        "decision_score",
        "flow_score",
        "max_flow_score"
    )

    number = number_value(value)

    if number is None:
        return -999999

    return number


def sorted_rows(rows):

    return sorted(
        rows,
        key=lambda x: (
            rank_value(x),
            -score_of(x)
        )
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
# FIND SYMBOL
# ============================================================

def find_symbol(rows, symbol):

    symbol = symbol.upper()

    for row in rows:

        if symbol_of(row) == symbol:

            return row

    return None


# ============================================================
# FIND TOP20 SYMBOLS
# ============================================================

def top_symbols(rows):

    result = []

    for row in sorted_rows(rows):

        symbol = symbol_of(row)

        if not symbol:
            continue

        if symbol not in result:

            result.append(symbol)

    return result[:20]


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
            "⚪ MARKET REGIME : UNAVAILABLE"
        )

        return lines

    row = rows[0]

    regime = get_value(
        row,
        "regime",
        "market_regime"
    )

    market_score = get_value(
        row,
        "market_score",
        "score",
        "overall_score"
    )

    timestamp = get_value(
        row,
        "timestamp_utc",
        "timestamp"
    )

    if regime:

        lines.append(
            f"Market Regime : {regime}"
        )

    if market_score:

        lines.append(
            f"Market Score  : {fmt_number(market_score)}/100"
        )

    if timestamp:

        lines.append(
            f"Timestamp     : {timestamp}"
        )

    lines.append("")

    indexes = [
        ("NDX", "ndx_price", "ndx_direction"),
        ("SPY", "spy_price", "spy_direction"),
        ("SOXX", "soxx_price", "soxx_direction"),
        ("DIA", "dia_price", "dia_direction"),
    ]

    for name, price_key, direction_key in indexes:

        price = get_value(
            row,
            price_key
        )

        direction = get_value(
            row,
            direction_key
        )

        lines.append(
            f"📊 {name}"
        )

        if price:

            lines.append(
                f"Price     {REAL(price)}"
            )

        if direction:

            lines.append(
                f"Direction {status_emoji(direction)} {direction}"
            )

        lines.append("")

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

    row = rows[0]

    score = get_value(
        row,
        "market_score",
        "score",
        "overall_score",
        "total_score"
    )

    regime = get_value(
        row,
        "regime",
        "market_regime",
        "overall_regime"
    )

    if score:

        lines.append(
            f"🔥 OVERALL {fmt_number(score)}/100"
        )

    if regime:

        lines.append(
            f"{status_emoji(regime)} {regime}"
        )

    lines.append("")

    directions = [
        ("NDX", "ndx_direction"),
        ("SPY", "spy_direction"),
        ("SOXX", "soxx_direction"),
        ("DIA", "dia_direction"),
    ]

    bull = 0
    bear = 0

    for name, key in directions:

        value = get_value(
            row,
            key
        )

        if value:

            if "BULL" in value.upper():
                bull += 1

            if "BEAR" in value.upper():
                bear += 1

            lines.append(
                f"{name:<6} "
                f"{status_emoji(value)} "
                f"{value}"
            )

    lines.append("")

    lines.append(
        f"Index Alignment "
        f"BULL {bull} / BEAR {bear}"
    )

    return lines


# ============================================================
# CALL BUY + PUT SELL
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

        call_strike = get_value(
            row,
            "rr_call_strike",
            "call_strike"
        )

        call_dte = get_value(
            row,
            "rr_call_dte",
            "call_dte"
        )

        call_premium = get_value(
            row,
            "rr_call_premium",
            "call_premium"
        )

        put_strike = get_value(
            row,
            "rr_put_strike",
            "put_strike"
        )

        put_dte = get_value(
            row,
            "rr_put_dte",
            "put_dte"
        )

        put_premium = get_value(
            row,
            "rr_put_premium",
            "put_premium"
        )

        rr_score = get_value(
            row,
            "rr_score"
        )

        lines.append("")

        lines.append(
            f"{index}. {symbol}"
        )

        if rr_score:

            lines.append(
                f"RR Score {fmt_number(rr_score)}"
            )

        lines.append(
            "   CALL BUY EST. 🟢"
        )

        if call_strike:

            lines.append(
                f"   Strike {call_strike}"
            )

        if call_dte:

            lines.append(
                f"   DTE {call_dte}"
            )

        if call_premium:

            lines.append(
                "   Premium "
                + fmt_money(call_premium)
                + " [CALCULATED]"
            )

        lines.append(
            "   PUT SELL EST. 🔴"
        )

        if put_strike:

            lines.append(
                f"   Strike {put_strike}"
            )

        if put_dte:

            lines.append(
                f"   DTE {put_dte}"
            )

        if put_premium:

            lines.append(
                "   Premium "
                + fmt_money(put_premium)
                + " [CALCULATED]"
            )

        lines.append(
            "   🔥 BULLISH RISK-REVERSAL"
        )

    return lines


# ============================================================
# FINAL TRADING LIST
# ============================================================

def build_final_list(
    final_rows,
    decision_rows,
    top20_rows
):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥🔥 FINAL TRADING LIST",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    source = final_rows

    if not source:

        source = decision_rows

    if not source:

        source = top20_rows

    if not source:

        lines.append(
            "⚪ FINAL REPORT : UNAVAILABLE"
        )

        return lines

    ordered = sorted_rows(source)

    for index, row in enumerate(
        ordered[:20],
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
            "decision_score",
            "final_score",
            "top20_score",
            "score"
        )

        reason = get_value(
            row,
            "decision_reason",
            "reason",
            "selection_reason",
            "summary"
        )

        lines.append("")

        if index == 1:
            rank = "🥇"
        elif index == 2:
            rank = "🥈"
        elif index == 3:
            rank = "🥉"
        else:
            rank = f"{index}️⃣"

        lines.append(
            f"{rank} {symbol}"
        )

        if decision:

            lines.append(
                f"{status_emoji(decision)} "
                f"{decision}"
            )

        if score:

            lines.append(
                f"Score {fmt_number(score)} "
                f"[CALCULATED]"
            )

        if reason:

            lines.append(
                "📌 확인된 조건"
            )

            for part in reason.split("|"):

                part = part.strip()

                if part:

                    lines.append(
                        f"• {part}"
                    )

    return lines


# ============================================================
# TOP 20
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

    ordered = sorted_rows(rows)

    for index, row in enumerate(
        ordered[:20],
        start=1
    ):

        symbol = symbol_of(row)

        if not symbol:
            continue

        top_score = get_value(
            row,
            "top20_score"
        )

        flow_score = get_value(
            row,
            "max_flow_score",
            "flow_score"
        )

        dte = get_value(
            row,
            "top_dte",
            "top_dte_min",
            "dte_min"
        )

        direction = get_value(
            row,
            "estimated_direction",
            "direction"
        )

        lines.append("")

        if index == 1:
            rank = "🥇"
        elif index == 2:
            rank = "🥈"
        elif index == 3:
            rank = "🥉"
        else:
            rank = f"{index}."

        lines.append(
            f"{rank} {symbol}"
        )

        if top_score:

            lines.append(
                f"TOP20 Score {fmt_number(top_score)}"
            )

        if flow_score:

            lines.append(
                f"Flow Score {fmt_number(flow_score)}"
            )

        if dte:

            lines.append(
                f"DTE {dte}"
            )

        if direction:

            lines.append(
                f"Direction {direction}"
            )

    return lines


# ============================================================
# UNUSUAL FLOW
# ============================================================

def build_unusual_flow(
    unusual_rows,
    top20_rows
):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 UNUSUAL OPTION FLOW",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not unusual_rows:

        lines.append(
            "⚪ UNUSUAL FLOW FILE : UNAVAILABLE"
        )

        return lines

    symbols = set()

    for row in unusual_rows:

        symbol = symbol_of(row)

        if symbol:

            symbols.add(symbol)

    lines.append(
        f"전체 분석 종목 {len(symbols):,}"
    )

    lines.append(
        f"옵션 Flow 분석 {len(unusual_rows):,}"
    )

    lines.append("")

    lines.append(
        "오늘 비정상 Flow TOP"
    )

    # 종목별 최고 score만 표시
    best = {}

    for row in unusual_rows:

        symbol = symbol_of(row)

        if not symbol:
            continue

        score = score_of(row)

        if (
            symbol not in best
            or score > score_of(best[symbol])
        ):

            best[symbol] = row

    ordered = sorted(
        best.values(),
        key=score_of,
        reverse=True
    )

    for index, row in enumerate(
        ordered[:20],
        start=1
    ):

        symbol = symbol_of(row)

        score = get_value(
            row,
            "unusual_flow_score",
            "flow_score",
            "score"
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
            f"{fmt_number(score)}"
        )

    return lines


# ============================================================
# TOP DETAIL
# ============================================================

def build_top_detail(
    top20_rows,
    decision_rows,
    option_rows
):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🥇 TOP 종목 상세 분석",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    symbols = top_symbols(top20_rows)

    if not symbols:

        symbols = [
            symbol_of(row)
            for row in sorted_rows(
                decision_rows
            )[:20]
            if symbol_of(row)
        ]

    if not symbols:

        lines.append(
            "⚪ TOP20 : UNAVAILABLE"
        )

        return lines

    # TOP 20 전체 출력
    for position, symbol in enumerate(
        symbols[:20],
        start=1
    ):

        top = find_symbol(
            top20_rows,
            symbol
        )

        decision = find_symbol(
            decision_rows,
            symbol
        )

        option = find_symbol(
            option_rows,
            symbol
        )

        lines.append("")

        if position == 1:
            medal = "🥇"
        elif position == 2:
            medal = "🥈"
        elif position == 3:
            medal = "🥉"
        else:
            medal = f"{position}."

        lines.append(
            f"{medal} {symbol}"
        )

        # ----------------------------------------------------
        # 선정 이유
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "📌 선정 이유"
        )

        reason = ""

        if top:

            reason = get_value(
                top,
                "selection_reason",
                "reason"
            )

        if not reason and decision:

            reason = get_value(
                decision,
                "decision_reason",
                "reason"
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
                "⚪ UNAVAILABLE"
            )

        # ----------------------------------------------------
        # MARKET ALIGNMENT
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🌎 MARKET ALIGNMENT"
        )

        if decision:

            market_regime = get_value(
                decision,
                "market_regime",
                "regime"
            )

            ndx = get_value(
                decision,
                "ndx_direction"
            )

            spy = get_value(
                decision,
                "spy_direction"
            )

            soxx = get_value(
                decision,
                "soxx_direction"
            )

            dia = get_value(
                decision,
                "dia_direction"
            )

            if market_regime:

                lines.append(
                    f"Market Regime : {market_regime}"
                )

            if ndx:

                lines.append(
                    f"NDX  : {status_emoji(ndx)} {ndx}"
                )

            if spy:

                lines.append(
                    f"SPY  : {status_emoji(spy)} {spy}"
                )

            if soxx:

                lines.append(
                    f"SOXX : {status_emoji(soxx)} {soxx}"
                )

            if dia:

                lines.append(
                    f"DIA  : {status_emoji(dia)} {dia}"
                )

            bull = 0
            bear = 0

            for value in (
                ndx,
                spy,
                soxx,
                dia
            ):

                if "BULL" in value.upper():
                    bull += 1

                if "BEAR" in value.upper():
                    bear += 1

            lines.append(
                f"Index Alignment "
                f"BULL {bull} / BEAR {bear}"
            )

        else:

            lines.append(
                "⚪ UNAVAILABLE"
            )

        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        current_price = ""

        if decision:

            current_price = get_value(
                decision,
                "current_price",
                "price"
            )

        if not current_price and option:

            current_price = get_value(
                option,
                "current_price",
                "price"
            )

        if current_price:

            lines.append("")
            lines.append(
                f"현재가 {REAL(current_price)}"
            )

        # ----------------------------------------------------
        # OPTION STRUCTURE
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🎯 OPTION STRUCTURE"
        )

        source = option or decision

        if source:

            structure = get_value(
                source,
                "structure",
                "structure_bias"
            )

            call_wall = get_value(
                source,
                "call_wall"
            )

            put_wall = get_value(
                source,
                "put_wall"
            )

            support = get_value(
                source,
                "support"
            )

            resistance = get_value(
                source,
                "resistance"
            )

            if structure:

                lines.append(
                    f"Structure : {structure}"
                )

            if call_wall:

                lines.append(
                    f"CALL WALL : "
                    f"{CALCULATED(call_wall)}"
                )

            if put_wall:

                lines.append(
                    f"PUT WALL  : "
                    f"{CALCULATED(put_wall)}"
                )

            if support:

                lines.append(
                    f"Support   : "
                    f"{CALCULATED(support)}"
                )

            if resistance:

                lines.append(
                    f"Resistance: "
                    f"{CALCULATED(resistance)}"
                )

        else:

            lines.append(
                "⚪ UNAVAILABLE"
            )

        # ----------------------------------------------------
        # GEX
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "📊 GREEKS / EXPOSURE"
        )

        if option:

            total_gex = get_value(
                option,
                "total_gex",
                "net_gex",
                "gex"
            )

            if total_gex:

                lines.append(
                    f"NET GEX : "
                    f"{CALCULATED(total_gex)}"
                )

        if decision:

            call_gex = get_value(
                decision,
                "call_gex"
            )

            put_gex = get_value(
                decision,
                "put_gex"
            )

            net_gex = get_value(
                decision,
                "net_gex",
                "total_gex"
            )

            if call_gex:

                lines.append(
                    f"CALL GEX : "
                    f"{CALCULATED(call_gex)}"
                )

            if put_gex:

                lines.append(
                    f"PUT GEX  : "
                    f"{CALCULATED(put_gex)}"
                )

            if net_gex:

                lines.append(
                    f"NET GEX  : "
                    f"{CALCULATED(net_gex)}"
                )

        # ----------------------------------------------------
        # CALL FLOW
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🟢 TOP CALL FLOW"
        )

        if top:

            call_text = get_value(
                top,
                "top_call_options",
                "top_calls"
            )

            if call_text:

                for item in call_text.split("/"):

                    item = item.strip()

                    if item:

                        lines.append(
                            f"• {item}"
                        )

            elif option:

                call_strike = get_value(
                    option,
                    "rr_call_strike"
                )

                call_dte = get_value(
                    option,
                    "rr_call_dte"
                )

                call_premium = get_value(
                    option,
                    "rr_call_premium"
                )

                if call_strike:

                    lines.append(
                        f"CALL ${call_strike}"
                    )

                if call_dte:

                    lines.append(
                        f"DTE {call_dte}"
                    )

                if call_premium:

                    lines.append(
                        f"Premium "
                        f"{fmt_money(call_premium)} "
                        f"[CALCULATED]"
                    )

            else:

                lines.append(
                    "⚪ UNAVAILABLE"
                )

        # ----------------------------------------------------
        # PUT FLOW
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🔴 TOP PUT FLOW"
        )

        if top:

            put_text = get_value(
                top,
                "top_put_options",
                "top_puts"
            )

            if put_text:

                for item in put_text.split("/"):

                    item = item.strip()

                    if item:

                        lines.append(
                            f"• {item}"
                        )

            elif option:

                put_strike = get_value(
                    option,
                    "rr_put_strike"
                )

                put_dte = get_value(
                    option,
                    "rr_put_dte"
                )

                put_premium = get_value(
                    option,
                    "rr_put_premium"
                )

                if put_strike:

                    lines.append(
                        f"PUT ${put_strike}"
                    )

                if put_dte:

                    lines.append(
                        f"DTE {put_dte}"
                    )

                if put_premium:

                    lines.append(
                        f"Premium "
                        f"{fmt_money(put_premium)} "
                        f"[CALCULATED]"
                    )

            else:

                lines.append(
                    "⚪ UNAVAILABLE"
                )

        # ----------------------------------------------------
        # RISK REVERSAL
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🔥 RISK-REVERSAL"
        )

        if option:

            rr_score = get_value(
                option,
                "rr_score"
            )

            rr_call_strike = get_value(
                option,
                "rr_call_strike"
            )

            rr_call_dte = get_value(
                option,
                "rr_call_dte"
            )

            rr_call_premium = get_value(
                option,
                "rr_call_premium"
            )

            rr_put_strike = get_value(
                option,
                "rr_put_strike"
            )

            rr_put_dte = get_value(
                option,
                "rr_put_dte"
            )

            rr_put_premium = get_value(
                option,
                "rr_put_premium"
            )

            if rr_score:

                lines.append(
                    f"RR Score {fmt_number(rr_score)}"
                )

            lines.append(
                "CALL BUY EST. 🟢"
            )

            if rr_call_strike:
                lines.append(
                    f"Strike {rr_call_strike}"
                )

            if rr_call_dte:
                lines.append(
                    f"DTE {rr_call_dte}"
                )

            if rr_call_premium:
                lines.append(
                    f"Premium "
                    f"{fmt_money(rr_call_premium)} "
                    f"[CALCULATED]"
                )

            lines.append(
                "PUT SELL EST. 🔴"
            )

            if rr_put_strike:
                lines.append(
                    f"Strike {rr_put_strike}"
                )

            if rr_put_dte:
                lines.append(
                    f"DTE {rr_put_dte}"
                )

            if rr_put_premium:
                lines.append(
                    f"Premium "
                    f"{fmt_money(rr_put_premium)} "
                    f"[CALCULATED]"
                )

        else:

            lines.append(
                "⚪ UNAVAILABLE"
            )

        # ----------------------------------------------------
        # SUPPORT / RESISTANCE
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🧭 SUPPORT / RESISTANCE"
        )

        source = option or decision

        if source:

            support = get_value(
                source,
                "support"
            )

            resistance = get_value(
                source,
                "resistance"
            )

            if support:

                lines.append(
                    f"🟢 Support "
                    f"{CALCULATED(support)}"
                )

            if resistance:

                lines.append(
                    f"🔴 Resistance "
                    f"{CALCULATED(resistance)}"
                )

        # ----------------------------------------------------
        # FINAL JUDGEMENT
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🧠 종합 판단"
        )

        if decision:

            decision_reason = get_value(
                decision,
                "decision_reason",
                "reason"
            )

            if decision_reason:

                for part in decision_reason.split("|"):

                    part = part.strip()

                    if part:

                        lines.append(part)

            decision_value = get_value(
                decision,
                "decision"
            )

            decision_score = get_value(
                decision,
                "decision_score",
                "final_score"
            )

            lines.append("")

            if decision_value:

                lines.append(
                    f"{status_emoji(decision_value)} "
                    f"{decision_value}"
                )

            if decision_score:

                lines.append(
                    f"Score "
                    f"{fmt_number(decision_score)}/100 "
                    f"[CALCULATED]"
                )

        # ----------------------------------------------------
        # FINAL DECISION
        # ----------------------------------------------------

        lines.append("")
        lines.append(
            "🎯 FINAL DECISION"
        )

        if decision:

            decision_value = get_value(
                decision,
                "decision"
            )

            decision_score = get_value(
                decision,
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
                    f"{fmt_number(decision_score)}/100"
                )

        else:

            lines.append(
                "⚪ UNAVAILABLE"
            )

        # 구분
        if position < len(symbols):

            lines.append("")
            lines.append(
                "────────────────────"
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
        "본 시스템은 무료 데이터 기반",
        "옵션 분석 시스템입니다.",
        "",
        "🟢 REAL",
        "무료 데이터에서 직접 제공된 값",
        "",
        "🔵 CALCULATED",
        "무료 데이터로 자체 계산한 값",
        "",
        "🟡 ESTIMATED",
        "무료 데이터로 직접 확인할 수 없어",
        "간접적으로 추정한 값",
        "",
        "⚪ UNAVAILABLE",
        "무료 데이터에서 확인할 수 없는 값",
        "",
        "특히 BUY / SELL 및",
        "OPEN / CLOSE는 추정값이며",
        "실제 체결 방향을 의미하지 않습니다.",
        "",
        "실제 기관 포지션을",
        "확정적으로 의미하지 않습니다.",
    ]


# ============================================================
# SCAN COMPLETE
# ============================================================

def build_complete(
    final_rows,
    special_rows,
    market_rows,
    top20_rows
):

    entries = 0
    watches = 0
    avoids = 0

    source = final_rows

    if not source:
        source = top20_rows

    for row in source:

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

    return [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🏁 SCAN COMPLETE",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "Market Regime",
        "🟢 CHECKED"
        if market_rows
        else "⚪ UNAVAILABLE",
        "",
        "TOP20",
        f"{min(20, len(top20_rows))} 종목",
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

    # ========================================================
    # 1. FINAL TRADING LIST
    # ========================================================

    report.extend(
        build_final_list(
            final_rows,
            decision_rows,
            top20_rows
        )
    )

    # ========================================================
    # 2. CALL BUY + PUT SELL
    # ========================================================

    report.extend(
        build_special(
            option_rows
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
    # 4. MARKET REGIME SCORE
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
            unusual_rows,
            top20_rows
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
    # 7. TOP DETAIL
    # ========================================================

    report.extend(
        build_top_detail(
            top20_rows,
            decision_rows,
            option_rows
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
        build_complete(
            final_rows,
            special_rows,
            market_rows,
            top20_rows
        )
    )

    return "\n".join(report)


# ============================================================
# SPLIT MESSAGE
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
                        start:start + limit
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
# TELEGRAM API
# ============================================================

def telegram_request(
    method,
    payload
):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
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
# SEND TELEGRAM
# ============================================================

def send_telegram(text):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing."
        )

    messages = split_message(text)

    total = len(messages)

    log(
        f"TELEGRAM MESSAGE COUNT : {total}"
    )

    for index, message in enumerate(
        messages,
        start=1
    ):

        prefix = ""

        if total > 1:

            prefix = (
                f"📨 MESSAGE {index}/{total}\n\n"
            )

        result = telegram_request(
            "sendMessage",
            {
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    prefix + message
            }
        )

        message_id = (
            result
            .get("result", {})
            .get("message_id", "N/A")
        )

        log(
            f"TELEGRAM SENT "
            f"{index}/{total} | "
            f"message_id={message_id}"
        )

    log(
        "🔥 TELEGRAM DELIVERY COMPLETE"
    )


# ============================================================
# VALIDATION
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
        "TOP 종목 상세 분석",
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
        "🔥 STEP 12 TELEGRAM REPORTER START"
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
        "🔥 STEP 12 TELEGRAM COMPLETE"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()


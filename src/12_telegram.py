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
    "analysis",
)

TELEGRAM_BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    "",
)

TELEGRAM_CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "",
)

TELEGRAM_LIMIT = 3900


FILES = {
    "market": os.path.join(
        DATA_DIR,
        "market_regime.csv",
    ),
    "unusual": os.path.join(
        DATA_DIR,
        "unusual_flow.csv",
    ),
    "top20": os.path.join(
        DATA_DIR,
        "top20.csv",
    ),
    "option_search": os.path.join(
        DATA_DIR,
        "option_search.csv",
    ),
    "greeks": os.path.join(
        DATA_DIR,
        "options_greeks.csv",
    ),
    "structure": os.path.join(
        DATA_DIR,
        "structure.csv",
    ),
    "decision": os.path.join(
        DATA_DIR,
        "decision.csv",
    ),
    "final": os.path.join(
        DATA_DIR,
        "final_report.csv",
    ),
    "special": os.path.join(
        DATA_DIR,
        "special_list.csv",
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
                BASE_DIR,
            )
        )

        return []

    try:

        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

            log(
                "LOADED | "
                + os.path.relpath(
                    path,
                    BASE_DIR,
                )
                + f" | ROWS={len(rows):,}"
            )

            return rows

    except Exception as exc:

        log(
            f"CSV READ ERROR | {exc}"
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

    normalized = {
        normalize_key(key): value
        for key, value in row.items()
    }

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
        return str(value).strip() or "N/A"

    if number.is_integer():
        return f"{int(number):,}"

    return f"{number:,.2f}"


def fmt_money(value):

    number = number_value(value)

    if number is None:
        return str(value).strip() or "N/A"

    absolute = abs(number)

    if absolute >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"${number / 1_000:.1f}K"

    return f"${number:,.0f}"


def symbol_of(row):

    return get_value(
        row,
        "symbol",
        "ticker",
        "underlying",
        "underlying_symbol",
        "stock",
    ).upper()


def score_of(row):

    value = get_value(
        row,
        "decision_score",
        "top20_score",
        "final_score",
        "score",
    )

    number = number_value(value)

    if number is None:
        return -999999

    return number


def rank_of(row):

    value = get_value(
        row,
        "final_rank",
        "rank",
    )

    number = number_value(value)

    if number is None:
        return 999999

    return number


def sorted_rows(rows):

    return sorted(
        rows,
        key=lambda row: (
            rank_of(row),
            -score_of(row),
        ),
    )


# ============================================================
# STATUS
# ============================================================

def status_emoji(value):

    text = str(
        value or ""
    ).upper()

    if (
        "진입" in text
        or "ENTRY" in text
        or "ENTER" in text
    ):
        return "🟢"

    if (
        "관망" in text
        or "WATCH" in text
        or "NEUTRAL" in text
    ):
        return "🟡"

    if (
        "회피" in text
        or "AVOID" in text
        or "SELL" in text
    ):
        return "🔴"

    return "⚪"


# ============================================================
# FIND
# ============================================================

def find_symbol(rows, symbol):

    symbol = symbol.upper()

    for row in rows:

        if symbol_of(row) == symbol:
            return row

    return None


# ============================================================
# FINAL TRADING LIST
# ============================================================

def build_final_list(final_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥🔥 FINAL TRADING LIST",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    rows = sorted_rows(
        final_rows
    )[:20]

    if not rows:

        lines.append(
            "데이터 없음"
        )

        return lines

    medals = [
        "🥇",
        "🥈",
        "🥉",
    ]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        symbol = symbol_of(row)

        decision = get_value(
            row,
            "final_decision",
            "decision",
        )

        score = get_value(
            row,
            "decision_score",
            "score",
        )

        prefix = (
            medals[index - 1]
            if index <= 3
            else f"{index}️⃣"
        )

        emoji = status_emoji(
            decision
        )

        lines.append(
            f"{prefix} {symbol}"
        )

        lines.append(
            f"{emoji} {decision}"
        )

        lines.append(
            f"Score {fmt_number(score)} "
            "[CALCULATED]"
        )

    return lines


# ============================================================
# SPECIAL LIST
# ============================================================

def build_special(special_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 CALL BUY + PUT SELL",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    rows = sorted_rows(
        special_rows
    )[:20]

    if not rows:

        lines.append(
            "해당 구조 없음"
        )

        return lines

    for index, row in enumerate(
        rows,
        start=1,
    ):

        symbol = symbol_of(row)

        rr_score = get_value(
            row,
            "rr_score",
        )

        call_strike = get_value(
            row,
            "call_strike",
            "rr_call_strike",
        )

        call_dte = get_value(
            row,
            "call_dte",
            "rr_call_dte",
        )

        call_premium = get_value(
            row,
            "call_premium",
            "rr_call_premium",
        )

        put_strike = get_value(
            row,
            "put_strike",
            "rr_put_strike",
        )

        put_dte = get_value(
            row,
            "put_dte",
            "rr_put_dte",
        )

        put_premium = get_value(
            row,
            "put_premium",
            "rr_put_premium",
        )

        lines.append("")
        lines.append(
            f"{index}. {symbol}"
        )

        lines.append(
            f"   RR Score {fmt_number(rr_score)}"
        )

        lines.append(
            "   CALL BUY EST. 🟢"
        )

        lines.append(
            f"   Strike {call_strike}"
        )

        lines.append(
            f"   DTE {call_dte}"
        )

        lines.append(
            f"   Premium "
            f"{fmt_money(call_premium)} "
            "[CALCULATED]"
        )

        lines.append(
            "   PUT SELL EST. 🔴"
        )

        lines.append(
            f"   Strike {put_strike}"
        )

        lines.append(
            f"   DTE {put_dte}"
        )

        lines.append(
            f"   Premium "
            f"{fmt_money(put_premium)} "
            "[CALCULATED]"
        )

        lines.append(
            "   🔥 BULLISH RISK-REVERSAL"
        )

    return lines


# ============================================================
# MARKET
# ============================================================

def build_market(market_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🌎 MARKET REGIME",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not market_rows:

        lines.append(
            "Market Regime : UNAVAILABLE"
        )

        return lines

    row = market_rows[-1]

    regime = get_value(
        row,
        "market_regime",
        "regime",
    )

    score = get_value(
        row,
        "market_score",
        "score",
    )

    lines.append(
        f"Market Regime : {regime or 'UNKNOWN'}"
    )

    lines.append(
        f"Market Score  : "
        f"{fmt_number(score)}/100"
    )

    lines.append("")

    for name, key in [
        ("NDX", "ndx_direction"),
        ("SPY", "spy_direction"),
        ("SOXX", "soxx_direction"),
        ("DIA", "dia_direction"),
    ]:

        value = get_value(
            row,
            key,
        )

        if value:

            lines.append(
                f"{name:<6} "
                f"{status_emoji(value)} "
                f"{value}"
            )

    return lines


# ============================================================
# MARKET SCORE
# ============================================================

def build_market_score(market_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 MARKET REGIME SCORE",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not market_rows:

        lines.append(
            "UNAVAILABLE"
        )

        return lines

    row = market_rows[-1]

    score = get_value(
        row,
        "market_score",
        "score",
    )

    regime = get_value(
        row,
        "market_regime",
        "regime",
    )

    lines.append(
        f"🔥 OVERALL "
        f"{fmt_number(score)}/100"
    )

    lines.append(
        f"{status_emoji(regime)} "
        f"{regime}"
    )

    bull = 0
    bear = 0

    for name, key in [
        ("NDX", "ndx_direction"),
        ("SPY", "spy_direction"),
        ("SOXX", "soxx_direction"),
        ("DIA", "dia_direction"),
    ]:

        value = get_value(
            row,
            key,
        )

        if "BULL" in value.upper():
            bull += 1

        if "BEAR" in value.upper():
            bear += 1

        if value:

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
# UNUSUAL FLOW
# ============================================================

def build_unusual(unusual_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 UNUSUAL OPTION FLOW",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if not unusual_rows:

        lines.append(
            "데이터 없음"
        )

        return lines

    lines.append(
        f"전체 분석 종목 "
        f"{len(set(symbol_of(r) for r in unusual_rows))}"
    )

    lines.append(
        f"옵션 Flow 분석 "
        f"{len(unusual_rows):,}"
    )

    # symbol별 최고 flow
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
        key=lambda row: score_of(row),
        reverse=True,
    )[:20]

    lines.append(
        "오늘 비정상 Flow TOP"
    )

    for index, row in enumerate(
        ordered,
        start=1,
    ):

        symbol = symbol_of(row)

        score = get_value(
            row,
            "flow_score",
            "score",
        )

        lines.append(
            f"{index:>2}. "
            f"{symbol:<6} "
            f"{fmt_number(score)}"
        )

    return lines


# ============================================================
# TOP20
# ============================================================

def build_top20(top20_rows):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 TOP 20 OPTION SEARCH",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    rows = sorted_rows(
        top20_rows
    )[:20]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        symbol = symbol_of(row)

        score = get_value(
            row,
            "top20_score",
            "score",
        )

        flow = get_value(
            row,
            "flow_score",
            "max_flow_score",
        )

        dte = get_value(
            row,
            "top_dte",
        )

        direction = get_value(
            row,
            "estimated_direction",
            "flow_direction",
        )

        lines.append(
            f"{index:>2}. {symbol}"
        )

        lines.append(
            f"    TOP20 Score {fmt_number(score)}"
        )

        lines.append(
            f"    Flow Score {fmt_number(flow)}"
        )

        lines.append(
            f"    DTE {fmt_number(dte)}"
        )

        lines.append(
            f"    Direction {direction}"
        )

    return lines


# ============================================================
# DETAIL
# ============================================================

def build_detail(
    final_rows,
    special_rows,
    structure_rows,
):

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🥇 TOP 종목 상세 분석",
        "━━━━━━━━━━━━━━━━━━━━━━",
    ]

    rows = sorted_rows(
        final_rows
    )[:20]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        symbol = symbol_of(row)

        decision = get_value(
            row,
            "final_decision",
            "decision",
        )

        score = get_value(
            row,
            "decision_score",
            "score",
        )

        structure = find_symbol(
            structure_rows,
            symbol,
        )

        special = find_symbol(
            special_rows,
            symbol,
        )

        lines.append("")

        # 중요:
        # 여기서 index는 항상 1~20으로 다시 시작한다.
        lines.append(
            f"{index}️⃣ {symbol}"
        )

        lines.append(
            f"🎯 FINAL DECISION"
        )

        lines.append(
            f"{status_emoji(decision)} "
            f"{decision}"
        )

        lines.append(
            f"Score {fmt_number(score)}/100"
        )

        if structure:

            current_price = get_value(
                structure,
                "current_price",
            )

            structure_name = get_value(
                structure,
                "structure",
            )

            call_wall = get_value(
                structure,
                "call_wall",
            )

            put_wall = get_value(
                structure,
                "put_wall",
            )

            gex = get_value(
                structure,
                "gex_structure",
            )

            lines.append("")

            lines.append(
                f"현재가 "
                f"{current_price or 'N/A'}"
            )

            lines.append(
                f"Structure : "
                f"{structure_name or 'N/A'}"
            )

            lines.append(
                f"CALL WALL : "
                f"{call_wall or 'N/A'}"
            )

            lines.append(
                f"PUT WALL  : "
                f"{put_wall or 'N/A'}"
            )

            lines.append(
                f"GEX : "
                f"{gex or 'N/A'}"
            )

        if special:

            lines.append("")

            lines.append(
                "🔥 RISK-REVERSAL"
            )

            rr = get_value(
                special,
                "rr_score",
            )

            lines.append(
                f"RR Score {fmt_number(rr)}"
            )

            lines.append(
                "CALL BUY EST. 🟢"
            )

            lines.append(
                f"Strike "
                f"{get_value(special, 'call_strike')}"
            )

            lines.append(
                f"DTE "
                f"{get_value(special, 'call_dte')}"
            )

            lines.append(
                f"Premium "
                f"{fmt_money(get_value(special, 'call_premium'))}"
            )

            lines.append(
                "PUT SELL EST. 🔴"
            )

            lines.append(
                f"Strike "
                f"{get_value(special, 'put_strike')}"
            )

            lines.append(
                f"DTE "
                f"{get_value(special, 'put_dte')}"
            )

            lines.append(
                f"Premium "
                f"{fmt_money(get_value(special, 'put_premium'))}"
            )

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
        "Option flow side는 일부 구간에서",
        "BID/ASK/LAST 기반으로 추정될 수 있습니다.",
        "본 리포트는 자동 계산 결과이며",
        "투자 판단의 참고자료로만 사용하세요.",
    ]


# ============================================================
# TELEGRAM
# ============================================================

def send_message(text):

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing"
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing"
        )

    url = (
        "https://api.telegram.org/"
        "bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type":
                "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:

        body = response.read().decode(
            "utf-8"
        )

        result = json.loads(
            body
        )

        if not result.get(
            "ok",
            False,
        ):
            raise RuntimeError(
                f"Telegram API error: {body}"
            )

        return result


# ============================================================
# MESSAGE SPLIT
# ============================================================

def split_messages(lines):

    messages = []

    current = ""

    for line in lines:

        candidate = (
            line
            if not current
            else current
            + "\n"
            + line
        )

        if len(candidate) <= TELEGRAM_LIMIT:

            current = candidate

        else:

            if current:
                messages.append(
                    current
                )

            current = line

    if current:
        messages.append(
            current
        )

    return messages


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("STEP 12 INPUT CHECK")
    print("=" * 78)

    final_rows = read_csv(
        FILES["final"]
    )

    special_rows = read_csv(
        FILES["special"]
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

    structure_rows = read_csv(
        FILES["structure"]
    )

    if not final_rows:
        raise ValueError(
            "final_report.csv is empty"
        )

    # ========================================================
    # BUILD REPORT
    # ========================================================

    lines = []

    lines.extend(
        build_final_list(
            final_rows
        )
    )

    lines.append("")

    lines.extend(
        build_special(
            special_rows
        )
    )

    lines.append("")

    lines.extend(
        build_market(
            market_rows
        )
    )

    lines.append("")

    lines.extend(
        build_market_score(
            market_rows
        )
    )

    lines.append("")

    lines.extend(
        build_unusual(
            unusual_rows
        )
    )

    lines.append("")

    lines.extend(
        build_top20(
            top20_rows
        )
    )

    lines.append("")

    lines.extend(
        build_detail(
            final_rows,
            special_rows,
            structure_rows,
        )
    )

    lines.append("")

    lines.extend(
        build_disclaimer()
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🏁 SCAN COMPLETE"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━"
    )

    messages = split_messages(
        lines
    )

    print()
    print(
        f"[STEP 12] TELEGRAM MESSAGE COUNT : "
        f"{len(messages)}"
    )

    # ========================================================
    # SEND
    # ========================================================

    for index, message in enumerate(
        messages,
        start=1,
    ):

        try:

            result = send_message(
                message
            )

            message_id = (
                result
                .get("result", {})
                .get("message_id", "N/A")
            )

            print(
                f"[STEP 12] TELEGRAM SENT "
                f"{index}/{len(messages)} | "
                f"message_id={message_id}"
            )

        except urllib.error.HTTPError as exc:

            body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                "Telegram HTTP error "
                f"{exc.code}: {body}"
            )

        except Exception as exc:

            raise RuntimeError(
                f"Telegram send failed: {exc}"
            )

    print()
    print(
        "[STEP 12] 🔥 TELEGRAM DELIVERY COMPLETE"
    )

    print(
        "[STEP 12] 🔥 STEP 12 TELEGRAM COMPLETE"
    )


if __name__ == "__main__":
    main()

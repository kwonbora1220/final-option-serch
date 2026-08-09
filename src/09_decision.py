from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "analysis",
)

MARKET_FILE = os.path.join(
    ANALYSIS_DIR,
    "market_regime.csv",
)

FLOW_FILE = os.path.join(
    ANALYSIS_DIR,
    "unusual_flow.csv",
)

TOP20_FILE = os.path.join(
    ANALYSIS_DIR,
    "top20.csv",
)

STRUCTURE_FILE = os.path.join(
    ANALYSIS_DIR,
    "structure.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv",
)


def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[09 DECISION] {now} | {message}"
    )


def find_col(df, names):

    normalized = {
        str(c)
        .strip()
        .lower()
        .replace(" ", "_"): c
        for c in df.columns
    }

    for name in names:

        key = (
            name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def numeric(value):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def clamp(value):

    if not np.isfinite(value):
        return 50.0

    return max(
        0.0,
        min(100.0, float(value)),
    )


def text(value):

    if pd.isna(value):
        return ""

    return str(value).upper().strip()


def load_market():

    df = pd.read_csv(
        MARKET_FILE
    )

    score_col = find_col(
        df,
        [
            "market_score",
            "market_regime_score",
            "score",
        ],
    )

    if score_col is None:
        raise RuntimeError(
            "market_score missing"
        )

    score = numeric(
        df.iloc[-1][score_col]
    )

    regime_col = find_col(
        df,
        [
            "market_regime",
            "regime",
        ],
    )

    regime = (
        text(df.iloc[-1][regime_col])
        if regime_col
        else "UNKNOWN"
    )

    directions = {}

    for name in [
        "ndx_direction",
        "spy_direction",
        "soxx_direction",
        "dia_direction",
    ]:

        col = find_col(
            df,
            [name],
        )

        directions[name] = (
            text(df.iloc[-1][col])
            if col
            else "UNAVAILABLE"
        )

    return {
        "market_score": clamp(score),
        "market_regime": regime,
        **directions,
    }


def load_top20():

    df = pd.read_csv(
        TOP20_FILE
    )

    ticker_col = find_col(
        df,
        ["ticker", "symbol"],
    )

    if ticker_col is None:
        raise RuntimeError(
            "TOP20 ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    for field in [
        "top20_score",
        "max_flow_score",
        "avg_flow_score",
        "total_volume",
        "total_premium",
        "call_premium",
        "put_premium",
        "call_put_imbalance",
        "directional_ratio",
        "top_dte",
    ]:

        col = find_col(
            df,
            [field],
        )

        if col:
            df[field] = pd.to_numeric(
                df[col],
                errors="coerce",
            )
        else:
            df[field] = np.nan

    for field in [
        "flow_direction",
        "estimated_direction",
    ]:

        col = find_col(
            df,
            [field],
        )

        if col:
            df[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )
        else:
            df[field] = ""

    return (
        df
        .drop_duplicates("ticker")
        .head(20)
    )


def load_flow():

    df = pd.read_csv(
        FLOW_FILE
    )

    ticker_col = find_col(
        df,
        ["ticker", "symbol"],
    )

    score_col = find_col(
        df,
        [
            "flow_score",
            "option_flow_score",
            "score",
        ],
    )

    if ticker_col is None:
        raise RuntimeError(
            "FLOW ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if score_col:

        df["flow_value"] = pd.to_numeric(
            df[score_col],
            errors="coerce",
        )

    else:

        df["flow_value"] = np.nan

    return (
        df
        .groupby(
            "ticker",
            as_index=False,
        )
        .agg(
            flow_score=(
                "flow_value",
                "max",
            )
        )
    )


def load_structure():

    df = pd.read_csv(
        STRUCTURE_FILE
    )

    ticker_col = find_col(
        df,
        ["ticker", "symbol"],
    )

    if ticker_col is None:
        raise RuntimeError(
            "STRUCTURE ticker missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_fields = [
        "current_price",
        "call_wall",
        "put_wall",
        "support",
        "resistance",
        "call_gex",
        "put_gex",
        "net_gex",
    ]

    for field in numeric_fields:

        col = find_col(
            df,
            [field],
        )

        if col:
            df[field] = pd.to_numeric(
                df[col],
                errors="coerce",
            )
        else:
            df[field] = np.nan

    for field in [
        "structure",
        "price_location",
        "gex_structure",
        "wall_structure",
    ]:

        col = find_col(
            df,
            [field],
        )

        if col:
            df[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )
        else:
            df[field] = ""

    return (
        df
        .drop_duplicates("ticker")
    )


def direction_score(row):

    direction = text(
        row["estimated_direction"]
    )

    flow_direction = text(
        row["flow_direction"]
    )

    ratio = numeric(
        row["directional_ratio"]
    )

    if "BUY" in direction:
        base = 80.0

    elif "SELL" in direction:
        base = 20.0

    elif "MIXED" in direction:
        base = 50.0

    else:
        base = 50.0

    if np.isfinite(ratio):

        ratio = clamp(ratio)

        base = (
            base * 0.65
            +
            ratio * 0.35
        )

    if "CALL DOMINANT" in flow_direction:
        base += 5

    if "PUT DOMINANT" in flow_direction:
        base -= 5

    return clamp(base)


def structure_score(row):

    structure = text(
        row["structure"]
    )

    wall = text(
        row["wall_structure"]
    )

    gex = text(
        row["gex_structure"]
    )

    if structure == "BULLISH":
        score = 85.0

    elif structure == "BEARISH":
        score = 20.0

    elif "POSITIVE GEX" in structure:
        score = 68.0

    elif "NEGATIVE GEX" in structure:
        score = 32.0

    else:
        score = 50.0

    if wall == "BULLISH BREAKOUT":
        score += 10

    elif wall == "BEARISH BREAKDOWN":
        score -= 10

    if gex == "POSITIVE GEX":
        score += 5

    elif gex == "NEGATIVE GEX":
        score -= 5

    return clamp(score)


def price_score(row):

    price = numeric(
        row["current_price"]
    )

    support = numeric(
        row["support"]
    )

    resistance = numeric(
        row["resistance"]
    )

    if not np.isfinite(price):
        return 50.0

    score = 50.0

    if np.isfinite(support):

        d = (
            price - support
        ) / price

        if 0 <= d <= 0.02:
            score += 12

        elif d > 0:
            score += 5

    if np.isfinite(resistance):

        d = (
            resistance - price
        ) / price

        if 0 <= d <= 0.02:
            score += 8

        elif d > 0:
            score += 4

    return clamp(score)


def index_score(market):

    values = [
        market["ndx_direction"],
        market["spy_direction"],
        market["soxx_direction"],
        market["dia_direction"],
    ]

    bull = sum(
        "BULL" in value
        for value in values
    )

    bear = sum(
        "BEAR" in value
        for value in values
    )

    neutral = (
        len(values)
        - bull
        - bear
    )

    if bull >= 3 and bear == 0:
        score = 90.0

    elif bull > bear:
        score = 70.0

    elif bear >= 3 and bull == 0:
        score = 20.0

    elif bear > bull:
        score = 35.0

    else:
        score = 50.0

    return (
        clamp(score),
        bull,
        bear,
        neutral,
    )


def final_decision(
    score,
    direction,
    structure,
    market_score,
):

    structure = text(
        structure
    )

    # 방향성이 매우 약하면 진입 금지
    if direction < 30:
        return "🟡 관망"

    # 약세 구조는 높은 flow만으로 진입시키지 않음
    if structure == "BEARISH":
        return "🔴 회피"

    # 시장 자체가 약하면 진입 기준 강화
    if market_score < 45:

        if score >= 82 and direction >= 70:
            return "🟡 관망"

        return "🔴 회피"

    if score >= 75 and direction >= 55:
        return "🟢 진입"

    if score >= 55:
        return "🟡 관망"

    return "🔴 회피"


def main():

    log("START")

    for path in [
        MARKET_FILE,
        FLOW_FILE,
        TOP20_FILE,
        STRUCTURE_FILE,
    ]:

        if not os.path.exists(path):
            raise FileNotFoundError(path)

    market = load_market()
    top20 = load_top20()
    flow = load_flow()
    structure = load_structure()

    market_component = market["market_score"]

    index_component, bull, bear, neutral = (
        index_score(market)
    )

    rows = []

    for rank, top in enumerate(
        top20.to_dict("records"),
        start=1,
    ):

        ticker = top["ticker"]

        flow_row = flow[
            flow["ticker"] == ticker
        ]

        structure_row = structure[
            structure["ticker"] == ticker
        ]

        if flow_row.empty:
            flow_score = numeric(
                top["max_flow_score"]
            )
        else:
            flow_score = numeric(
                flow_row.iloc[0]["flow_score"]
            )

        if not np.isfinite(flow_score):
            flow_score = numeric(
                top["avg_flow_score"]
            )

        if not np.isfinite(flow_score):
            flow_score = 50.0

        if structure_row.empty:
            s = {
                "current_price": np.nan,
                "call_wall": np.nan,
                "put_wall": np.nan,
                "support": np.nan,
                "resistance": np.nan,
                "net_gex": np.nan,
                "structure": "NO DATA",
                "price_location": "",
                "gex_structure": "GEX UNAVAILABLE",
                "wall_structure": "WALL UNAVAILABLE",
            }

        else:
            s = structure_row.iloc[0].to_dict()

        direction = direction_score(
            top
        )

        structure_component = structure_score(
            s
        )

        price_component = price_score(
            s
        )

        # TOP20의 score가 존재하면 보조적으로 사용
        top20_score = numeric(
            top["top20_score"]
        )

        if np.isfinite(top20_score):
            top20_component = clamp(
                top20_score
            )
        else:
            top20_component = flow_score

        # 최종 점수
        #
        # Market       20
        # Flow         25
        # Direction    20
        # Structure    10
        # Price        15
        # Index        10

        score = (
            market_component * 0.20
            +
            flow_score * 0.25
            +
            direction * 0.20
            +
            structure_component * 0.10
            +
            price_component * 0.15
            +
            index_component * 0.10
        )

        score = clamp(score)

        decision = final_decision(
            score,
            direction,
            s["structure"],
            market_component,
        )

        reasons = []

        if market_component >= 70:
            reasons.append(
                "Strong market regime"
            )

        elif market_component < 45:
            reasons.append(
                "Weak market regime"
            )

        if flow_score >= 80:
            reasons.append(
                "Very high option flow"
            )

        elif flow_score >= 65:
            reasons.append(
                "Strong option flow"
            )

        if direction >= 70:
            reasons.append(
                "Bullish direction"
            )

        elif direction < 30:
            reasons.append(
                "Weak direction"
            )

        if (
            "BULLISH"
            in text(s["structure"])
        ):
            reasons.append(
                "Bullish structure"
            )

        elif (
            "BEARISH"
            in text(s["structure"])
        ):
            reasons.append(
                "Bearish structure"
            )

        if (
            "POSITIVE GEX"
            in text(s["gex_structure"])
        ):
            reasons.append(
                "Positive GEX"
            )

        elif (
            "NEGATIVE GEX"
            in text(s["gex_structure"])
        ):
            reasons.append(
                "Negative GEX"
            )

        if not reasons:
            reasons.append(
                "Mixed structure"
            )

        rows.append(
            {
                "rank": rank,
                "ticker": ticker,
                "market_score": round(
                    market_component,
                    2,
                ),
                "market_regime": market[
                    "market_regime"
                ],
                "ndx_direction": market[
                    "ndx_direction"
                ],
                "spy_direction": market[
                    "spy_direction"
                ],
                "soxx_direction": market[
                    "soxx_direction"
                ],
                "dia_direction": market[
                    "dia_direction"
                ],
                "top20_score": top20_component,
                "flow_score": round(
                    flow_score,
                    2,
                ),
                "direction_score": round(
                    direction,
                    2,
                ),
                "structure_score": round(
                    structure_component,
                    2,
                ),
                "price_score": round(
                    price_component,
                    2,
                ),
                "index_score": round(
                    index_component,
                    2,
                ),
                "current_price": s[
                    "current_price"
                ],
                "call_wall": s[
                    "call_wall"
                ],
                "put_wall": s[
                    "put_wall"
                ],
                "support": s[
                    "support"
                ],
                "resistance": s[
                    "resistance"
                ],
                "call_gex": s.get(
                    "call_gex",
                    np.nan,
                ),
                "put_gex": s.get(
                    "put_gex",
                    np.nan,
                ),
                "net_gex": s.get(
                    "net_gex",
                    np.nan,
                ),
                "structure": s[
                    "structure"
                ],
                "price_location": s[
                    "price_location"
                ],
                "gex_structure": s[
                    "gex_structure"
                ],
                "wall_structure": s[
                    "wall_structure"
                ],
                "index_bull": bull,
                "index_bear": bear,
                "index_neutral": neutral,
                "decision_score": round(
                    score,
                    2,
                ),
                "decision": decision,
                "reason": " | ".join(
                    reasons
                ),
                "data_source": "CALCULATED",
            }
        )

    output = pd.DataFrame(rows)

    if len(output) != 20:
        raise RuntimeError(
            "STEP 9 must produce 20 rows"
        )

    if output["ticker"].nunique() != 20:
        raise RuntimeError(
            "Duplicate ticker detected"
        )

    if (
        output["decision_score"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "decision_score contains NaN"
        )

    if (
        (output["decision_score"] < 0)
        |
        (output["decision_score"] > 100)
    ).any():
        raise RuntimeError(
            "decision_score outside 0-100"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("STEP 9 DECISION COMPLETE")
    print()
    print(
        output[
            [
                "rank",
                "ticker",
                "direction_score",
                "decision_score",
                "decision",
            ]
        ].to_string(index=False)
    )

    print()
    print("DECISION SUMMARY")
    print(
        output[
            "decision"
        ].value_counts()
    )

    print("STEP 9 OUTPUT : OK")


if __name__ == "__main__":
    main()

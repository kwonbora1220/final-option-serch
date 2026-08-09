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


# ============================================================
# WEIGHTS
# ============================================================

MARKET_WEIGHT = 20.0
FLOW_WEIGHT = 25.0
DIRECTION_WEIGHT = 20.0
STRUCTURE_WEIGHT = 10.0
PRICE_WEIGHT = 15.0
INDEX_WEIGHT = 10.0

ENTER_THRESHOLD = 75.0
WATCH_THRESHOLD = 55.0


# ============================================================
# HELPERS
# ============================================================

def log(message):

    now = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    print(
        f"[09 DECISION] {now} | {message}"
    )


def find_column(df, candidates):

    mapping = {
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_"): c
        for c in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if key in mapping:
            return mapping[key]

    return None


def numeric(df, column):

    if column is None:

        return pd.Series(
            np.nan,
            index=df.index,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def clean_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
    )


def clamp(value, low=0, high=100):

    if not np.isfinite(value):
        return 0.0

    return max(
        low,
        min(high, float(value)),
    )


# ============================================================
# FILE CHECK
# ============================================================

def check_files():

    files = {
        "MARKET": MARKET_FILE,
        "FLOW": FLOW_FILE,
        "TOP20": TOP20_FILE,
        "STRUCTURE": STRUCTURE_FILE,
    }

    for name, path in files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{name} file not found: {path}"
            )

        print(
            f"{name:<12}: OK"
        )


# ============================================================
# MARKET
# ============================================================

def load_market():

    log(
        "Loading STEP 1 market regime"
    )

    df = pd.read_csv(
        MARKET_FILE
    )

    if df.empty:
        raise ValueError(
            "market_regime.csv is empty"
        )

    score_col = find_column(
        df,
        [
            "market_score",
            "market_regime_score",
            "score",
        ],
    )

    if score_col is None:
        raise ValueError(
            "market_score column not found"
        )

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce",
    ).dropna()

    if scores.empty:
        raise ValueError(
            "No valid market score"
        )

    market_score = clamp(
        float(scores.iloc[-1])
    )

    regime_col = find_column(
        df,
        [
            "market_regime",
            "regime",
        ],
    )

    regime = (
        clean_text(df[regime_col].iloc[-1])
        if regime_col
        else "UNKNOWN"
    )

    latest = df.iloc[-1]

    directions = {}

    for key in [
        "ndx_direction",
        "spy_direction",
        "soxx_direction",
        "dia_direction",
    ]:

        column = find_column(
            df,
            [key],
        )

        if column is None:
            directions[key] = "UNAVAILABLE"
        else:
            directions[key] = clean_text(
                latest[column]
            ) or "UNAVAILABLE"

    return {
        "market_score": market_score,
        "market_regime": regime,
        **directions,
    }


# ============================================================
# FLOW
# ============================================================

def load_flow():

    log(
        "Loading unusual flow"
    )

    df = pd.read_csv(
        FLOW_FILE
    )

    if df.empty:
        raise ValueError(
            "unusual_flow.csv is empty"
        )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
        ],
    )

    if ticker_col is None:
        raise ValueError(
            "Ticker column missing in unusual_flow.csv"
        )

    flow_score_col = find_column(
        df,
        [
            "flow_score",
            "option_flow_score",
            "score",
        ],
    )

    premium_col = find_column(
        df,
        [
            "premium",
            "estimated_premium",
            "estimated_traded_premium",
            "premium_flow",
            "total_premium",
        ],
    )

    result = pd.DataFrame()

    result["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result["flow_score"] = numeric(
        df,
        flow_score_col,
    )

    result["premium"] = numeric(
        df,
        premium_col,
    )

    grouped = (
        result
        .groupby(
            "ticker",
            as_index=False,
        )
        .agg({
            "flow_score": "max",
            "premium": "sum",
        })
    )

    log(
        f"FLOW TICKERS : {len(grouped)}"
    )

    return grouped


# ============================================================
# TOP20
# ============================================================

def load_top20():

    log(
        "Loading TOP20"
    )

    df = pd.read_csv(
        TOP20_FILE
    )

    if df.empty:
        raise ValueError(
            "top20.csv is empty"
        )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
        ],
    )

    if ticker_col is None:
        raise ValueError(
            "TOP20 ticker column missing"
        )

    df["ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    numeric_fields = [
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
    ]

    for field in numeric_fields:

        col = find_column(
            df,
            [field],
        )

        if col:
            df[field] = numeric(
                df,
                col,
            )
        else:
            df[field] = np.nan

    text_fields = [
        "flow_direction",
        "estimated_direction",
        "selection_reason",
    ]

    for field in text_fields:

        col = find_column(
            df,
            [field],
        )

        if col:
            df[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )
        else:
            df[field] = ""

    df = (
        df
        .drop_duplicates(
            "ticker"
        )
        .head(20)
        .reset_index(drop=True)
    )

    log(
        f"TOP20 TICKERS : {len(df)}"
    )

    return df


# ============================================================
# STRUCTURE
# ============================================================

def load_structure():

    log(
        "Loading STEP 8 structure"
    )

    df = pd.read_csv(
        STRUCTURE_FILE
    )

    if df.empty:
        raise ValueError(
            "structure.csv is empty"
        )

    ticker_col = find_column(
        df,
        [
            "ticker",
            "symbol",
            "underlying",
            "underlying_symbol",
        ],
    )

    if ticker_col is None:
        raise ValueError(
            "Structure ticker column missing"
        )

    result = pd.DataFrame()

    result["ticker"] = (
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

        col = find_column(
            df,
            [field],
        )

        result[field] = numeric(
            df,
            col,
        )

    text_fields = [
        "structure",
        "price_location",
        "gex_structure",
        "wall_structure",
    ]

    for field in text_fields:

        col = find_column(
            df,
            [field],
        )

        if col:

            result[field] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

        else:

            result[field] = ""

    result = (
        result
        .drop_duplicates(
            "ticker"
        )
    )

    log(
        f"STRUCTURE ROWS : {len(result)}"
    )

    return result


# ============================================================
# DIRECTION SCORE
# ============================================================

def direction_score(row):

    direction = clean_text(
        row.get(
            "estimated_direction",
            ""
        )
    )

    flow_direction = clean_text(
        row.get(
            "flow_direction",
            ""
        )
    )

    directional_ratio = row.get(
        "directional_ratio",
        np.nan,
    )

    # TOP20의 estimated direction이 존재하면 우선 사용
    if "BUY" in direction:
        base = 90.0
    elif "SELL" in direction:
        base = 20.0
    elif "MIXED" in direction:
        base = 50.0
    else:
        base = 50.0

    # directional ratio가 있으면 보정
    if np.isfinite(directional_ratio):

        ratio_score = clamp(
            float(directional_ratio)
        )

        base = (
            base * 0.65
            +
            ratio_score * 0.35
        )

    # flow direction 보정
    if "CALL DOMINANT" in flow_direction:
        base += 5

    elif "PUT DOMINANT" in flow_direction:
        base -= 5

    return clamp(base)


# ============================================================
# PRICE SCORE
# ============================================================

def price_score(row):

    price = row.get(
        "current_price",
        np.nan,
    )

    support = row.get(
        "support",
        np.nan,
    )

    resistance = row.get(
        "resistance",
        np.nan,
    )

    if not np.isfinite(price):
        return 50.0

    score = 50.0

    if np.isfinite(support):

        distance = (
            price - support
        ) / price

        if 0 <= distance <= 0.03:
            score += 15

        elif distance > 0:
            score += 8

    if np.isfinite(resistance):

        distance = (
            resistance - price
        ) / price

        if 0 <= distance <= 0.03:
            score += 10

        elif distance > 0:
            score += 5

    return clamp(score)


# ============================================================
# STRUCTURE SCORE
# ============================================================

def structure_score(row):

    structure = clean_text(
        row.get(
            "structure",
            ""
        )
    )

    wall = clean_text(
        row.get(
            "wall_structure",
            ""
        )
    )

    gex = clean_text(
        row.get(
            "gex_structure",
            ""
        )
    )

    # 기본값: neutral
    score = 50.0

    if structure == "BULLISH":
        score = 90.0

    elif structure == "BEARISH":
        score = 20.0

    elif structure == "NEUTRAL":
        score = 55.0

    if wall == "BULLISH BREAKOUT":
        score += 8

    elif wall == "BEARISH BREAKDOWN":
        score -= 8

    # GEX unavailable은 중립 처리
    if gex == "POSITIVE GEX":
        score += 3

    elif gex == "NEGATIVE GEX":
        score -= 3

    return clamp(score)


# ============================================================
# INDEX ALIGNMENT
# ============================================================

def index_score(market):

    directions = [
        market["ndx_direction"],
        market["spy_direction"],
        market["soxx_direction"],
        market["dia_direction"],
    ]

    bull = sum(
        "BULL" in x
        for x in directions
    )

    bear = sum(
        "BEAR" in x
        for x in directions
    )

    neutral = len(directions) - bull - bear

    if bull >= 3 and bear == 0:
        score = 95.0

    elif bull > bear:
        score = 75.0

    elif bear >= 3 and bull == 0:
        score = 20.0

    elif bear > bull:
        score = 35.0

    else:
        score = 55.0

    return (
        clamp(score),
        bull,
        bear,
        neutral,
    )


# ============================================================
# DECISION
# ============================================================

def decide(score, direction, structure):

    # --------------------------------------------------------
    # HARD SAFETY RULE
    #
    # 방향성이 매우 약한데 단순 flow 점수 때문에
    # 진입하는 것을 방지한다.
    # --------------------------------------------------------

    if direction < 30:
        return "🟡 관망"

    if direction < 20:
        return "🔴 회피"

    if structure == "BEARISH" and direction < 55:
        return "🔴 회피"

    if score >= ENTER_THRESHOLD:

        # 진입은 최소 방향성 조건을 만족해야 한다.
        if direction >= 55:
            return "🟢 진입"

        return "🟡 관망"

    if score >= WATCH_THRESHOLD:
        return "🟡 관망"

    return "🔴 회피"


# ============================================================
# REASON
# ============================================================

def build_reason(row):

    reasons = []

    if row["flow_score"] >= 85:
        reasons.append(
            "Very high option flow"
        )

    if row["direction_score"] >= 70:
        reasons.append(
            "Bullish directional flow"
        )

    elif row["direction_score"] <= 30:
        reasons.append(
            "Weak / bearish directional flow"
        )

    structure = row["structure"]

    if structure == "BULLISH":
        reasons.append(
            "Bullish structure"
        )

    elif structure == "BEARISH":
        reasons.append(
            "Bearish structure"
        )

    else:
        reasons.append(
            "Neutral structure"
        )

    if row["index_score"] >= 75:
        reasons.append(
            "Index alignment bullish"
        )

    return " | ".join(reasons)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("STEP 9 INPUT CHECK")
    print("=" * 78)

    check_files()

    print()
    print("=" * 78)
    print("RUN STEP 9 DECISION")
    print("=" * 78)

    market = load_market()
    flow = load_flow()
    top20 = load_top20()
    structure = load_structure()

    merged = (
        top20
        .merge(
            flow,
            on="ticker",
            how="left",
            suffixes=(
                "",
                "_flow",
            ),
        )
        .merge(
            structure,
            on="ticker",
            how="left",
            suffixes=(
                "",
                "_structure",
            ),
        )
    )

    idx_score, bull, bear, neutral = index_score(
        market
    )

    rows = []

    for _, source in merged.iterrows():

        ticker = source["ticker"]

        flow_value = source.get(
            "max_flow_score",
            np.nan,
        )

        if not np.isfinite(flow_value):

            flow_value = source.get(
                "flow_score",
                np.nan,
            )

        flow_value = clamp(
            float(flow_value)
            if np.isfinite(flow_value)
            else 0
        )

        direction = direction_score(
            source
        )

        structure_value = structure_score(
            source
        )

        price_value = price_score(
            source
        )

        market_value = market["market_score"]

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (
            market_value * MARKET_WEIGHT / 100
            +
            flow_value * FLOW_WEIGHT / 100
            +
            direction * DIRECTION_WEIGHT / 100
            +
            structure_value * STRUCTURE_WEIGHT / 100
            +
            price_value * PRICE_WEIGHT / 100
            +
            idx_score * INDEX_WEIGHT / 100
        )

        final_score = clamp(
            final_score
        )

        structure_text = clean_text(
            source.get(
                "structure",
                ""
            )
        )

        decision = decide(
            final_score,
            direction,
            structure_text,
        )

        row = {
            "ticker": ticker,
            "market_score": market_value,
            "market_regime": market["market_regime"],
            "ndx_direction": market["ndx_direction"],
            "spy_direction": market["spy_direction"],
            "soxx_direction": market["soxx_direction"],
            "dia_direction": market["dia_direction"],

            "top20_score": source.get(
                "top20_score",
                np.nan,
            ),

            "max_flow_score": source.get(
                "max_flow_score",
                np.nan,
            ),

            "avg_flow_score": source.get(
                "avg_flow_score",
                np.nan,
            ),

            "total_volume": source.get(
                "total_volume",
                np.nan,
            ),

            "total_premium": source.get(
                "total_premium",
                np.nan,
            ),

            "call_premium": source.get(
                "call_premium",
                np.nan,
            ),

            "put_premium": source.get(
                "put_premium",
                np.nan,
            ),

            "call_put_imbalance": source.get(
                "call_put_imbalance",
                np.nan,
            ),

            "directional_ratio": source.get(
                "directional_ratio",
                np.nan,
            ),

            "flow_direction": source.get(
                "flow_direction",
                "",
            ),

            "estimated_direction": source.get(
                "estimated_direction",
                "",
            ),

            "top_dte": source.get(
                "top_dte",
                np.nan,
            ),

            "flow_score": flow_value,

            "effective_flow_score": flow_value,

            "current_price": source.get(
                "current_price",
                np.nan,
            ),

            "call_wall": source.get(
                "call_wall",
                np.nan,
            ),

            "put_wall": source.get(
                "put_wall",
                np.nan,
            ),

            "support": source.get(
                "support",
                np.nan,
            ),

            "resistance": source.get(
                "resistance",
                np.nan,
            ),

            "call_gex": source.get(
                "call_gex",
                np.nan,
            ),

            "put_gex": source.get(
                "put_gex",
                np.nan,
            ),

            "net_gex": source.get(
                "net_gex",
                np.nan,
            ),

            "gex_source": (
                "CALCULATED"
            ),

            "structure": structure_text,

            "price_location": source.get(
                "price_location",
                "",
            ),

            "gex_structure": source.get(
                "gex_structure",
                "",
            ),

            "wall_structure": source.get(
                "wall_structure",
                "",
            ),

            "market_component": (
                market_value
                * MARKET_WEIGHT
                / 100
            ),

            "flow_component": (
                flow_value
                * FLOW_WEIGHT
                / 100
            ),

            "direction_component": (
                direction
                * DIRECTION_WEIGHT
                / 100
            ),

            "structure_component": (
                structure_value
                * STRUCTURE_WEIGHT
                / 100
            ),

            "price_component": (
                price_value
                * PRICE_WEIGHT
                / 100
            ),

            "index_component": (
                idx_score
                * INDEX_WEIGHT
                / 100
            ),

            "direction_score": direction,

            "structure_score": structure_value,

            "price_score": price_value,

            "index_score": idx_score,

            "decision_score": final_score,

            "decision": decision,

            "reason": "",

            "data_source": "CALCULATED",
        }

        row["reason"] = build_reason(
            row
        )

        rows.append(row)

        print(
            f"[09 DECISION] {ticker} | "
            f"FLOW {flow_value:.1f} | "
            f"DIRECTION {direction:.1f} | "
            f"STRUCTURE {structure_value:.1f} | "
            f"PRICE {price_value:.1f} | "
            f"FINAL {final_score:.2f} | "
            f"{decision}"
        )

    result = pd.DataFrame(rows)

    # 최종 rank는 결정점수 기준
    result = (
        result
        .sort_values(
            "decision_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["final_rank"] = (
        result.index + 1
    )

    OUTPUT_DIR = os.path.dirname(
        OUTPUT_FILE
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"MARKET SCORE       : "
        f"{market['market_score']:.2f}"
    )

    print(
        f"MARKET REGIME      : "
        f"{market['market_regime']}"
    )

    print(
        f"TOP20 INPUT        : "
        f"{len(top20)}"
    )

    print(
        f"DECISION ROWS      : "
        f"{len(result)}"
    )

    print(
        f"UNIQUE TICKERS     : "
        f"{result['ticker'].nunique()}"
    )

    print(
        f"VALID SCORES       : "
        f"{result['decision_score'].notna().sum()}"
    )

    print(
        f"VALID DECISIONS    : "
        f"{result['decision'].notna().sum()}"
    )

    print()
    print("DECISION SUMMARY")

    print(
        result["decision"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        f"OUTPUT FILE : "
        f"{OUTPUT_FILE}"
    )

    print(
        "STEP 9 OUTPUT : OK"
    )


if __name__ == "__main__":
    main()

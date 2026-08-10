from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd


# ============================================================
# STEP 16 - CONFIDENCE / RELIABILITY REPORT
# ============================================================
#
# 목적:
#
# STEP 13
#   prediction_history.csv
#
# STEP 14
#   performance_history.csv
#
# STEP 15
#   accuracy_report.csv
#
# 를 기반으로
#
# "우리 옵션분석기의 신호가 실제로 얼마나 신뢰할 만한가?"
#
# 를 누적해서 확인한다.
#
# IMPORTANT
#
# 여기서 "신뢰도"는 미래 수익을 보장하는 확률이 아니다.
#
# 실제 과거 신호 표본 수
# + D5 적중률
# + 평균 수익률
# + MFE
# + MAE
# + 95% Wilson interval
#
# 을 이용해 현재까지 축적된 검증 증거의 수준을 보여준다.
#
# ============================================================


# ============================================================
# PATH
# ============================================================

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


PERFORMANCE_FILE = os.path.join(
    ANALYSIS_DIR,
    "performance_history.csv",
)


ACCURACY_FILE = os.path.join(
    ANALYSIS_DIR,
    "accuracy_report.csv",
)


OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "confidence_report.csv",
)


# ============================================================
# LOAD
# ============================================================

if not os.path.exists(
    PERFORMANCE_FILE
):

    raise RuntimeError(
        "performance_history.csv not found"
    )


if not os.path.exists(
    ACCURACY_FILE
):

    raise RuntimeError(
        "accuracy_report.csv not found"
    )


performance = pd.read_csv(
    PERFORMANCE_FILE
)


accuracy = pd.read_csv(
    ACCURACY_FILE
)


if performance.empty:

    raise RuntimeError(
        "performance_history.csv is empty"
    )


if accuracy.empty:

    raise RuntimeError(
        "accuracy_report.csv is empty"
    )


# ============================================================
# NORMALIZE
# ============================================================

if "ticker" in performance.columns:

    performance["ticker"] = (
        performance["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


if "decision" in performance.columns:

    performance["decision"] = (
        performance["decision"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# NUMERIC
# ============================================================

numeric_columns = [

    "decision_score",

    "d1_return",
    "d1_mfe",
    "d1_mae",

    "d3_return",
    "d3_mfe",
    "d3_mae",

    "d5_return",
    "d5_mfe",
    "d5_mae",

    "mfe",
    "mae",

]


for column in numeric_columns:

    if column in performance.columns:

        performance[column] = pd.to_numeric(
            performance[column],
            errors="coerce",
        )


# ============================================================
# HELPERS
# ============================================================

def safe_mean(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:

        return np.nan

    return float(
        values.mean()
    )


def safe_median(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:

        return np.nan

    return float(
        values.median()
    )


def safe_min(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:

        return np.nan

    return float(
        values.min()
    )


def safe_max(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:

        return np.nan

    return float(
        values.max()
    )


def completed_returns(subset):

    if "d5_return" not in subset.columns:

        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        subset["d5_return"],
        errors="coerce",
    ).dropna()


# ============================================================
# WILSON 95% CONFIDENCE INTERVAL
#
# 이것은 "미래 성공확률"이 아니다.
#
# 현재까지 관찰된 D5 적중률의
# 통계적 불확실성 범위를 표시한다.
# ============================================================

def wilson_interval(
    successes,
    trials,
    z=1.96,
):

    if trials <= 0:

        return (
            np.nan,
            np.nan,
        )


    p = successes / trials

    denominator = (
        1
        +
        (z ** 2 / trials)
    )

    centre = (
        p
        +
        (z ** 2 / (2 * trials))
    ) / denominator

    margin = (
        z
        *
        math.sqrt(
            (
                p * (1 - p)
                / trials
            )
            +
            (
                z ** 2
                / (4 * trials ** 2)
            )
        )
        / denominator
    )

    lower = max(
        0.0,
        centre - margin,
    )

    upper = min(
        1.0,
        centre + margin,
    )

    return (
        lower * 100,
        upper * 100,
    )


# ============================================================
# CONFIDENCE LEVEL
#
# 표본 수가 많아질수록 검증 근거가 강해진다.
#
# 이 값은 "매매 성공 확률"이 아니라
# "현재까지의 검증 데이터가 얼마나 쌓였는가"를 의미한다.
# ============================================================

def confidence_level(
    completed,
    hit_rate,
):

    if completed < 30:

        return (
            "⚪ 데이터 부족"
        )

    if completed < 60:

        return (
            "🟡 초기 검증"
        )

    if completed < 100:

        return (
            "🟠 검증 진행"
        )

    if completed < 200:

        return (
            "🟢 유의미한 표본"
        )

    if completed < 500:

        return (
            "🟢 높은 검증량"
        )

    return (
        "🔵 대규모 검증"
    )


# ============================================================
# GROUP METRICS
# ============================================================

def build_metrics(
    subset,
    group_type,
    group_name,
):

    if subset.empty:

        return None


    if "d5_return" not in subset.columns:

        return None


    returns = completed_returns(
        subset
    )


    completed = len(
        returns
    )


    if completed == 0:

        return None


    successes = int(
        (
            returns > 0
        ).sum()
    )


    hit_rate = (
        successes
        /
        completed
    ) * 100


    ci_low, ci_high = (
        wilson_interval(
            successes,
            completed,
        )
    )


    avg_return = safe_mean(
        returns
    )


    median_return = safe_median(
        returns
    )


    avg_mfe = safe_mean(
        subset["d5_mfe"]
    )


    median_mfe = safe_median(
        subset["d5_mfe"]
    )


    avg_mae = safe_mean(
        subset["d5_mae"]
    )


    median_mae = safe_median(
        subset["d5_mae"]
    )


    positive_mfe = pd.to_numeric(
        subset["d5_mfe"],
        errors="coerce",
    ).dropna()


    if positive_mfe.empty:

        positive_mfe_rate = np.nan

    else:

        positive_mfe_rate = (
            (
                positive_mfe > 0
            ).mean()
            * 100
        )


    return {

        "group_type":
            group_type,

        "group":
            group_name,

        "signals":
            len(subset),

        "completed_d5":
            completed,

        "wins_d5":
            successes,

        "losses_d5":
            completed - successes,

        "d5_hit_rate":
            round(
                hit_rate,
                2,
            ),

        "wilson_95_low":
            round(
                ci_low,
                2,
            ),

        "wilson_95_high":
            round(
                ci_high,
                2,
            ),

        "avg_d5_return":
            round(
                avg_return,
                2,
            ),

        "median_d5_return":
            round(
                median_return,
                2,
            ),

        "avg_d5_mfe":
            round(
                avg_mfe,
                2,
            ),

        "median_d5_mfe":
            round(
                median_mfe,
                2,
            ),

        "avg_d5_mae":
            round(
                avg_mae,
                2,
            ),

        "median_d5_mae":
            round(
                median_mae,
                2,
            ),

        "positive_mfe_rate":
            round(
                positive_mfe_rate,
                2,
            ),

        "confidence_level":
            confidence_level(
                completed,
                hit_rate,
            ),

    }


# ============================================================
# BUILD REPORT
# ============================================================

rows = []


# ============================================================
# 1. OVERALL
# ============================================================

overall = build_metrics(
    performance,
    "OVERALL",
    "ALL",
)


if overall is not None:

    rows.append(
        overall
    )


# ============================================================
# 2. DECISION
# ============================================================

if "decision" in performance.columns:

    decision_values = [

        "🟢 진입",
        "🟡 관망",
        "🔴 회피",

    ]


    for decision in decision_values:

        subset = performance[
            performance["decision"]
            == decision
        ].copy()


        if subset.empty:

            continue


        row = build_metrics(
            subset,
            "DECISION",
            decision,
        )


        if row is not None:

            rows.append(
                row
            )


# ============================================================
# 3. SCORE GROUP
# ============================================================

if "decision_score" in performance.columns:

    score_groups = [

        (
            "90-100",
            90,
            101,
        ),

        (
            "80-89.99",
            80,
            90,
        ),

        (
            "70-79.99",
            70,
            80,
        ),

        (
            "60-69.99",
            60,
            70,
        ),

        (
            "0-59.99",
            0,
            60,
        ),

    ]


    for label, minimum, maximum in score_groups:

        subset = performance[
            (
                performance[
                    "decision_score"
                ]
                >= minimum
            )
            &
            (
                performance[
                    "decision_score"
                ]
                < maximum
            )
        ].copy()


        if subset.empty:

            continue


        row = build_metrics(
            subset,
            "SCORE",
            label,
        )


        if row is not None:

            rows.append(
                row
            )


# ============================================================
# 4. TICKER
# ============================================================

if "ticker" in performance.columns:

    tickers = sorted(
        performance["ticker"]
        .dropna()
        .unique()
    )


    for ticker in tickers:

        subset = performance[
            performance["ticker"]
            == ticker
        ].copy()


        if subset.empty:

            continue


        row = build_metrics(
            subset,
            "TICKER",
            ticker,
        )


        if row is not None:

            rows.append(
                row
            )


# ============================================================
# DATAFRAME
# ============================================================

if not rows:

    raise RuntimeError(
        "No completed D5 performance data available yet."
    )


report = pd.DataFrame(
    rows
)


# ============================================================
# SORT
# ============================================================

type_order = {

    "OVERALL": 0,
    "DECISION": 1,
    "SCORE": 2,
    "TICKER": 3,

}


report["_sort"] = (
    report["group_type"]
    .map(type_order)
    .fillna(99)
)


report = (
    report
    .sort_values(
        [
            "_sort",
            "group",
        ]
    )
    .drop(
        columns=[
            "_sort"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE
# ============================================================

report.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 75)
print("🔥 STEP 16 - OPTION SCANNER RELIABILITY")
print("=" * 75)


overall_rows = report[
    report["group_type"]
    == "OVERALL"
]


if not overall_rows.empty:

    row = overall_rows.iloc[0]


    print()
    print("==========================================")
    print("📊 누적 검증 현황")
    print("==========================================")


    print(
        "전체 신호 수       :",
        int(
            row["signals"]
        )
    )


    print(
        "D5 완료 신호       :",
        int(
            row["completed_d5"]
        )
    )


    print(
        "D5 적중 수         :",
        int(
            row["wins_d5"]
        )
    )


    print(
        "D5 실패 수         :",
        int(
            row["losses_d5"]
        )
    )


    print(
        "D5 적중률          :",
        row["d5_hit_rate"],
        "%"
    )


    print(
        "95% 구간           :",
        row["wilson_95_low"],
        "~",
        row["wilson_95_high"],
        "%"
    )


    print(
        "평균 D5 수익률     :",
        row["avg_d5_return"],
        "%"
    )


    print(
        "평균 D5 MFE        :",
        row["avg_d5_mfe"],
        "%"
    )


    print(
        "평균 D5 MAE        :",
        row["avg_d5_mae"],
        "%"
    )


    print(
        "검증 수준          :",
        row["confidence_level"]
    )


# ============================================================
# DECISION PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("🔥 DECISION RELIABILITY")
print("=" * 75)


decision_report = report[
    report["group_type"]
    == "DECISION"
]


if not decision_report.empty:

    columns = [

        "group",
        "signals",
        "completed_d5",
        "d5_hit_rate",
        "wilson_95_low",
        "wilson_95_high",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",
        "confidence_level",

    ]


    print(
        decision_report[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SCORE PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("🔥 SCORE RELIABILITY")
print("=" * 75)


score_report = report[
    report["group_type"]
    == "SCORE"
]


if not score_report.empty:

    columns = [

        "group",
        "signals",
        "completed_d5",
        "d5_hit_rate",
        "wilson_95_low",
        "wilson_95_high",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",
        "confidence_level",

    ]


    print(
        score_report[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# TICKER PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("🔥 TICKER RELIABILITY")
print("=" * 75)


ticker_report = report[
    report["group_type"]
    == "TICKER"
]


if not ticker_report.empty:

    columns = [

        "group",
        "signals",
        "completed_d5",
        "d5_hit_rate",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",
        "confidence_level",

    ]


    print(
        ticker_report[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 16 OUTPUT")
print("=" * 75)


print(
    "OUTPUT :",
    OUTPUT_FILE
)


print(
    "ROWS   :",
    len(report)
)


print()
print(
    "🔥 STEP 16 CONFIDENCE REPORT : OK"
)

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ============================================================
# STEP 16 - RELIABILITY REPORT
# ============================================================
#
# 목적:
#
# STEP 13 prediction_history.csv
# STEP 14 performance_history.csv
# STEP 15 accuracy_report.csv
#
# 를 이용해서 OPTION FLOW SCANNER의
# 실제 검증 상태를 평가한다.
#
# 중요:
#
# reliability_score는 "통계적 확률"이 아니다.
#
# 실제 누적 성과를 운영상 보기 쉽게 만든
# 검증 상태 점수다.
#
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

OUTPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "reliability_report.csv",
)

OUTPUT_MD = os.path.join(
    ANALYSIS_DIR,
    "reliability_report.md",
)


# ============================================================
# CHECK FILES
# ============================================================

for file in [
    PERFORMANCE_FILE,
    ACCURACY_FILE,
]:

    if not os.path.exists(file):

        raise RuntimeError(
            f"Required file not found: {file}"
        )


# ============================================================
# LOAD
# ============================================================

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

performance["ticker"] = (
    performance["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

performance["decision"] = (
    performance["decision"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# NUMERIC
# ============================================================

numeric_columns = [
    "decision_score",
    "market_score",

    "d1_return",
    "d3_return",
    "d5_return",

    "d5_mfe",
    "d5_mae",

    "d1_mfe",
    "d1_mae",
    "d3_mfe",
    "d3_mae",
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


def mean_value(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def hit_rate(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        (values > 0).mean() * 100
    )


def safe_round(value, digits=2):

    if pd.isna(value):
        return np.nan

    return round(
        float(value),
        digits,
    )


# ============================================================
# D5 BASE DATA
# ============================================================

d5 = performance[
    performance["d5_return"].notna()
].copy()


total_signals = len(
    performance
)

d5_completed = len(
    d5
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

d5_hit = hit_rate(
    d5["d5_return"]
) if not d5.empty else np.nan


d5_avg_return = (
    mean_value(
        d5["d5_return"]
    )
    if not d5.empty
    else np.nan
)


d5_avg_mfe = (
    mean_value(
        d5["d5_mfe"]
    )
    if not d5.empty
    else np.nan
)


d5_avg_mae = (
    mean_value(
        d5["d5_mae"]
    )
    if not d5.empty
    else np.nan
)


# ============================================================
# RECENT PERFORMANCE
# ============================================================

recent = (
    d5.sort_values(
        "signal_date"
    )
    .tail(50)
    .copy()
)


recent_50_count = len(
    recent
)


recent_50_hit = (
    hit_rate(
        recent["d5_return"]
    )
    if not recent.empty
    else np.nan
)


recent_50_return = (
    mean_value(
        recent["d5_return"]
    )
    if not recent.empty
    else np.nan
)


# ============================================================
# DECISION PERFORMANCE
# ============================================================

enter = d5[
    d5["decision"]
    == "🟢 진입"
].copy()


watch = d5[
    d5["decision"]
    == "🟡 관망"
].copy()


avoid = d5[
    d5["decision"]
    == "🔴 회피"
].copy()


enter_hit = (
    hit_rate(
        enter["d5_return"]
    )
    if not enter.empty
    else np.nan
)


watch_hit = (
    hit_rate(
        watch["d5_return"]
    )
    if not watch.empty
    else np.nan
)


# 회피는 "상승을 맞췄는가"가 아니라
# 실제로 하락/비상승을 피했는지를 본다.

avoid_success = (
    float(
        (
            avoid["d5_return"]
            <= 0
        ).mean()
        * 100
    )
    if not avoid.empty
    else np.nan
)


# ============================================================
# SCORE VALIDATION
# ============================================================

score_data = d5[
    d5["decision_score"].notna()
].copy()


score_groups = []


score_ranges = [
    ("90-100", 90, 101),
    ("80-89", 80, 90),
    ("70-79", 70, 80),
    ("60-69", 60, 70),
    ("0-59", 0, 60),
]


for label, minimum, maximum in score_ranges:

    subset = score_data[
        (
            score_data["decision_score"]
            >= minimum
        )
        &
        (
            score_data["decision_score"]
            < maximum
        )
    ].copy()

    if subset.empty:
        continue

    score_groups.append(
        {
            "group": label,
            "signals": len(subset),
            "hit_rate": hit_rate(
                subset["d5_return"]
            ),
            "avg_return": mean_value(
                subset["d5_return"]
            ),
        }
    )


score_report = pd.DataFrame(
    score_groups
)


# ============================================================
# SCORE MONOTONICITY
# ============================================================
#
# 높은 Score일수록 실제 적중률이 높은가?
#
# 완벽하게 선형일 필요는 없지만
# 전체적인 방향성을 본다.
# ============================================================

score_monotonic = False


if len(score_report) >= 3:

    ordered = (
        score_report
        .copy()
    )

    order_map = {
        "0-59": 0,
        "60-69": 1,
        "70-79": 2,
        "80-89": 3,
        "90-100": 4,
    }

    ordered["order"] = (
        ordered["group"]
        .map(order_map)
    )

    ordered = (
        ordered
        .sort_values("order")
    )

    rates = (
        ordered["hit_rate"]
        .dropna()
        .tolist()
    )

    if len(rates) >= 3:

        non_decreasing = all(
            rates[i]
            <= rates[i + 1]
            for i in range(
                len(rates) - 1
            )
        )

        score_monotonic = (
            non_decreasing
        )


# ============================================================
# SAMPLE SIZE STATUS
# ============================================================

if d5_completed < 30:

    sample_status = "🔴 표본 부족"

elif d5_completed < 100:

    sample_status = "🟠 초기 검증"

elif d5_completed < 300:

    sample_status = "🟡 검증 진행"

elif d5_completed < 500:

    sample_status = "🟢 양호"

elif d5_completed < 1000:

    sample_status = "🟢 신뢰도 상승"

else:

    sample_status = "🔵 충분한 표본"


# ============================================================
# RELIABILITY SCORE
# ============================================================
#
# 운영상 평가 점수
#
# 1. 표본        20점
# 2. D5 적중률   30점
# 3. 평균수익    20점
# 4. Score 구조  15점
# 5. 최근성과    15점
#
# 총 100점
#
# "통계적 유의성"을 의미하지 않는다.
# ============================================================


reliability_score = 0.0


# ------------------------------------------------------------
# SAMPLE SCORE - 20
# ------------------------------------------------------------

if d5_completed >= 1000:
    sample_points = 20

elif d5_completed >= 500:
    sample_points = 18

elif d5_completed >= 300:
    sample_points = 15

elif d5_completed >= 100:
    sample_points = 10

elif d5_completed >= 30:
    sample_points = 5

else:
    sample_points = 0


reliability_score += sample_points


# ------------------------------------------------------------
# HIT RATE SCORE - 30
# ------------------------------------------------------------

if pd.notna(d5_hit):

    if d5_hit >= 75:
        hit_points = 30

    elif d5_hit >= 70:
        hit_points = 26

    elif d5_hit >= 65:
        hit_points = 22

    elif d5_hit >= 60:
        hit_points = 18

    elif d5_hit >= 55:
        hit_points = 12

    elif d5_hit >= 50:
        hit_points = 7

    else:
        hit_points = 0

else:

    hit_points = 0


reliability_score += hit_points


# ------------------------------------------------------------
# RETURN SCORE - 20
# ------------------------------------------------------------

if pd.notna(d5_avg_return):

    if d5_avg_return >= 5:
        return_points = 20

    elif d5_avg_return >= 3:
        return_points = 17

    elif d5_avg_return >= 2:
        return_points = 14

    elif d5_avg_return >= 1:
        return_points = 10

    elif d5_avg_return > 0:
        return_points = 6

    else:
        return_points = 0

else:

    return_points = 0


reliability_score += return_points


# ------------------------------------------------------------
# SCORE STRUCTURE - 15
# ------------------------------------------------------------

if score_monotonic:

    score_points = 15

elif len(score_report) >= 3:

    score_points = 5

else:

    score_points = 0


reliability_score += score_points


# ------------------------------------------------------------
# RECENT CONSISTENCY - 15
# ------------------------------------------------------------

if (
    pd.notna(recent_50_hit)
    and pd.notna(d5_hit)
):

    difference = (
        abs(
            recent_50_hit
            - d5_hit
        )
    )

    if difference <= 5:

        recent_points = 15

    elif difference <= 10:

        recent_points = 10

    elif difference <= 15:

        recent_points = 5

    else:

        recent_points = 0

else:

    recent_points = 0


reliability_score += recent_points


reliability_score = round(
    reliability_score,
    1,
)


# ============================================================
# RELIABILITY STATUS
# ============================================================

if d5_completed < 30:

    reliability_status = (
        "🔴 데이터 부족"
    )

elif reliability_score >= 80:

    reliability_status = (
        "🟢 강한 검증"
    )

elif reliability_score >= 70:

    reliability_status = (
        "🟢 양호"
    )

elif reliability_score >= 60:

    reliability_status = (
        "🟡 검증 진행"
    )

elif reliability_score >= 45:

    reliability_status = (
        "🟠 주의"
    )

else:

    reliability_status = (
        "🔴 낮음"
    )


# ============================================================
# SCORE TABLE
# ============================================================

score_table = ""


if not score_report.empty:

    score_table = (
        score_report
        .to_string(
            index=False
        )
    )

else:

    score_table = (
        "아직 충분한 Score 데이터 없음"
    )


# ============================================================
# CSV OUTPUT
# ============================================================

summary = pd.DataFrame([
    {
        "metric": "total_signals",
        "value": total_signals,
    },
    {
        "metric": "d5_completed",
        "value": d5_completed,
    },
    {
        "metric": "d5_hit_rate",
        "value": safe_round(
            d5_hit
        ),
    },
    {
        "metric": "d5_avg_return",
        "value": safe_round(
            d5_avg_return
        ),
    },
    {
        "metric": "d5_avg_mfe",
        "value": safe_round(
            d5_avg_mfe
        ),
    },
    {
        "metric": "d5_avg_mae",
        "value": safe_round(
            d5_avg_mae
        ),
    },
    {
        "metric": "recent_50_count",
        "value": recent_50_count,
    },
    {
        "metric": "recent_50_hit_rate",
        "value": safe_round(
            recent_50_hit
        ),
    },
    {
        "metric": "recent_50_avg_return",
        "value": safe_round(
            recent_50_return
        ),
    },
    {
        "metric": "entry_hit_rate",
        "value": safe_round(
            enter_hit
        ),
    },
    {
        "metric": "watch_hit_rate",
        "value": safe_round(
            watch_hit
        ),
    },
    {
        "metric": "avoid_success_rate",
        "value": safe_round(
            avoid_success
        ),
    },
    {
        "metric": "score_monotonic",
        "value": score_monotonic,
    },
    {
        "metric": "sample_status",
        "value": sample_status,
    },
    {
        "metric": "reliability_score",
        "value": reliability_score,
    },
    {
        "metric": "reliability_status",
        "value": reliability_status,
    },
])


summary.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# MARKDOWN REPORT
# ============================================================

md = []

md.append(
    "# 🧠 OPTION FLOW SCANNER RELIABILITY REPORT"
)

md.append("")

md.append(
    "## Overall"
)

md.append("")

md.append(
    f"- 누적 신호: **{total_signals}**"
)

md.append(
    f"- D+5 검증 완료: **{d5_completed}**"
)

md.append(
    f"- D+5 적중률: **{safe_round(d5_hit)}%**"
)

md.append(
    f"- D+5 평균 수익률: **{safe_round(d5_avg_return)}%**"
)

md.append(
    f"- 평균 MFE: **{safe_round(d5_avg_mfe)}%**"
)

md.append(
    f"- 평균 MAE: **{safe_round(d5_avg_mae)}%**"
)

md.append("")

md.append(
    "## Decision Validation"
)

md.append("")

md.append(
    f"- 🟢 진입 적중률: **{safe_round(enter_hit)}%**"
)

md.append(
    f"- 🟡 관망 적중률: **{safe_round(watch_hit)}%**"
)

md.append(
    f"- 🔴 회피 성공률: **{safe_round(avoid_success)}%**"
)

md.append("")

md.append(
    "## Score Validation"
)

md.append("")

md.append(
    "Score가 높아질수록 적중률이 "
    "단조롭게 상승하는지 확인한다."
)

md.append("")

md.append(
    f"- Score Monotonicity: "
    f"**{'✅ 확인' if score_monotonic else '❌ 미확인'}**"
)

md.append("")

md.append("```text")
md.append(score_table)
md.append("```")

md.append("")

md.append(
    "## Recent Performance"
)

md.append("")

md.append(
    f"- 최근 50개 신호: **{recent_50_count}**"
)

md.append(
    f"- 최근 50개 D+5 적중률: "
    f"**{safe_round(recent_50_hit)}%**"
)

md.append(
    f"- 최근 50개 평균 수익률: "
    f"**{safe_round(recent_50_return)}%**"
)

md.append("")

md.append(
    "## Reliability"
)

md.append("")

md.append(
    f"- 표본 상태: **{sample_status}**"
)

md.append(
    f"- Reliability Score: "
    f"**{reliability_score}/100**"
)

md.append(
    f"- 최종 상태: **{reliability_status}**"
)

md.append("")

md.append(
    "> Reliability Score는 통계적 유의성이나 "
    "미래 수익을 보장하는 확률이 아니다. "
    "누적 성과를 운영상 확인하기 위한 지표다."
)

md.append("")

with open(
    OUTPUT_MD,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "\n".join(md)
    )


# ============================================================
# CONSOLE
# ============================================================

print()
print("=" * 70)
print("🔥 STEP 16 - RELIABILITY REPORT")
print("=" * 70)

print()

print(
    "TOTAL SIGNALS       :",
    total_signals
)

print(
    "D5 COMPLETED        :",
    d5_completed
)

print(
    "D5 HIT RATE         :",
    safe_round(d5_hit),
    "%"
)

print(
    "D5 AVG RETURN       :",
    safe_round(d5_avg_return),
    "%"
)

print(
    "D5 AVG MFE          :",
    safe_round(d5_avg_mfe),
    "%"
)

print(
    "D5 AVG MAE          :",
    safe_round(d5_avg_mae),
    "%"
)

print()

print(
    "RECENT 50 HIT RATE  :",
    safe_round(recent_50_hit),
    "%"
)

print(
    "ENTRY HIT RATE      :",
    safe_round(enter_hit),
    "%"
)

print(
    "WATCH HIT RATE      :",
    safe_round(watch_hit),
    "%"
)

print(
    "AVOID SUCCESS RATE  :",
    safe_round(avoid_success),
    "%"
)

print()

print(
    "SCORE MONOTONIC     :",
    "YES"
    if score_monotonic
    else "NO"
)

print(
    "SAMPLE STATUS       :",
    sample_status
)

print(
    "RELIABILITY SCORE   :",
    reliability_score,
    "/ 100"
)

print(
    "RELIABILITY STATUS  :",
    reliability_status
)

print()

print(
    "CSV OUTPUT          :",
    OUTPUT_CSV
)

print(
    "MD OUTPUT           :",
    OUTPUT_MD
)

print()

print(
    "STEP 16 OUTPUT : OK"
)

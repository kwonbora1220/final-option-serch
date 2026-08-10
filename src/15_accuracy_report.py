```python
from __future__ import annotations

import os
import pandas as pd
import numpy as np


# ============================================================
# STEP 15 - ACCURACY REPORT
#
# 목적:
# ------------------------------------------------------------
# STEP 13 prediction_history.csv
# STEP 14 performance_history.csv
#
# 를 이용하여 옵션분석기의 실제 성과를 누적 분석한다.
#
# 분석 항목:
#
# 1. D+1 / D+3 / D+5 Hit Rate
# 2. D+1 / D+3 / D+5 Average Return
# 3. MFE = Maximum Favorable Excursion
# 4. MAE = Maximum Adverse Excursion
# 5. Decision별 성과
# 6. Score 구간별 성과
# 7. Market Regime별 성과
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
    "analysis"
)

PERFORMANCE_FILE = os.path.join(
    ANALYSIS_DIR,
    "performance_history.csv"
)

PREDICTION_FILE = os.path.join(
    ANALYSIS_DIR,
    "prediction_history.csv"
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "accuracy_report.csv"
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("🔥 STEP 15 - ACCURACY REPORT")
print("=" * 70)
print()


# ============================================================
# LOAD PERFORMANCE HISTORY
# ============================================================

if not os.path.exists(
    PERFORMANCE_FILE
):

    raise RuntimeError(
        "performance_history.csv not found:\n"
        + PERFORMANCE_FILE
    )


df = pd.read_csv(
    PERFORMANCE_FILE
)


if df.empty:

    raise RuntimeError(
        "performance_history.csv is empty"
    )


print(
    "PERFORMANCE ROWS :",
    len(df)
)


# ============================================================
# OPTIONAL PREDICTION HISTORY
#
# performance_history가 prediction_history를
# 이미 포함하고 있지 않은 경우 보완한다.
# ============================================================

if os.path.exists(
    PREDICTION_FILE
):

    try:

        prediction = pd.read_csv(
            PREDICTION_FILE
        )

        if not prediction.empty:

            print(
                "PREDICTION ROWS  :",
                len(prediction)
            )

    except Exception:

        prediction = None

else:

    prediction = None


# ============================================================
# BASIC COLUMN CHECK
# ============================================================

required_columns = [
    "ticker",
    "decision"
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [

    "decision_score",

    "market_score",
    "flow_score",

    "signal_price",

    "d1_price",
    "d1_return",
    "d1_mfe",
    "d1_mae",

    "d3_price",
    "d3_return",
    "d3_mfe",
    "d3_mae",

    "d5_price",
    "d5_return",
    "d5_mfe",
    "d5_mae",

    "mfe",
    "mae",

]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# HIT RATE FUNCTION
# ============================================================

def hit_rate(
    series: pd.Series
) -> float:

    if series is None:
        return np.nan

    series = series.dropna()

    if len(series) == 0:
        return np.nan

    return (
        (
            series > 0
        ).mean()
        * 100
    )


# ============================================================
# MEAN FUNCTION
# ============================================================

def safe_mean(
    series: pd.Series
) -> float:

    if series is None:
        return np.nan

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return np.nan

    return float(
        series.mean()
    )


# ============================================================
# MEDIAN FUNCTION
# ============================================================

def safe_median(
    series: pd.Series
) -> float:

    if series is None:
        return np.nan

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return np.nan

    return float(
        series.median()
    )


# ============================================================
# BUILD METRICS
# ============================================================

def build_metrics(
    subset: pd.DataFrame
) -> dict:

    result = {

        "signals":
            len(subset),

        # ----------------------------------------------------
        # D+1
        # ----------------------------------------------------

        "d1_completed":
            (
                subset["d1_return"]
                .notna()
                .sum()
                if "d1_return" in subset.columns
                else 0
            ),

        "d1_hit_rate":
            (
                hit_rate(
                    subset["d1_return"]
                )
                if "d1_return" in subset.columns
                else np.nan
            ),

        "avg_d1_return":
            (
                safe_mean(
                    subset["d1_return"]
                )
                if "d1_return" in subset.columns
                else np.nan
            ),

        "avg_d1_mfe":
            (
                safe_mean(
                    subset["d1_mfe"]
                )
                if "d1_mfe" in subset.columns
                else np.nan
            ),

        "avg_d1_mae":
            (
                safe_mean(
                    subset["d1_mae"]
                )
                if "d1_mae" in subset.columns
                else np.nan
            ),

        # ----------------------------------------------------
        # D+3
        # ----------------------------------------------------

        "d3_completed":
            (
                subset["d3_return"]
                .notna()
                .sum()
                if "d3_return" in subset.columns
                else 0
            ),

        "d3_hit_rate":
            (
                hit_rate(
                    subset["d3_return"]
                )
                if "d3_return" in subset.columns
                else np.nan
            ),

        "avg_d3_return":
            (
                safe_mean(
                    subset["d3_return"]
                )
                if "d3_return" in subset.columns
                else np.nan
            ),

        "avg_d3_mfe":
            (
                safe_mean(
                    subset["d3_mfe"]
                )
                if "d3_mfe" in subset.columns
                else np.nan
            ),

        "avg_d3_mae":
            (
                safe_mean(
                    subset["d3_mae"]
                )
                if "d3_mae" in subset.columns
                else np.nan
            ),

        # ----------------------------------------------------
        # D+5
        # ----------------------------------------------------

        "d5_completed":
            (
                subset["d5_return"]
                .notna()
                .sum()
                if "d5_return" in subset.columns
                else 0
            ),

        "d5_hit_rate":
            (
                hit_rate(
                    subset["d5_return"]
                )
                if "d5_return" in subset.columns
                else np.nan
            ),

        "avg_d5_return":
            (
                safe_mean(
                    subset["d5_return"]
                )
                if "d5_return" in subset.columns
                else np.nan
            ),

        "median_d5_return":
            (
                safe_median(
                    subset["d5_return"]
                )
                if "d5_return" in subset.columns
                else np.nan
            ),

        "avg_d5_mfe":
            (
                safe_mean(
                    subset["d5_mfe"]
                )
                if "d5_mfe" in subset.columns
                else np.nan
            ),

        "median_d5_mfe":
            (
                safe_median(
                    subset["d5_mfe"]
                )
                if "d5_mfe" in subset.columns
                else np.nan
            ),

        "avg_d5_mae":
            (
                safe_mean(
                    subset["d5_mae"]
                )
                if "d5_mae" in subset.columns
                else np.nan
            ),

        "median_d5_mae":
            (
                safe_median(
                    subset["d5_mae"]
                )
                if "d5_mae" in subset.columns
                else np.nan
            ),

        # ----------------------------------------------------
        # Overall MFE / MAE
        # ----------------------------------------------------

        "avg_mfe":
            (
                safe_mean(
                    subset["mfe"]
                )
                if "mfe" in subset.columns
                else np.nan
            ),

        "avg_mae":
            (
                safe_mean(
                    subset["mae"]
                )
                if "mae" in subset.columns
                else np.nan
            ),

    }

    return result


# ============================================================
# REPORT ROWS
# ============================================================

report_rows = []


# ============================================================
# 1. OVERALL
# ============================================================

overall = build_metrics(
    df
)

overall["category"] = "OVERALL"
overall["group"] = "ALL"

report_rows.append(
    overall
)


# ============================================================
# 2. DECISION
# ============================================================

print()
print("=" * 70)
print("📊 DECISION PERFORMANCE")
print("=" * 70)


for decision in [
    "🟢 진입",
    "🟡 관망",
    "🔴 회피",
]:

    subset = df[
        df["decision"]
        .astype(str)
        .str.strip()
        == decision
    ].copy()


    if subset.empty:
        continue


    metrics = build_metrics(
        subset
    )

    metrics["category"] = "DECISION"
    metrics["group"] = decision

    report_rows.append(
        metrics
    )


# ============================================================
# 3. SCORE GROUP
# ============================================================

print()
print("=" * 70)
print("📊 SCORE PERFORMANCE")
print("=" * 70)


if "decision_score" in df.columns:

    score_groups = [

        (
            "90-100",
            90,
            100
        ),

        (
            "80-89.99",
            80,
            90
        ),

        (
            "70-79.99",
            70,
            80
        ),

        (
            "60-69.99",
            60,
            70
        ),

        (
            "0-59.99",
            0,
            60
        ),

    ]


    for label, minimum, maximum in score_groups:

        subset = df[
            (
                df["decision_score"]
                >= minimum
            )
            &
            (
                df["decision_score"]
                < maximum
            )
        ].copy()


        if subset.empty:
            continue


        metrics = build_metrics(
            subset
        )

        metrics["category"] = "SCORE"
        metrics["group"] = label

        report_rows.append(
            metrics
        )


# ============================================================
# 4. MARKET REGIME
# ============================================================

print()
print("=" * 70)
print("📊 MARKET REGIME PERFORMANCE")
print("=" * 70)


regime_column = None


for candidate in [
    "market_regime",
    "regime",
    "market_state",
]:

    if candidate in df.columns:

        regime_column = candidate
        break


if regime_column:

    regimes = (
        df[
            regime_column
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )


    for regime in sorted(
        regimes
    ):

        subset = df[
            df[
                regime_column
            ]
            .astype(str)
            .str.strip()
            == regime
        ].copy()


        if subset.empty:
            continue


        metrics = build_metrics(
            subset
        )

        metrics["category"] = (
            "MARKET_REGIME"
        )

        metrics["group"] = regime

        report_rows.append(
            metrics
        )


# ============================================================
# 5. TICKER PERFORMANCE
# ============================================================

print()
print("=" * 70)
print("📊 TICKER PERFORMANCE")
print("=" * 70)


ticker_groups = (
    df["ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .str.strip()
    .unique()
)


for ticker in sorted(
    ticker_groups
):

    subset = df[
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        == ticker
    ].copy()


    if subset.empty:
        continue


    metrics = build_metrics(
        subset
    )

    metrics["category"] = "TICKER"
    metrics["group"] = ticker

    report_rows.append(
        metrics
    )


# ============================================================
# CREATE REPORT
# ============================================================

report = pd.DataFrame(
    report_rows
)


# ============================================================
# ROUND NUMBERS
# ============================================================

numeric_report_columns = [

    "d1_hit_rate",
    "avg_d1_return",
    "avg_d1_mfe",
    "avg_d1_mae",

    "d3_hit_rate",
    "avg_d3_return",
    "avg_d3_mfe",
    "avg_d3_mae",

    "d5_hit_rate",
    "avg_d5_return",
    "median_d5_return",

    "avg_d5_mfe",
    "median_d5_mfe",

    "avg_d5_mae",
    "median_d5_mae",

    "avg_mfe",
    "avg_mae",

]


for column in numeric_report_columns:

    if column in report.columns:

        report[column] = pd.to_numeric(
            report[column],
            errors="coerce"
        ).round(2)


# ============================================================
# COLUMN ORDER
# ============================================================

preferred_order = [

    "category",
    "group",

    "signals",

    "d1_completed",
    "d1_hit_rate",
    "avg_d1_return",
    "avg_d1_mfe",
    "avg_d1_mae",

    "d3_completed",
    "d3_hit_rate",
    "avg_d3_return",
    "avg_d3_mfe",
    "avg_d3_mae",

    "d5_completed",
    "d5_hit_rate",
    "avg_d5_return",
    "median_d5_return",

    "avg_d5_mfe",
    "median_d5_mfe",

    "avg_d5_mae",
    "median_d5_mae",

    "avg_mfe",
    "avg_mae",

]


existing_order = [
    column
    for column in preferred_order
    if column in report.columns
]


remaining_columns = [
    column
    for column in report.columns
    if column not in existing_order
]


report = report[
    existing_order
    + remaining_columns
]


# ============================================================
# SAVE
# ============================================================

report.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("🔥 OVERALL SYSTEM PERFORMANCE")
print("=" * 70)
print()


overall_row = report[
    (
        report["category"]
        == "OVERALL"
    )
    &
    (
        report["group"]
        == "ALL"
    )
]


if not overall_row.empty:

    row = overall_row.iloc[0]


    print(
        "TOTAL SIGNALS :",
        int(
            row["signals"]
        )
    )


    print()


    print(
        "D+1 HIT RATE  :",
        row["d1_hit_rate"],
        "%"
    )


    print(
        "D+3 HIT RATE  :",
        row["d3_hit_rate"],
        "%"
    )


    print(
        "D+5 HIT RATE  :",
        row["d5_hit_rate"],
        "%"
    )


    print()


    print(
        "AVG D+1 RETURN :",
        row["avg_d1_return"],
        "%"
    )


    print(
        "AVG D+3 RETURN :",
        row["avg_d3_return"],
        "%"
    )


    print(
        "AVG D+5 RETURN :",
        row["avg_d5_return"],
        "%"
    )


    print()


    print(
        "AVG MFE        :",
        row["avg_mfe"],
        "%"
    )


    print(
        "AVG MAE        :",
        row["avg_mae"],
        "%"
    )


# ============================================================
# DECISION SUMMARY
# ============================================================

print()
print("=" * 70)
print("🔥 DECISION PERFORMANCE")
print("=" * 70)
print()


decision_report = report[
    report["category"]
    == "DECISION"
].copy()


if not decision_report.empty:

    print(
        decision_report[
            [
                "group",
                "signals",
                "d5_hit_rate",
                "avg_d5_return",
                "avg_d5_mfe",
                "avg_d5_mae",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# SCORE SUMMARY
# ============================================================

print()
print("=" * 70)
print("🔥 SCORE PERFORMANCE")
print("=" * 70)
print()


score_report = report[
    report["category"]
    == "SCORE"
].copy()


if not score_report.empty:

    print(
        score_report[
            [
                "group",
                "signals",
                "d5_hit_rate",
                "avg_d5_return",
                "avg_d5_mfe",
                "avg_d5_mae",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# MARKET REGIME SUMMARY
# ============================================================

print()
print("=" * 70)
print("🔥 MARKET REGIME PERFORMANCE")
print("=" * 70)
print()


regime_report = report[
    report["category"]
    == "MARKET_REGIME"
].copy()


if not regime_report.empty:

    print(
        regime_report[
            [
                "group",
                "signals",
                "d5_hit_rate",
                "avg_d5_return",
                "avg_d5_mfe",
                "avg_d5_mae",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 15 OUTPUT")
print("=" * 70)

print(
    "OUTPUT FILE :",
    OUTPUT_FILE
)

print(
    "REPORT ROWS :",
    len(report)
)

print()

print("STEP 15 OUTPUT : OK")
```

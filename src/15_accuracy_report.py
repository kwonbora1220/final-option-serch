from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ============================================================
# STEP 15 - ACCURACY REPORT
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


INPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "performance_history.csv",
)

OUTPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "accuracy_report.csv",
)


# ============================================================
# LOAD
# ============================================================


if not os.path.exists(
    INPUT_FILE
):

    raise RuntimeError(
        "performance_history.csv not found"
    )


df = pd.read_csv(
    INPUT_FILE
)


if df.empty:

    raise RuntimeError(
        "performance_history.csv is empty"
    )


# ============================================================
# NORMALIZE
# ============================================================


df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)


if "decision" not in df.columns:

    raise RuntimeError(
        "decision column missing"
    )


# ============================================================
# NUMERIC
# ============================================================


numeric_columns = [

    "decision_score",
    "market_score",

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

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
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


def hit_rate(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


    if values.empty:

        return np.nan


    return float(
        (
            values > 0
        ).mean()
        * 100
    )


def metrics(subset):

    result = {}

    result["signals"] = len(
        subset
    )


    # --------------------------------------------------------
    # D1
    # --------------------------------------------------------

    if "d1_return" in subset.columns:

        result["d1_completed"] = (
            subset["d1_return"]
            .notna()
            .sum()
        )

        result["d1_hit_rate"] = (
            hit_rate(
                subset["d1_return"]
            )
        )

        result["avg_d1_return"] = (
            safe_mean(
                subset["d1_return"]
            )
        )

        result["avg_d1_mfe"] = (
            safe_mean(
                subset["d1_mfe"]
            )
        )

        result["avg_d1_mae"] = (
            safe_mean(
                subset["d1_mae"]
            )
        )


    # --------------------------------------------------------
    # D3
    # --------------------------------------------------------

    if "d3_return" in subset.columns:

        result["d3_completed"] = (
            subset["d3_return"]
            .notna()
            .sum()
        )

        result["d3_hit_rate"] = (
            hit_rate(
                subset["d3_return"]
            )
        )

        result["avg_d3_return"] = (
            safe_mean(
                subset["d3_return"]
            )
        )

        result["avg_d3_mfe"] = (
            safe_mean(
                subset["d3_mfe"]
            )
        )

        result["avg_d3_mae"] = (
            safe_mean(
                subset["d3_mae"]
            )
        )


    # --------------------------------------------------------
    # D5
    # --------------------------------------------------------

    if "d5_return" in subset.columns:

        result["d5_completed"] = (
            subset["d5_return"]
            .notna()
            .sum()
        )

        result["d5_hit_rate"] = (
            hit_rate(
                subset["d5_return"]
            )
        )

        result["avg_d5_return"] = (
            safe_mean(
                subset["d5_return"]
            )
        )

        result["median_d5_return"] = (
            safe_median(
                subset["d5_return"]
            )
        )

        result["avg_d5_mfe"] = (
            safe_mean(
                subset["d5_mfe"]
            )
        )

        result["median_d5_mfe"] = (
            safe_median(
                subset["d5_mfe"]
            )
        )

        result["avg_d5_mae"] = (
            safe_mean(
                subset["d5_mae"]
            )
        )

        result["median_d5_mae"] = (
            safe_median(
                subset["d5_mae"]
            )
        )


    return result


# ============================================================
# REPORT
# ============================================================


rows = []


# ============================================================
# OVERALL
# ============================================================


overall = metrics(
    df
)

overall["category"] = "OVERALL"
overall["group"] = "ALL"

rows.append(
    overall
)


# ============================================================
# DECISION
# ============================================================


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


    row = metrics(
        subset
    )

    row["category"] = (
        "DECISION"
    )

    row["group"] = decision

    rows.append(
        row
    )


# ============================================================
# SCORE GROUP
# ============================================================


if "decision_score" in df.columns:

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


        row = metrics(
            subset
        )

        row["category"] = (
            "SCORE"
        )

        row["group"] = label

        rows.append(
            row
        )


# ============================================================
# MARKET REGIME
# ============================================================


if "market_regime" in df.columns:

    regimes = (
        df["market_regime"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )


    for regime in sorted(
        regimes
    ):

        if not regime:

            continue


        subset = df[
            df["market_regime"]
            .astype(str)
            .str.strip()
            == regime
        ].copy()


        if subset.empty:

            continue


        row = metrics(
            subset
        )

        row["category"] = (
            "MARKET_REGIME"
        )

        row["group"] = regime

        rows.append(
            row
        )


# ============================================================
# TICKER
# ============================================================


for ticker in sorted(
    df["ticker"]
    .dropna()
    .unique()
):

    subset = df[
        df["ticker"]
        == ticker
    ].copy()


    if subset.empty:

        continue


    row = metrics(
        subset
    )

    row["category"] = (
        "TICKER"
    )

    row["group"] = ticker

    rows.append(
        row
    )


# ============================================================
# DATAFRAME
# ============================================================


report = pd.DataFrame(
    rows
)


# ============================================================
# ROUND
# ============================================================


for column in report.columns:

    if column in [
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

    ]:

        report[column] = pd.to_numeric(
            report[column],
            errors="coerce",
        ).round(2)


# ============================================================
# SAVE
# ============================================================


report.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================


print()
print("=" * 70)
print("🔥 STEP 15 - ACCURACY REPORT")
print("=" * 70)


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

    print()
    print(
        "TOTAL SIGNALS :",
        int(
            row["signals"]
        )
    )

    print(
        "D1 COMPLETED  :",
        int(
            row["d1_completed"]
        )
    )

    print(
        "D3 COMPLETED  :",
        int(
            row["d3_completed"]
        )
    )

    print(
        "D5 COMPLETED  :",
        int(
            row["d5_completed"]
        )
    )

    print()

    print(
        "D1 HIT RATE   :",
        row["d1_hit_rate"],
        "%"
    )

    print(
        "D3 HIT RATE   :",
        row["d3_hit_rate"],
        "%"
    )

    print(
        "D5 HIT RATE   :",
        row["d5_hit_rate"],
        "%"
    )

    print()

    print(
        "D5 AVG RETURN :",
        row["avg_d5_return"],
        "%"
    )

    print(
        "D5 AVG MFE    :",
        row["avg_d5_mfe"],
        "%"
    )

    print(
        "D5 AVG MAE    :",
        row["avg_d5_mae"],
        "%"
    )


# ============================================================
# DECISION OUTPUT
# ============================================================


print()
print("=" * 70)
print("🔥 DECISION PERFORMANCE")
print("=" * 70)


decision_report = report[
    report["category"]
    == "DECISION"
]


if not decision_report.empty:

    columns = [

        "group",
        "signals",
        "d5_completed",
        "d5_hit_rate",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",

    ]

    print(
        decision_report[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SCORE OUTPUT
# ============================================================


print()
print("=" * 70)
print("🔥 SCORE PERFORMANCE")
print("=" * 70)


score_report = report[
    report["category"]
    == "SCORE"
]


if not score_report.empty:

    columns = [

        "group",
        "signals",
        "d5_completed",
        "d5_hit_rate",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",

    ]

    print(
        score_report[
            columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# MARKET OUTPUT
# ============================================================


print()
print("=" * 70)
print("🔥 MARKET REGIME PERFORMANCE")
print("=" * 70)


market_report = report[
    report["category"]
    == "MARKET_REGIME"
]


if not market_report.empty:

    columns = [

        "group",
        "signals",
        "d5_completed",
        "d5_hit_rate",
        "avg_d5_return",
        "avg_d5_mfe",
        "avg_d5_mae",

    ]

    print(
        market_report[
            columns
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
    "OUTPUT :",
    OUTPUT_FILE
)

print(
    "ROWS   :",
    len(report)
)

print()

print(
    "STEP 15 OUTPUT : OK"
)

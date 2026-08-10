from __future__ import annotations

import os

import numpy as np
import pandas as pd


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

df = pd.read_csv(
    INPUT_FILE
)

if df.empty:
    raise RuntimeError(
        "performance_history.csv is empty"
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


def hit_rate(series):

    values = (
        series
        .dropna()
        .astype(bool)
    )

    if values.empty:
        return np.nan

    return float(
        values.mean()
        * 100.0
    )


# ============================================================
# REPORT BUILDER
# ============================================================

rows = []


# ============================================================
# OVERALL
# ============================================================

rows.append({

    "category": "OVERALL",

    "group": "ALL",

    "signals": len(df),

    "d1_hit_rate":
        hit_rate(
            df["hit_d1"]
        ),

    "d3_hit_rate":
        hit_rate(
            df["hit_d3"]
        ),

    "d5_hit_rate":
        hit_rate(
            df["hit_d5"]
        ),

    "avg_d1_return":
        safe_mean(
            df["d1_return"]
        ),

    "avg_d3_return":
        safe_mean(
            df["d3_return"]
        ),

    "avg_d5_return":
        safe_mean(
            df["d5_return"]
        ),

})


# ============================================================
# DECISION
# ============================================================

if "decision" in df.columns:

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
        ]

        if subset.empty:
            continue

        rows.append({

            "category": "DECISION",

            "group": decision,

            "signals": len(subset),

            "d1_hit_rate":
                hit_rate(
                    subset["hit_d1"]
                ),

            "d3_hit_rate":
                hit_rate(
                    subset["hit_d3"]
                ),

            "d5_hit_rate":
                hit_rate(
                    subset["hit_d5"]
                ),

            "avg_d1_return":
                safe_mean(
                    subset["d1_return"]
                ),

            "avg_d3_return":
                safe_mean(
                    subset["d3_return"]
                ),

            "avg_d5_return":
                safe_mean(
                    subset["d5_return"]
                ),

        })


# ============================================================
# SCORE BUCKETS
# ============================================================

df["decision_score_num"] = pd.to_numeric(
    df["decision_score"],
    errors="coerce",
)


score_ranges = [
    ("90-100", 90, 100),
    ("80-89", 80, 90),
    ("70-79", 70, 80),
    ("60-69", 60, 70),
    ("0-59", 0, 60),
]


for label, low, high in score_ranges:

    subset = df[
        (
            df["decision_score_num"]
            >= low
        )
        &
        (
            df["decision_score_num"]
            < high
        )
    ]

    if subset.empty:
        continue

    rows.append({

        "category": "SCORE",

        "group": label,

        "signals": len(subset),

        "d1_hit_rate":
            hit_rate(
                subset["hit_d1"]
            ),

        "d3_hit_rate":
            hit_rate(
                subset["hit_d3"]
            ),

        "d5_hit_rate":
            hit_rate(
                subset["hit_d5"]
            ),

        "avg_d1_return":
            safe_mean(
                subset["d1_return"]
            ),

        "avg_d3_return":
            safe_mean(
                subset["d3_return"]
            ),

        "avg_d5_return":
            safe_mean(
                subset["d5_return"]
            ),

    })


# ============================================================
# MARKET REGIME
# ============================================================

if "market_regime" in df.columns:

    for regime in sorted(
        df["market_regime"]
        .dropna()
        .astype(str)
        .unique()
    ):

        subset = df[
            df["market_regime"]
            .astype(str)
            == regime
        ]

        if subset.empty:
            continue

        rows.append({

            "category":
                "MARKET_REGIME",

            "group":
                regime,

            "signals":
                len(subset),

            "d1_hit_rate":
                hit_rate(
                    subset["hit_d1"]
                ),

            "d3_hit_rate":
                hit_rate(
                    subset["hit_d3"]
                ),

            "d5_hit_rate":
                hit_rate(
                    subset["hit_d5"]
                ),

            "avg_d1_return":
                safe_mean(
                    subset["d1_return"]
                ),

            "avg_d3_return":
                safe_mean(
                    subset["d3_return"]
                ),

            "avg_d5_return":
                safe_mean(
                    subset["d5_return"]
                ),

        })


# ============================================================
# SAVE
# ============================================================

report = pd.DataFrame(
    rows
)

report.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "=========================================="
)

print(
    "STEP 15 - ACCURACY REPORT"
)

print(
    "=========================================="
)

print(
    report.to_string(
        index=False
    )
)

print()

print(
    "OUTPUT :",
    OUTPUT_FILE,
)

print()

print(
    "STEP 15 OUTPUT : OK"
)

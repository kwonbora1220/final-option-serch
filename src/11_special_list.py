import os
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# STEP 10 - FINAL REPORT
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

INPUT_FILE = os.path.join(
    ANALYSIS_DIR,
    "decision.csv"
)

OUTPUT_CSV = os.path.join(
    ANALYSIS_DIR,
    "final_report.csv"
)

OUTPUT_MD = os.path.join(
    ANALYSIS_DIR,
    "final_report.md"
)


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
        f"[10 FINAL REPORT] {now} | {message}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    log("START")

    # --------------------------------------------------------
    # INPUT CHECK
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"Decision file not found: {INPUT_FILE}"
        )

    log(
        f"INPUT : {INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "decision.csv is empty"
        )

    log(
        f"INPUT ROWS : {len(df)}"
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [

        "ticker",
        "market_score",
        "market_regime",

        "ndx_direction",
        "spy_direction",
        "soxx_direction",
        "dia_direction",

        "flow_score",

        "current_price",

        "call_wall",
        "put_wall",

        "support",
        "resistance",

        "net_gex",

        "structure",

        "decision_score",
        "decision",

        "reason",

    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # NUMERIC STANDARDIZATION
    # --------------------------------------------------------

    numeric_columns = [

        "market_score",
        "flow_score",
        "current_price",

        "call_wall",
        "put_wall",

        "support",
        "resistance",

        "net_gex",

        "decision_score",

    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # FINAL SORT
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "decision_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    df["final_rank"] = (
        df.index + 1
    )

    # --------------------------------------------------------
    # FINAL REPORT COLUMNS
    # --------------------------------------------------------

    report_columns = [

        "final_rank",
        "ticker",

        "market_score",
        "market_regime",

        "ndx_direction",
        "spy_direction",
        "soxx_direction",
        "dia_direction",

        "flow_score",

        "current_price",

        "call_wall",
        "put_wall",

        "support",
        "resistance",

        "net_gex",

        "structure",

        "decision_score",
        "decision",

        "reason",

    ]

    report = df[
        report_columns
    ].copy()

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    os.makedirs(
        ANALYSIS_DIR,
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_CSV,
        index=False
    )

    log(
        f"CSV SAVED : {OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # MARKET SUMMARY
    # --------------------------------------------------------

    latest = report.iloc[0]

    market_score = float(
        report["market_score"]
        .dropna()
        .iloc[0]
    )

    market_regime = str(
        report["market_regime"]
        .iloc[0]
    )

    ndx_direction = str(
        report["ndx_direction"]
        .iloc[0]
    )

    spy_direction = str(
        report["spy_direction"]
        .iloc[0]
    )

    soxx_direction = str(
        report["soxx_direction"]
        .iloc[0]
    )

    dia_direction = str(
        report["dia_direction"]
        .iloc[0]
    )

    # --------------------------------------------------------
    # DECISION COUNTS
    # --------------------------------------------------------

    entry_count = (
        report["decision"]
        == "🟢 진입"
    ).sum()

    watch_count = (
        report["decision"]
        == "🟡 관망"
    ).sum()

    avoid_count = (
        report["decision"]
        == "🔴 회피"
    ).sum()

    # --------------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------------

    generated_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    lines = []

    lines.append(
        "# 🇺🇸 US OPTIONS FINAL REPORT"
    )

    lines.append("")

    lines.append(
        f"Generated: {generated_at}"
    )

    lines.append("")

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    lines.append(
        "## 1. MARKET REGIME"
    )

    lines.append("")

    lines.append(
        f"- Market Score: **{market_score:.2f}**"
    )

    lines.append(
        f"- Market Regime: **{market_regime}**"
    )

    lines.append(
        f"- NDX: **{ndx_direction}**"
    )

    lines.append(
        f"- SPY: **{spy_direction}**"
    )

    lines.append(
        f"- SOXX: **{soxx_direction}**"
    )

    lines.append(
        f"- DIA: **{dia_direction}**"
    )

    lines.append("")

    # --------------------------------------------------------
    # DECISION SUMMARY
    # --------------------------------------------------------

    lines.append(
        "## 2. DECISION SUMMARY"
    )

    lines.append("")

    lines.append(
        f"- 🟢 진입: **{entry_count}**"
    )

    lines.append(
        f"- 🟡 관망: **{watch_count}**"
    )

    lines.append(
        f"- 🔴 회피: **{avoid_count}**"
    )

    lines.append("")

    # --------------------------------------------------------
    # TOP DECISIONS
    # --------------------------------------------------------

    lines.append(
        "## 3. TOP DECISIONS"
    )

    lines.append("")

    lines.append(
        "| Rank | Ticker | Decision Score | Flow | Price | Decision |"
    )

    lines.append(
        "|---:|---|---:|---:|---:|---|"
    )

    for _, row in report.iterrows():

        price = row["current_price"]

        if pd.isna(price):

            price_text = "-"

        else:

            price_text = f"{price:.2f}"

        flow = row["flow_score"]

        if pd.isna(flow):

            flow_text = "-"

        else:

            flow_text = f"{flow:.2f}"

        lines.append(
            f"| {int(row['final_rank'])} "
            f"| {row['ticker']} "
            f"| {row['decision_score']:.2f} "
            f"| {flow_text} "
            f"| {price_text} "
            f"| {row['decision']} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # DETAILED ANALYSIS
    # --------------------------------------------------------

    lines.append(
        "## 4. DETAILED ANALYSIS"
    )

    lines.append("")

    for _, row in report.iterrows():

        ticker = row["ticker"]

        lines.append(
            f"### {int(row['final_rank'])}. {ticker}"
        )

        lines.append("")

        lines.append(
            f"- Decision: **{row['decision']}**"
        )

        lines.append(
            f"- Decision Score: **{row['decision_score']:.2f}**"
        )

        if not pd.isna(
            row["flow_score"]
        ):

            lines.append(
                f"- Flow Score: **{row['flow_score']:.2f}**"
            )

        if not pd.isna(
            row["current_price"]
        ):

            lines.append(
                f"- Current Price: **{row['current_price']:.2f}**"
            )

        if not pd.isna(
            row["call_wall"]
        ):

            lines.append(
                f"- Call Wall: **{row['call_wall']:.2f}**"
            )

        if not pd.isna(
            row["put_wall"]
        ):

            lines.append(
                f"- Put Wall: **{row['put_wall']:.2f}**"
            )

        if not pd.isna(
            row["support"]
        ):

            lines.append(
                f"- Support: **{row['support']:.2f}**"
            )

        if not pd.isna(
            row["resistance"]
        ):

            lines.append(
                f"- Resistance: **{row['resistance']:.2f}**"
            )

        if not pd.isna(
            row["net_gex"]
        ):

            lines.append(
                f"- Net GEX: **{row['net_gex']:.4e}**"
            )

        lines.append(
            f"- Structure: **{row['structure']}**"
        )

        lines.append(
            f"- Reason: {row['reason']}"
        )

        lines.append("")

    # --------------------------------------------------------
    # WEIGHT INFORMATION
    # --------------------------------------------------------

    lines.append(
        "## 5. DECISION ENGINE"
    )

    lines.append("")

    lines.append(
        "| Component | Weight |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        "| Market | 25% |"
    )

    lines.append(
        "| Flow | 30% |"
    )

    lines.append(
        "| Structure | 20% |"
    )

    lines.append(
        "| Price / Wall | 15% |"
    )

    lines.append(
        "| Index | 10% |"
    )

    lines.append("")

    lines.append(
        "### Decision Threshold"
    )

    lines.append("")

    lines.append(
        "- 🟢 ENTRY: **>= 75**"
    )

    lines.append(
        "- 🟡 WATCH: **55 - 74.99**"
    )

    lines.append(
        "- 🔴 AVOID: **< 55**"
    )

    lines.append("")

    # --------------------------------------------------------
    # SAVE MARKDOWN
    # --------------------------------------------------------

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    log(
        f"MARKDOWN SAVED : {OUTPUT_MD}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()

    print("=" * 72)
    print("🔎 STEP 10 VALIDATION")
    print("=" * 72)

    print(
        f"INPUT DECISION ROWS : {len(df)}"
    )

    print(
        f"FINAL REPORT ROWS   : {len(report)}"
    )

    print(
        f"UNIQUE TICKERS      : "
        f"{report['ticker'].nunique()}"
    )

    print()

    print(
        f"🟢 ENTRY            : "
        f"{entry_count}"
    )

    print(
        f"🟡 WATCH            : "
        f"{watch_count}"
    )

    print(
        f"🔴 AVOID            : "
        f"{avoid_count}"
    )

    print()

    print(
        "DECISION SCORE VALID : "
        f"{report['decision_score'].notna().sum()}"
    )

    print(
        "DECISION VALID       : "
        f"{report['decision'].notna().sum()}"
    )

    print()

    print(
        "TOP 10 FINAL REPORT"
    )

    print("-" * 72)

    print(
        report[
            [
                "final_rank",
                "ticker",
                "decision_score",
                "flow_score",
                "decision"
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"CSV OUTPUT : {OUTPUT_CSV}"
    )

    print(
        f"MD OUTPUT  : {OUTPUT_MD}"
    )

    print("=" * 72)

    log(
        "STEP 10 FINAL REPORT COMPLETE"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


TOP20_FILE = "data/analysis/top20.csv"
GREEKS_FILE = "data/analysis/options_greeks.csv"
FLOW_FILE = "data/analysis/unusual_flow.csv"

OUTPUT_FILE = "data/analysis/option_search.csv"

TOP_CALLS = 5
TOP_PUTS = 5
MAX_DTE = 180


def log(message):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[07 SEARCH] {now} | {message}")


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def norm_text(series):
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
    )


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


def percentile(series):
    s = numeric(series)

    if s.notna().sum() <= 1:
        return pd.Series(50.0, index=s.index)

    rank = s.rank(pct=True, method="average")
    return rank.fillna(0.5) * 100.0


def find_column(df, candidates):
    normalized = {
        str(c).strip().lower().replace(" ", "_"): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]

    return None


def estimate_premium(df):
    existing = find_column(
        df,
        [
            "estimated_traded_premium",
            "premium",
            "estimated_premium",
            "premium_flow",
        ],
    )

    if existing:
        return numeric(df[existing]).clip(lower=0).fillna(0)

    bid = numeric(df["bid"]).fillna(0)
    ask = numeric(df["ask"]).fillna(0)
    last = numeric(df["lastPrice"]).fillna(0)
    volume = numeric(df["volume"]).fillna(0)

    mid = (bid + ask) / 2

    mid = np.where(
        mid > 0,
        mid,
        last,
    )

    return (
        volume
        * pd.Series(mid, index=df.index).clip(lower=0)
        * 100
    )


def normalize_side(value):
    text = str(value).upper().strip()

    if text in {
        "BUY",
        "BTO",
        "BOT",
        "BUY EST.",
        "BUY EST",
        "BUY_EST.",
        "BUY_EST",
    }:
        return "BUY"

    if text in {
        "SELL",
        "STO",
        "SOLD",
        "SELL EST.",
        "SELL EST",
        "SELL_EST.",
        "SELL_EST",
    }:
        return "SELL"

    return "UNKNOWN"


def main():

    log("START")

    for path in [
        TOP20_FILE,
        GREEKS_FILE,
        FLOW_FILE,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    top20 = pd.read_csv(TOP20_FILE)
    greeks = pd.read_csv(GREEKS_FILE)
    flow = pd.read_csv(FLOW_FILE)

    symbol_col = find_column(
        top20,
        ["symbol", "ticker"],
    )

    if symbol_col is None:
        raise RuntimeError("TOP20 symbol column missing")

    symbols = (
        top20[symbol_col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    required = [
        "symbol",
        "option_type",
        "strike",
        "DTE",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "delta",
        "gamma",
        "vega",
        "underlying_price",
    ]

    missing = [
        c for c in required
        if c not in greeks.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing Greeks columns: "
            + ", ".join(missing)
        )

    greeks["symbol"] = norm_text(greeks["symbol"])
    greeks["option_type"] = norm_text(greeks["option_type"])

    greeks["option_type"] = greeks["option_type"].replace(
        {
            "C": "CALL",
            "CALLS": "CALL",
            "P": "PUT",
            "PUTS": "PUT",
        }
    )

    numeric_columns = [
        "strike",
        "DTE",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "delta",
        "gamma",
        "vega",
        "underlying_price",
    ]

    for column in numeric_columns:
        greeks[column] = numeric(greeks[column])

    greeks["premium"] = estimate_premium(greeks)

    greeks = greeks[
        greeks["symbol"].isin(symbols)
        &
        greeks["DTE"].between(0, MAX_DTE)
        &
        greeks["option_type"].isin(["CALL", "PUT"])
        &
        greeks["strike"].notna()
        &
        greeks["underlying_price"].gt(0)
    ].copy()

    # ---------------------------------------------------------
    # FLOW SCORE
    # ---------------------------------------------------------

    flow_symbol = find_column(
        flow,
        ["symbol", "ticker"],
    )

    flow_score_col = find_column(
        flow,
        [
            "flow_score",
            "unusual_flow_score",
            "option_flow_score",
            "score",
        ],
    )

    side_col = find_column(
        flow,
        [
            "trade_side",
            "trade_side_estimate",
            "side",
            "buy_sell",
        ],
    )

    if flow_symbol:
        flow["symbol"] = norm_text(flow[flow_symbol])

    if flow_score_col:
        flow["flow_score_value"] = numeric(
            flow[flow_score_col]
        )
    else:
        flow["flow_score_value"] = np.nan

    if side_col:
        flow["side_normalized"] = (
            flow[side_col]
            .apply(normalize_side)
        )
    else:
        flow["side_normalized"] = "UNKNOWN"

    flow_lookup = (
        flow[
            flow["symbol"].isin(symbols)
        ]
        .groupby("symbol", as_index=False)
        .agg(
            flow_score=(
                "flow_score_value",
                "max",
            )
        )
    )

    results = []

    for rank, symbol in enumerate(symbols, start=1):

        group = greeks[
            greeks["symbol"] == symbol
        ].copy()

        if group.empty:
            continue

        current_price = safe_float(
            group["underlying_price"].median()
        )

        group["premium_score"] = percentile(
            np.log1p(group["premium"])
        )

        group["volume_score"] = percentile(
            np.log1p(
                group["volume"].clip(lower=0)
            )
        )

        group["oi_score"] = percentile(
            np.log1p(
                group["openInterest"].clip(lower=0)
            )
        )

        group["gamma_score"] = percentile(
            group["gamma"].abs()
        )

        group["delta_score"] = percentile(
            group["delta"].abs()
        )

        group["distance"] = (
            (
                group["strike"]
                - current_price
            ).abs()
            / current_price
        )

        group["moneyness_score"] = (
            100
            -
            group["distance"] * 500
        ).clip(0, 100)

        group["option_score"] = (
            group["premium_score"] * 0.30
            +
            group["volume_score"] * 0.15
            +
            group["oi_score"] * 0.10
            +
            group["gamma_score"] * 0.15
            +
            group["delta_score"] * 0.10
            +
            group["moneyness_score"] * 0.20
        )

        calls = group[
            group["option_type"] == "CALL"
        ].copy()

        puts = group[
            group["option_type"] == "PUT"
        ].copy()

        top_calls = (
            calls
            .sort_values(
                [
                    "option_score",
                    "premium",
                    "volume",
                ],
                ascending=False,
            )
            .head(TOP_CALLS)
        )

        top_puts = (
            puts
            .sort_values(
                [
                    "option_score",
                    "premium",
                    "volume",
                ],
                ascending=False,
            )
            .head(TOP_PUTS)
        )

        flow_score = 0.0

        match = flow_lookup[
            flow_lookup["symbol"] == symbol
        ]

        if not match.empty:
            flow_score = safe_float(
                match.iloc[0]["flow_score"],
                0.0,
            )

        # -----------------------------------------------------
        # BEST BULLISH RISK REVERSAL
        # -----------------------------------------------------

        rr = None

        if not calls.empty and not puts.empty:

            call_candidates = calls[
                (calls["delta"] > 0)
                &
                (calls["strike"] >= current_price)
            ].copy()

            put_candidates = puts[
                (puts["delta"] < 0)
                &
                (puts["strike"] <= current_price)
            ].copy()

            if (
                not call_candidates.empty
                and not put_candidates.empty
            ):

                call_candidates = (
                    call_candidates
                    .sort_values(
                        [
                            "option_score",
                            "premium",
                        ],
                        ascending=False,
                    )
                    .head(20)
                )

                put_candidates = (
                    put_candidates
                    .sort_values(
                        [
                            "option_score",
                            "premium",
                        ],
                        ascending=False,
                    )
                    .head(20)
                )

                pairs = []

                for _, call in call_candidates.iterrows():
                    for _, put in put_candidates.iterrows():

                        dte_gap = abs(
                            float(call["DTE"])
                            -
                            float(put["DTE"])
                        )

                        if dte_gap > 14:
                            continue

                        call_premium = max(
                            safe_float(
                                call["premium"],
                                0,
                            ),
                            0,
                        )

                        put_premium = max(
                            safe_float(
                                put["premium"],
                                0,
                            ),
                            0,
                        )

                        combined = (
                            call_premium
                            +
                            put_premium
                        )

                        if combined <= 0:
                            continue

                        # 양쪽 premium 모두 의미 있는 구조만
                        if (
                            call_premium
                            < combined * 0.02
                        ):
                            continue

                        if (
                            put_premium
                            < combined * 0.02
                        ):
                            continue

                        distance = (
                            abs(
                                call["strike"]
                                - current_price
                            )
                            +
                            abs(
                                put["strike"]
                                - current_price
                            )
                        ) / current_price

                        proximity = max(
                            0,
                            100 - distance * 500,
                        )

                        quality = (
                            call["option_score"] * 0.35
                            +
                            put["option_score"] * 0.25
                            +
                            proximity * 0.20
                            +
                            min(
                                flow_score,
                                100,
                            ) * 0.20
                        )

                        pairs.append(
                            (
                                quality,
                                call,
                                put,
                            )
                        )

                if pairs:

                    pairs.sort(
                        key=lambda x: (
                            x[0],
                            x[1]["premium"]
                            + x[2]["premium"],
                        ),
                        reverse=True,
                    )

                    rr = pairs[0]

        row = {
            "rank": rank,
            "ticker": symbol,
            "current_price": current_price,
            "flow_score": round(
                flow_score,
                2,
            ),
            "top_call_count": len(top_calls),
            "top_put_count": len(top_puts),
        }

        for i in range(TOP_CALLS):

            if i < len(top_calls):

                option = top_calls.iloc[i]

                row[
                    f"call_{i+1}_strike"
                ] = option["strike"]

                row[
                    f"call_{i+1}_dte"
                ] = option["DTE"]

                row[
                    f"call_{i+1}_score"
                ] = round(
                    float(
                        option["option_score"]
                    ),
                    2,
                )

                row[
                    f"call_{i+1}_premium"
                ] = option["premium"]

        for i in range(TOP_PUTS):

            if i < len(top_puts):

                option = top_puts.iloc[i]

                row[
                    f"put_{i+1}_strike"
                ] = option["strike"]

                row[
                    f"put_{i+1}_dte"
                ] = option["DTE"]

                row[
                    f"put_{i+1}_score"
                ] = round(
                    float(
                        option["option_score"]
                    ),
                    2,
                )

                row[
                    f"put_{i+1}_premium"
                ] = option["premium"]

        if rr is not None:

            quality, call, put = rr

            row["risk_reversal"] = (
                "BULLISH RISK-REVERSAL"
            )

            row["rr_score"] = round(
                float(quality),
                2,
            )

            row["rr_call_strike"] = call["strike"]
            row["rr_call_dte"] = call["DTE"]
            row["rr_call_premium"] = call["premium"]

            row["rr_put_strike"] = put["strike"]
            row["rr_put_dte"] = put["DTE"]
            row["rr_put_premium"] = put["premium"]

        else:

            row["risk_reversal"] = "NONE DETECTED"
            row["rr_score"] = 0.0

        results.append(row)

    output = pd.DataFrame(results)

    if len(output) != 20:
        raise RuntimeError(
            f"STEP 7 must contain 20 tickers: {len(output)}"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("STEP 7 OUTPUT")
    print("ROWS    :", len(output))
    print("TICKERS :", output["ticker"].nunique())
    print(
        "RR COUNT:",
        (
            output["risk_reversal"]
            == "BULLISH RISK-REVERSAL"
        ).sum(),
    )

    print("STEP 7 OUTPUT : OK")


if __name__ == "__main__":
    main()


# ============================================================
# REPORT
# ============================================================

def build_report():

    log(
        "LOADING data/analysis RESULTS"
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

    option_rows = read_csv(
        FILES["option_search"]
    )

    greek_rows = read_csv(
        FILES["greeks"]
    )

    structure_rows = read_csv(
        FILES["structure"]
    )

    decision_rows = read_csv(
        FILES["decision"]
    )

    final_rows = read_csv(
        FILES["final"]
    )

    special_rows = read_csv(
        FILES["special"]
    )

    report = []

    # ========================================================
    # TELEGRAM FINAL ORDER
    #
    # 1. FINAL TRADING LIST
    # 2. CALL BUY + PUT SELL
    # 3. MARKET REGIME
    # 4. MARKET REGIME SCORE
    # 5. UNUSUAL OPTION FLOW
    # 6. TOP 20 OPTION SEARCH
    # 7. TOP STOCK DETAIL
    # 8. DATA DISCLAIMER
    # 9. SCAN COMPLETE
    # ========================================================

    # --------------------------------------------------------
    # 1. FINAL TRADING LIST
    # --------------------------------------------------------

    report.extend(
        build_final_list(
            final_rows
        )
    )

    # --------------------------------------------------------
    # 2. CALL BUY + PUT SELL
    # --------------------------------------------------------

    report.extend(
        build_special(
            special_rows
        )
    )

    # --------------------------------------------------------
    # 3. MARKET REGIME
    # --------------------------------------------------------

    report.extend(
        build_market_regime(
            market_rows
        )
    )

    # --------------------------------------------------------
    # 4. MARKET REGIME SCORE
    # --------------------------------------------------------

    report.extend(
        build_market_score(
            market_rows
        )
    )

    # --------------------------------------------------------
    # 5. UNUSUAL OPTION FLOW
    # --------------------------------------------------------

    report.extend(
        build_unusual_flow(
            unusual_rows
        )
    )

    # --------------------------------------------------------
    # 6. TOP 20 OPTION SEARCH
    # --------------------------------------------------------

    report.extend(
        build_top20(
            top20_rows
        )
    )

    # --------------------------------------------------------
    # 7. TOP STOCK DETAIL
    #
    # 선정 이유
    # Market Alignment
    # Option Structure
    # Greeks / Exposure
    # Call Flow
    # Put Flow
    # Risk-Reversal
    # Support / Resistance
    # 종합 판단
    # Final Decision
    # --------------------------------------------------------

    report.extend(
        build_top_detail(
            top20_rows,
            option_rows,
            greek_rows,
            structure_rows,
            decision_rows
        )
    )

    # --------------------------------------------------------
    # 8. DATA DISCLAIMER
    # --------------------------------------------------------

    report.extend(
        build_disclaimer()
    )

    # --------------------------------------------------------
    # 9. SCAN COMPLETE
    # --------------------------------------------------------

    report.extend(
        build_complete(
            final_rows,
            special_rows,
            market_rows
        )
    )

    return "\n".join(report)


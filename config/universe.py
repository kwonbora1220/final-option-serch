# ============================================================
# OPTION FLOW SCANNER V3
# STOCK / ETF UNIVERSE
# ============================================================

# STEP 1에서는 시장 지수만 사용합니다.
# STEP 2 이후 미국 주식/ETF 전체 옵션 스캔용
# Universe를 확장합니다.

MARKET_UNIVERSE = {
    "NDX": "^NDX",
    "SPY": "SPY",
    "SOXX": "SOXX",
    "DIA": "DIA",
}

DEFAULT_OPTION_UNIVERSE = []

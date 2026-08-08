# ============================================================
# OPTION FLOW SCANNER V3 CONFIG
# ============================================================

PROJECT_NAME = "OPTION_FLOW_SCANNER_V3"

MARKET_TIMEZONE = "America/New_York"

MARKET_TICKERS = {
    "NDX": "^NDX",
    "SPY": "SPY",
    "SOXX": "SOXX",
    "DIA": "DIA",
}

OPTION_MIN_DTE = 0
OPTION_MAX_DTE = 180

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"

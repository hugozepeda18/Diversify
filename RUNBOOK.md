# Local Environment Runbook

## Prerequisites
* Docker & Docker Compose
* Python 3.12+
* `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## 1. Infrastructure Setup
```bash
# Start TimescaleDB
docker compose up -d

# Verify TimescaleDB is ready
docker compose ps
```

## 2. Environment Setup
```bash
cp .env.example .env
# Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or WEBHOOK_URL) in .env

# Create virtualenv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run database migrations
alembic upgrade head
```

## 3. Verification & Testing
```bash
# Run unit and integration tests
pytest tests/ -v

# Run formatting check
ruff check .
```

## 4. Operational Commands
```bash
# 0. Seed historical 15m + 1h candles from Binance (resumes from the newest stored candle)
python -m src.main backfill --symbol BTC/USDT --days 1460   # ~4y, ~2 min first run

# 1. Backtest with VectorBT (ATR SL/TP + 1% risk sizing; --no-risk for all-in, no stops).
#    Judge per-year with --days 365 too, and always against buy_hold_return_pct.
python -m src.main backtest --symbol BTC/USDT --timeframe 1h --days 1460

# 1b. Parameter search: fit on data before --split, report out-of-sample after it.
#     Pick a plateau that holds in test, not the single best train row.
python -m src.main optimize --strategy DoubleEmaCross --timeframe 15m 1h 4h 1d --split 2025-01-01

# 2. (Alert dispatcher / test-alert: not built — alerts skipped by request.)

# 3. Start live ingestion + signal engine (defaults: DoubleEmaCross 20/200, 4h, 5xATR SL, 10R TP).
#    Signals are logged as "SIGNAL {json}" and stored in trade_signals; open trades in active_positions.
python -m src.main monitor --symbol BTC/USDT --timeframe 4h

# 4. Execute on Binance Demo Trading (real prices, fake funds). Needs BINANCE_TESTNET_API_KEY /
#    BINANCE_TESTNET_SECRET in .env (HMAC keys from demo.binance.com). Each entry = market buy +
#    exchange-side OCO (SL/TP), so stops fire even while the bot is down; fills are synced every 30s.
python -m src.main broker-check          # read-only: demo balance + exchange status of positions
python -m src.main monitor --execute     # top-10 coins, 4h, real demo orders
```
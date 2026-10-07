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
python -m src.main backfill --symbol BTC/USDT --days 60

# 1. Backtest a strategy with VectorBT
python -m src.main backtest --symbol BTC/USDT --timeframe 1h --days 60

# 2. Test the Signal Dispatcher (Sends a mock JSON trade payload to your Telegram/Webhook)
python -m src.main test-alert

# 3. Start Live Candle Ingestion & Signal Generator
python -m src.main monitor --symbol BTC/USDT --timeframe 15m
```
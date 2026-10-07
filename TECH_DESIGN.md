# Technical Design Document (TDD) - Trading Bot MVP

## 1. Stack & Architecture
* **Language & Runtime:** Python 3.12+ (Strict typing, `asyncio`).
* **Exchange Gateway:** `ccxt` (Async / `ccxt.pro` for WebSocket state maintenance).
* **Vectorized Backtesting Engine:** `vectorbt` (NumPy + Numba).
* **Database:** PostgreSQL 16 with **TimescaleDB** (Hypertables optimized for 15m and 1h Klines).
* **Persistence Layer:** SQLAlchemy 2.0 (Async) + Alembic.
* **Signal Alerting:** Direct HTTP Dispatcher (Telegram Bot API / Discord Webhook) emitting structured JSON.

## 2. Directory Layout
```text
trading-bot/
├── src/
│   ├── core/               # App configuration, DB session, CCXT exchange client
│   ├── data/               # Historical sync and live WebSocket candle ingestion
│   ├── indicators/         # Vectorized indicator formulas (EMA, RSI, BB)
│   ├── strategies/         # BaseStrategy contract and concrete strategy definitions
│   ├── backtest/           # VectorBT historical backtest runner and reporting
│   ├── signals/            # Signal generation, JSON payload formatting, and alert dispatcher
│   ├── models/             # SQLAlchemy ORM models (Hypertables, Signals, Trades)
│   └── main.py             # CLI entrypoint for backtest, live sync, and signal daemon
├── tests/
│   ├── unit/
│   └── integration/
├── CLAUDE.md
├── RUNBOOK.md
└── TASKS.md
```

## 3. Architecture & Data Model (TimescaleDB)
* **`market_candles` (Hypertable chunked by time):**
  * `timestamp`: TIMESTAMPTZ (PK)
  * `symbol`: VARCHAR(20) (PK)
  * `timeframe`: VARCHAR(10) (PK, e.g., '15m', '1h')
  * `open`, `high`, `low`, `close`, `volume`: NUMERIC(18, 8)
* **`trade_signals` (JSON Audit Log):**
  * `id`: UUID (PK)
  * `timestamp`: TIMESTAMPTZ
  * `strategy_name`: VARCHAR(50)
  * `symbol`: VARCHAR(20)
  * `action`: VARCHAR(10)
  * `price`: NUMERIC(18, 8)
  * `payload`: JSONB
* **`active_positions` (Crash Recovery):**
  * Tracks currently open paper trades. Queried on boot to ensure the bot can rebuild its state and prevent duplicate entries if the server restarts.

## 4. Core Engine & Safeguards
* **Resilient Ingestion:** `ccxt.pro` streams live WebSockets to an in-memory price book, decoupled from the strategy evaluation loop to prevent blocking.
* **Vectorized Strategy Evaluation:** Indicators are calculated using `vectorbt` operations. Arrays are strictly shifted by 1 index (`.shift(1)`) to eliminate look-ahead bias before trigger evaluation.
* **Risk Controls & Position Sizing:**
  * Every emitted signal MUST include a hard Stop-Loss (SL) and Take-Profit (TP).
  * Position sizing is calculated dynamically based on a fixed risk percentage (e.g., risking 1% of account equity per trade based on the SL distance).

## 5. Signal Trigger Engine & JSON Payload Contract
When a 15m or 1h candle closes, the active strategy evaluates the updated series. If a trigger fires, the engine generates and dispatches this strict JSON payload via the configured Webhook/Telegram integration:

```json
{
  "event": "TRADE_SIGNAL",
  "timestamp": "2026-10-07T08:00:00Z",
  "strategy": "DoubleEmaCross",
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "action": "ENTER_LONG",
  "trigger_price": 64250.50,
  "risk_management": {
    "stop_loss": 62300.00,
    "take_profit": 68150.00,
    "risk_reward_ratio": 2.0,
    "position_size_usd": 1500.00
  },
  "indicators_snapshot": {
    "ema_fast": 64100.20,
    "ema_slow": 63890.10,
    "rsi_14": 58.4
  }
}
```
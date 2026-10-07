# MVP Implementation Tasks

Rules for the Agent:
1. Complete tasks sequentially.
2. Mark tasks `[x]` only when implementation and passing automated tests are confirmed.
3. Commit progress using conventional commits.

---

- [x] **Phase 1: Project Setup, Database & State Recovery**
  - [x] Initialize Python project with `uv` and configure `pyproject.toml` (`ccxt`, `vectorbt`, `sqlalchemy`, `asyncpg`, `alembic`, `pydantic-settings`).
  - [x] Configure `docker-compose.yml` with TimescaleDB (`timescale/timescaledb:latest-pg16`).
  - [x] Configure SQLAlchemy models: create `market_candles` and execute the TimescaleDB `create_hypertable` migration script.
  - [x] Create `trade_signals` model to store historical JSON signal dispatches.
  - [x] Create `active_positions` model and implement a startup initialization script to query it on boot to rebuild the bot's live trading state (Crash Recovery).

- [ ] **Phase 2: Live Market Data Pipeline (`ccxt.pro`)**
  - [ ] Implement `ExchangeGateway` class using async `ccxt.pro` WebSockets to maintain a zero-latency local price book.
  - [ ] Build historical candle backfiller (REST) for seeding TimescaleDB with past 15m and 1h Klines from Binance.
  - [ ] Implement live ingestion loop using `ccxt.watch_ohlcv()` to listen for candle closes, persist them to TimescaleDB, and trigger the Strategy Engine.
  - [ ] Write integration test verifying candles are correctly written and deduplicated in TimescaleDB.

- [ ] **Phase 3: Vectorized Strategy Engine**
  - [ ] Define `BaseStrategy` abstract interface with vectorized evaluation logic.
  - [ ] Implement `DoubleEmaCross` and `RsiThreshold` strategies using `vectorbt` indicators.
  - [ ] Eliminate Look-Ahead Bias: Implement strict `.shift(1)` logic on indicator arrays inside the strategy evaluator.
  - [ ] Build the Backtest CLI runner to output `vectorbt` metrics over historical data (Sharpe ratio, Max Drawdown, Total Return, Win Rate).
  - [ ] Write unit tests verifying that strategy buy/sell triggers produce correct boolean signal arrays.

- [ ] **Phase 4: Risk Management & Dynamic Execution**
  - [ ] Build `RiskManager`: Calculate dynamic position sizes based on a 1% account equity risk parameter.
  - [ ] Implement Hard Stops: Ensure the `RiskManager` calculates absolute price targets for Stop-Loss and Take-Profit for every entry.
  - [ ] Write unit tests verifying that position sizes shrink appropriately when the Stop-Loss is wider.

- [ ] **Phase 5: Structured Signal Trigger & Free Alert Dispatcher**
  - [ ] Build `SignalPayloadBuilder` to assemble complete trade metadata, risk limits, and indicator snapshots into structured JSON.
  - [ ] Implement `SignalDispatcher`: Telegram Bot API client using `httpx`, with a generic HTTP Webhook POST fallback.
  - [ ] Add CLI command `test-alert` to verify your receiving endpoint receives the formatted JSON payload.
  - [ ] Implement live monitoring daemon: loops on candle close -> evaluates strategy -> calculates risk -> saves position to DB -> dispatches JSON alert.
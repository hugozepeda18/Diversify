# Agent Directives (`CLAUDE.md`)

## Operational Rules
* Work strictly phase-by-phase according to `TASKS.md`.
* Run test suites before marking tasks as complete.
* Never hardcode API keys or webhook URLs. Load them strictly via `.env`.
* Keep the code vectorized: Use NumPy, Pandas, or VectorBT for indicator and signal calculations. Never use raw Python `for` loops across candle rows.

## Tooling & CLI Commands
* Environment manager: Use `uv` strictly.
* Lint & Format: `ruff check .` and `ruff format .`
* Type Checking: `mypy src/`
* Test Runner: `pytest tests/ -v`

## CLI Execution Patterns
* Run backtest: `python -m src.main backtest --strategy DoubleEma --symbol BTC/USDT --timeframe 1h`
* Run live signal engine: `python -m src.main monitor --symbol BTC/USDT --timeframe 15m`
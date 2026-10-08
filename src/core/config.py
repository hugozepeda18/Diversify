from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    account_equity_usd: float = 10_000.0
    risk_per_trade: float = 0.01
    # Binance Demo Trading keys (real prices, fake funds). Env names say TESTNET for history.
    binance_testnet_api_key: SecretStr | None = None
    binance_testnet_secret: SecretStr | None = None


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    account_equity_usd: float = 10_000.0
    risk_per_trade: float = 0.01


settings = Settings()

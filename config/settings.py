from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "EDITH"
    VERSION: str = "8.0.0"

    BASE_DIR: Path = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # Ollama
    OLLAMA_HOST: str = (
        "http://localhost:11434"
    )

    OLLAMA_MODEL: str = (
        "qwen2.5:7b"
    )

    # Let Ollama automatically manage GPU usage
    # unless explicitly overridden through .env.
    OLLAMA_NUM_GPU: int = -1

    # Context size sent to the model.
    OLLAMA_NUM_CTX: int = 4096

    # Lower temperature keeps EDITH more consistent
    # for commands and tool routing.
    OLLAMA_TEMPERATURE: float = 0.3

    # 0 lets Ollama choose automatically.
    OLLAMA_NUM_THREAD: int = 0

    # Maximum generated tokens for one response.
    OLLAMA_NUM_PREDICT: int = 384

    # Keep the model loaded between requests.
    OLLAMA_KEEP_ALIVE: str = "30m"

    # Database
    DATABASE_PATH: Path = (
        BASE_DIR / "edith.db"
    )

    # Logs
    LOGS_DIR: Path = (
        BASE_DIR / "logs"
    )

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Ensure required EDITH directories exist."""

        self.LOGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.DATABASE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


settings = Settings()

settings.ensure_directories()
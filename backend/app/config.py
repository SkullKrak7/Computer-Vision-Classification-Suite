"""Application configuration using Pydantic BaseSettings"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API
    api_title: str = "CV Classification API"
    api_version: str = "1.0.0"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Security
    rate_limit_per_minute: int = 100
    max_file_size_mb: int = 10
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Model
    model_path: str = "models/pytorch_cnn_tuned.pth"

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables.
    This class uses pydantic_settings to automatically load values
    from .env file or system environment variables.
    """

    SERVER_NAME: str = "DBServer"
    SERVER_DESCRIPTION: str = "A server for executing PostgreSQL queries and managing database interactions."
    SERVER_TRANSPORT: str = "streamable-http"    # options: "stdio", "streamable-http", "sse"(do not use, deprecated)

    DATABASE_URL: str = "postgresql+asyncpg://unisql_user:unisql_user@localhost:5432/unisqldb"

    # DB_USER: str = "unisql_user"
    # DB_PASSWORD: str = "unisql_user"
    # DB_NAME: str = "unisqldb"
    # DB_HOST: str = "localhost"
    # DB_PORT: int = 5432

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()

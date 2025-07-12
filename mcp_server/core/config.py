from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables.
    This class uses pydantic_settings to automatically load values
    from .env file or system environment variables.
    """

    DATABASE_URL: str = "postgresql+asyncpg://movie_user:movie_user@localhost:5432/movies"

    # DB_USER: str = "unisql_user"
    # DB_PASSWORD: str = "unisql_user"
    # DB_NAME: str = "unisqldb"
    # DB_HOST: str = "localhost"
    # DB_PORT: int = 5432
    SERVER_TRANSPORT: str = "stdio"

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()

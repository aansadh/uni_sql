# unisql_backend/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Configuration settings for the uniSQL backend application.

    Attributes:
        NL2SQL_LLM_URL (str): URL for the NL2SQL LLM API.
        NL2SQL_LLM_MODEL (str): Model name for the NL2SQL LLM.
        MCP_SERVER_URL (str): URL for the MCP server.
        DATABASE_URL (str): Database connection string.
        LOG_LEVEL (str): Logging level for the application.
    """

    # Ollama (NL2SQL LLM) settings
    NL2SQL_LLM_URL: str = "http://localhost:11434/api/chat"
    NL2SQL_LLM_MODEL: str = "sqlcoder:7b"

    LLM_URL: str = "http://localhost:11434/api/chat"
    LLM_MODEL: str = "qwen2.5:7b"

    # MCP Server (Database Connector) settings
    MCP_SERVER_URL: str = "http://localhost:8000/mcp"

    LOG_LEVEL:int = logging.INFO

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Settings initialized with LOG_LEVEL=%s", self.LOG_LEVEL)

settings = Settings()
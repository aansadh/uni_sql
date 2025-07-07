from fastapi import FastAPI
from routes import nl2sql
from contextlib import asynccontextmanager
from core.config import settings
from core.exception_handlers import generic_exception_handler
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager to initialize resources.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """
    logger.info("Initializing application lifespan context.")
    app.state.settings = settings
    yield
    logger.info("Application lifespan context closed.")

app = FastAPI(lifespan=lifespan)

logger.info("FastAPI application instance created.")

app.add_exception_handler(Exception, generic_exception_handler)
logger.info("Generic exception handler added to the application.")

app.include_router(nl2sql.router, prefix="/api")
logger.info("Router for NL2SQL added with prefix '/api'.")

@app.get("/")
async def read_root():
    """
    Root endpoint of the application.

    Returns:
        dict: A welcome message.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to uniSQL MCP Client API. Visit /docs or /redoc for documentation."}

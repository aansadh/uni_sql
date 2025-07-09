from services.db_services import DBServices
import logging
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

_async_engine: Engine = None
_Session = None

def create_engine(database_url: str):
    """
    Connect to the PostgreSQL database using SQLAlchemy.

    Args:
        database_url (str): The database URL in the format:
                            postgresql+asyncpg://<user>:<password>@<host>:<port>/<db>

    Returns:
        DbEngine, Session: A tuple containing the SQLAlchemy async engine and sessionmaker.
    """
    if not database_url.strip():
        raise ValueError("Database URL cannot be empty or whitespace.")
    try:
        global _async_engine, _Session
        if _async_engine is None:
            # pool_pre_ping maintains active connections.
            _async_engine = create_async_engine(database_url.strip())
            _Session = sessionmaker(
                bind=_async_engine,
                class_=AsyncSession
            )
        return _async_engine, _Session
    except Exception as e:
        raise RuntimeError(f"Failed to create database engine: {e}")

def get_engine(mcp: FastMCP = None) -> Engine:
    """
    Get the SQLAlchemy async engine.

    Args:
        mcp (FastMCP): The FastMCP application instance.

    Returns:
        Engine: The SQLAlchemy async engine.
    """
    ctx = mcp.get_context()
    engine = ctx.request_context.lifespan_context.db_engine or _async_engine
    if engine is None:
        raise RuntimeError("Database engine not initialized. Call create_engine first.")
    return engine

def get_new_session(mcp: FastMCP = None) -> AsyncSession:
    """
    Get a new SQLAlchemy async session.

    Args:
        mcp (FastMCP): The FastMCP application instance.

    Returns:
        AsyncSession: A new SQLAlchemy async session.
    """
    ctx = mcp.get_context()
    sessionmaker = ctx.request_context.lifespan_context.db_sessionmaker or _Session
    if sessionmaker is None:
        raise RuntimeError("Sessionmaker not initialized. Call create_engine first.")
    return sessionmaker()

def get_sessionmaker(mcp: FastMCP = None):
    """
    Get the SQLAlchemy sessionmaker.

    Args:
        mcp (FastMCP): The FastMCP application instance.

    Returns:
        sessionmaker: The SQLAlchemy sessionmaker.

    Raises:
        RuntimeError: If the sessionmaker is not initialized.

    Note: Use this as:
        with get_sessionmaker(mcp)() as session:
            # Your code here...
    """
    ctx = mcp.get_context()
    sessionmaker = ctx.request_context.lifespan_context.db_sessionmaker or _Session
    if sessionmaker is None:
        raise RuntimeError("Sessionmaker not initialized. Call create_engine first.")
    return sessionmaker

def get_db_services(mcp: FastMCP) -> DBServices:
    """
    Dependency to get the DBServices instance.

    Args:
        Session (AsyncSession): The SQLAlchemy async session.

    Returns:
        DBServices: An instance of DBServices.
    """
    new_session = get_new_session(mcp)
    return DBServices(db_session=new_session)
    
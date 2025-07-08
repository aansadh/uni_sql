from fastapi import Request, Depends
from mcp_server.services.db_services import DBServices
import logging
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

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
        None
    """
    global _async_engine
    if _async_engine is None:
        # pool_pre_ping maintains active connections.
        _async_engine = create_async_engine(database_url, pool_pre_ping=True, echo=True)
        _Session = sessionmaker(
            bind=_async_engine,
            class_=AsyncSession
        )
    return _async_engine

def get_engine():
    """
    Get the SQLAlchemy async engine.

    Returns:
        Engine: The SQLAlchemy async engine.
    """
    global _async_engine
    if _async_engine is None:
        raise RuntimeError("Database engine not initialized. Call create_engine first.")
    return _async_engine

def get_session():
    """
    Get a new SQLAlchemy async session.

    Returns:
        AsyncSession: A new SQLAlchemy async session.
    """
    global _Session
    if _Session is None:
        raise RuntimeError("Session factory not initialized. Call create_engine first.")
    return _Session()

def get_db_services(request: Request, db_engine: Engine = Depends(get_engine)):
    if not hasattr(request.app.state, 'db_services') or request.app.state.db_services is None:
        request.app.state.db_services = DBServices(db_engine=db_engine)
    return request.app.state.db_services
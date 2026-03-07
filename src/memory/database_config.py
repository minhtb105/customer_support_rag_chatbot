"""
Database Configuration Module

Implements environment-based configuration for PostgreSQL with pgvector
and SQLAlchemy integration. Supports local development with SQLite fallback.
"""

import os
from enum import Enum
from typing import Dict, Any
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from config import PDF_DB_DIR


class Environment(Enum):
    """Environment types for database configuration."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


# Get current environment
ENV = Environment(os.getenv("ENV", "local"))

# Database connection strings
DATABASE_URLS = {
    Environment.LOCAL: f"sqlite:///{PDF_DB_DIR}/memory.db",
    Environment.DEVELOPMENT: os.getenv(
        "DATABASE_URL_DEV", 
        "postgresql://postgres:password@localhost:5432/memory_dev"
    ),
    Environment.PRODUCTION: os.getenv(
        "DATABASE_URL_PROD",
        "postgresql://postgres:password@prod-server:5432/memory_prod"
    )
}

# Database configurations
DATABASE_CONFIGS = {
    Environment.LOCAL: {
        "echo": True,
        "poolclass": None,  # No pooling for SQLite
        "connect_args": {"check_same_thread": False}
    },
    Environment.DEVELOPMENT: {
        "echo": False,
        "poolclass": QueuePool,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "pool_recycle": 3600
    },
    Environment.PRODUCTION: {
        "echo": False,
        "poolclass": QueuePool,
        "pool_size": 20,
        "max_overflow": 30,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_timeout": 30
    }
}


def get_database_url() -> str:
    """Get database URL for current environment."""
    return DATABASE_URLS[ENV]


def get_database_config() -> Dict[str, Any]:
    """Get database configuration for current environment."""
    return DATABASE_CONFIGS[ENV]


def create_database_engine() -> Engine:
    """Create and configure database engine."""
    url = get_database_url()
    config = get_database_config()
    
    print(f"🔧 Database Engine: {ENV.value}")
    print(f"📍 Database URL: {url}")
    
    try:
        engine = create_engine(url, **config)
        
        # Test connection
        with engine.connect() as conn:
            print("✅ Database connection successful")
        
        return engine
    
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise


def create_session_factory(engine: Engine) -> sessionmaker:
    """Create session factory for database operations."""
    return sessionmaker(bind=engine)


# Global instances
engine = create_database_engine()
SessionLocal = create_session_factory(engine)
Base = declarative_base()


def get_db_session() -> Session:
    """Get database session for operations."""
    return SessionLocal()


def close_db_session(session: Session):
    """Close database session."""
    session.close()


def init_database():
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        raise


def test_connection():
    """Test database connection and pgvector availability."""
    try:
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute("SELECT 1").scalar()
            print(f"✅ Basic connection test: {result}")
            
            # Test pgvector if using PostgreSQL
            if ENV != Environment.LOCAL:
                try:
                    result = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").scalar()
                    if result:
                        print("✅ pgvector extension available")
                    else:
                        print("⚠️  pgvector extension not found - install with: CREATE EXTENSION vector;")
                except Exception as e:
                    print(f"⚠️  pgvector test failed: {e}")
            
            return True
    
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Database Configuration Test")
    print(f"Environment: {ENV.value}")
    print(f"Database URL: {get_database_url()}")
    
    if test_connection():
        init_database()
        print("🎉 Database setup complete!")
    else:
        print("💥 Database setup failed!")
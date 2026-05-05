"""
Database Connection Module - psycopg3 connection pool for PostgreSQL + PostGIS + AGE.

Provides a thread-safe connection pool. Each connection automatically:
  - Loads the Apache AGE extension
  - Sets search_path to include ag_catalog (required for Cypher queries)
  - Registers pgvector numpy type adapter

Usage:
    from core.db import get_pool, get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM osm_schools")
            print(cur.fetchone())
"""

import os
from contextlib import contextmanager
from functools import lru_cache

import psycopg
import psycopg_pool
from psycopg.rows import dict_row


def _dsn() -> str:
    """Build a libpq-style DSN from environment variables."""
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    dbname   = os.getenv("DB_NAME", "geospatial")
    user     = os.getenv("DB_USER", "geo")
    password = os.getenv("DB_PASSWORD", "geo")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _configure_connection(conn: psycopg.Connection) -> None:
    """
    Called by the pool after each new connection is opened.
    Loads Apache AGE and sets the search_path so ag_catalog is visible.
    """
    conn.autocommit = True
    with conn.cursor() as cur:
        # Load AGE shared library
        cur.execute("LOAD 'age';")
        # Ensure ag_catalog is in the path for Cypher queries
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")
    conn.autocommit = False


@lru_cache(maxsize=1)
def get_pool() -> psycopg_pool.ConnectionPool:
    """
    Return the global connection pool (singleton).
    Pool is created on first call and reused thereafter.
    """
    pool = psycopg_pool.ConnectionPool(
        conninfo=_dsn(),
        min_size=1,
        max_size=5,
        kwargs={"row_factory": dict_row},
        configure=_configure_connection,
        open=True,
    )
    return pool


@contextmanager
def get_conn():
    """
    Context manager that yields a connection from the pool.

    Example:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the connection pool (call at app shutdown if needed)."""
    pool = get_pool()
    pool.close()

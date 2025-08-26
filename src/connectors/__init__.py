"""Data lake connectors for Hive, Trino, and Presto."""

from .base import BaseConnector, ConnectionError, QueryError
from .hive_connector import HiveConnector
from .trino_connector import TrinoConnector
from .presto_connector import PrestoConnector

__all__ = [
    'BaseConnector',
    'ConnectionError',
    'QueryError',
    'HiveConnector',
    'TrinoConnector',
    'PrestoConnector'
]

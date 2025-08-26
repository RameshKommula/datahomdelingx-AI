"""Base connector class for data lake connections."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from loguru import logger


class ConnectionError(Exception):
    """Raised when connection to data lake fails."""
    pass


class QueryError(Exception):
    """Raised when query execution fails."""
    pass


class BaseConnector(ABC):
    """Base class for data lake connectors."""
    
    def __init__(self, host: str, port: int, username: str, **kwargs):
        self.host = host
        self.port = port
        self.username = username
        self.connection = None
        self._connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data lake."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the data lake."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        pass
    
    @abstractmethod
    def get_databases(self) -> List[str]:
        """Get list of available databases."""
        pass
    
    @abstractmethod
    def get_tables(self, database: str) -> List[str]:
        """Get list of tables in a database."""
        pass
    
    @abstractmethod
    def get_table_schema(self, database: str, table: str) -> List[Dict[str, Any]]:
        """Get schema information for a table."""
        pass
    
    @abstractmethod
    def get_table_stats(self, database: str, table: str) -> Dict[str, Any]:
        """Get statistics for a table."""
        pass
    
    def get_sample_data(self, database: str, table: str, limit: int = 1000) -> pd.DataFrame:
        """Get sample data from a table."""
        query = f"SELECT * FROM {database}.{table} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_column_stats(self, database: str, table: str, column: str) -> Dict[str, Any]:
        """Get statistics for a specific column."""
        query = f"""
        SELECT 
            COUNT(*) as total_count,
            COUNT({column}) as non_null_count,
            COUNT(DISTINCT {column}) as distinct_count
        FROM {database}.{table}
        """
        result = self.execute_query(query)
        
        if not result.empty:
            stats = result.iloc[0].to_dict()
            stats['null_count'] = stats['total_count'] - stats['non_null_count']
            stats['null_percentage'] = (stats['null_count'] / stats['total_count']) * 100 if stats['total_count'] > 0 else 0
            return stats
        
        return {}
    
    def test_connection(self) -> bool:
        """Test the connection to the data lake."""
        try:
            self.connect()
            databases = self.get_databases()
            self.disconnect()
            logger.info(f"Connection test successful. Found {len(databases)} databases.")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to the data lake."""
        return self._connected
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

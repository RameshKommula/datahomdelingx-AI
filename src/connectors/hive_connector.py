"""Hive connector for data lake access."""

from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectionError, QueryError

try:
    from pyhive import hive
    PYHIVE_AVAILABLE = True
except ImportError:
    PYHIVE_AVAILABLE = False
    logger.warning("pyhive not available - Hive connector will not work")


class HiveConnector(BaseConnector):
    """Connector for Apache Hive."""
    
    def __init__(self, host: str, port: int, username: str, password: Optional[str] = None, 
                 database: str = "default", **kwargs):
        super().__init__(host, port, username, **kwargs)
        self.password = password
        self.database = database
    
    def connect(self) -> None:
        """Connect to Hive database."""
        if not PYHIVE_AVAILABLE:
            raise ConnectionError("pyhive is not installed. Install with: pip install 'pyhive[hive]'")
        try:
            self.connection = hive.Connection(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database
            )
            self._connected = True
            logger.info(f"Connected to Hive at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Hive: {e}")
            raise ConnectionError(f"Failed to connect to Hive: {e}")
    
    def disconnect(self) -> None:
        """Close connection to Hive."""
        if self.connection:
            try:
                self.connection.close()
                self._connected = False
                logger.info("Disconnected from Hive")
            except Exception as e:
                logger.warning(f"Error during Hive disconnection: {e}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        if not self._connected:
            raise ConnectionError("Not connected to Hive")
        
        try:
            logger.debug(f"Executing Hive query: {query}")
            df = pd.read_sql(query, self.connection)
            logger.debug(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise QueryError(f"Query execution failed: {e}")
    
    def get_databases(self) -> List[str]:
        """Get list of available databases."""
        query = "SHOW DATABASES"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def get_tables(self, database: str) -> List[str]:
        """Get list of tables in a database."""
        query = f"SHOW TABLES IN {database}"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def get_table_schema(self, database: str, table: str) -> List[Dict[str, Any]]:
        """Get schema information for a table."""
        query = f"DESCRIBE {database}.{table}"
        result = self.execute_query(query)
        
        schema = []
        for _, row in result.iterrows():
            if len(row) >= 3:
                schema.append({
                    'column_name': row.iloc[0],
                    'data_type': row.iloc[1],
                    'comment': row.iloc[2] if pd.notna(row.iloc[2]) else None
                })
        
        return schema
    
    def get_table_stats(self, database: str, table: str) -> Dict[str, Any]:
        """Get statistics for a table."""
        try:
            # Get basic table info
            query = f"DESCRIBE FORMATTED {database}.{table}"
            result = self.execute_query(query)
            
            stats = {
                'database': database,
                'table': table,
                'row_count': None,
                'size_bytes': None,
                'last_modified': None,
                'location': None
            }
            
            # Parse the formatted output
            for _, row in result.iterrows():
                if len(row) >= 2:
                    key = str(row.iloc[0]).strip().lower()
                    value = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
                    
                    if 'numrows' in key:
                        try:
                            stats['row_count'] = int(value) if value and value != '-1' else None
                        except ValueError:
                            pass
                    elif 'totalsize' in key or 'rawdatasize' in key:
                        try:
                            stats['size_bytes'] = int(value) if value and value != '-1' else None
                        except ValueError:
                            pass
                    elif 'lastmodifiedtime' in key:
                        stats['last_modified'] = value
                    elif 'location' in key:
                        stats['location'] = value
            
            # Try to get row count if not available
            if stats['row_count'] is None:
                try:
                    count_query = f"SELECT COUNT(*) FROM {database}.{table}"
                    count_result = self.execute_query(count_query)
                    if not count_result.empty:
                        stats['row_count'] = int(count_result.iloc[0, 0])
                except Exception:
                    logger.warning(f"Could not get row count for {database}.{table}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get table stats for {database}.{table}: {e}")
            return {
                'database': database,
                'table': table,
                'error': str(e)
            }
    
    def get_partitions(self, database: str, table: str) -> List[str]:
        """Get partition information for a table."""
        try:
            query = f"SHOW PARTITIONS {database}.{table}"
            result = self.execute_query(query)
            return result.iloc[:, 0].tolist()
        except Exception as e:
            logger.debug(f"Table {database}.{table} is not partitioned or error occurred: {e}")
            return []
    
    def analyze_table(self, database: str, table: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a table."""
        try:
            # Update table statistics
            analyze_query = f"ANALYZE TABLE {database}.{table} COMPUTE STATISTICS"
            self.execute_query(analyze_query)
            
            # Analyze columns
            analyze_columns_query = f"ANALYZE TABLE {database}.{table} COMPUTE STATISTICS FOR COLUMNS"
            self.execute_query(analyze_columns_query)
            
            logger.info(f"Analysis completed for {database}.{table}")
            return {'status': 'success', 'message': 'Table analysis completed'}
            
        except Exception as e:
            logger.error(f"Failed to analyze table {database}.{table}: {e}")
            return {'status': 'error', 'message': str(e)}

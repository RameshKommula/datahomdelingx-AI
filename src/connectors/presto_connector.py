"""Presto connector for data lake access."""

from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

from .base import BaseConnector, ConnectionError, QueryError

try:
    import prestodb
    PRESTODB_AVAILABLE = True
except ImportError:
    PRESTODB_AVAILABLE = False
    logger.warning("prestodb not available - Presto connector will not work")


class PrestoConnector(BaseConnector):
    """Connector for Presto."""
    
    def __init__(self, host: str, port: int, username: str, catalog: str = "hive",
                 schema: str = "default", **kwargs):
        super().__init__(host, port, username, **kwargs)
        self.catalog = catalog
        self.schema = schema
    
    def connect(self) -> None:
        """Establish connection to Presto."""
        if not PRESTODB_AVAILABLE:
            raise ConnectionError("prestodb is not installed. Install with: pip install prestodb")
        try:
            self.connection = prestodb.dbapi.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                catalog=self.catalog,
                schema=self.schema
            )
            self._connected = True
            logger.info(f"Connected to Presto at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Presto: {e}")
            raise ConnectionError(f"Failed to connect to Presto: {e}")
    
    def disconnect(self) -> None:
        """Close connection to Presto."""
        if self.connection:
            try:
                self.connection.close()
                self._connected = False
                logger.info("Disconnected from Presto")
            except Exception as e:
                logger.warning(f"Error during Presto disconnection: {e}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        if not self._connected:
            raise ConnectionError("Not connected to Presto")
        
        try:
            logger.debug(f"Executing Presto query: {query}")
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            logger.debug(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise QueryError(f"Query execution failed: {e}")
    
    def get_databases(self) -> List[str]:
        """Get list of available schemas (databases in Presto terminology)."""
        query = f"SHOW SCHEMAS FROM {self.catalog}"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def get_tables(self, database: str) -> List[str]:
        """Get list of tables in a schema."""
        query = f"SHOW TABLES FROM {self.catalog}.{database}"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def get_table_schema(self, database: str, table: str) -> List[Dict[str, Any]]:
        """Get schema information for a table."""
        query = f"DESCRIBE {self.catalog}.{database}.{table}"
        result = self.execute_query(query)
        
        schema = []
        for _, row in result.iterrows():
            schema.append({
                'column_name': row['Column'],
                'data_type': row['Type'],
                'nullable': row.get('Null', 'YES') == 'YES',
                'comment': row.get('Comment', None)
            })
        
        return schema
    
    def get_table_stats(self, database: str, table: str) -> Dict[str, Any]:
        """Get statistics for a table."""
        try:
            # Try to get table stats
            query = f"SHOW STATS FOR {self.catalog}.{database}.{table}"
            result = self.execute_query(query)
            
            stats = {
                'database': database,
                'table': table,
                'catalog': self.catalog,
                'columns': {}
            }
            
            # Parse statistics
            for _, row in result.iterrows():
                column_name = row.get('column_name', None)
                if column_name:
                    stats['columns'][column_name] = {
                        'data_size': row.get('data_size', None),
                        'distinct_values_count': row.get('distinct_values_count', None),
                        'nulls_fraction': row.get('nulls_fraction', None)
                    }
                else:
                    # Table-level stats
                    stats['row_count'] = row.get('row_count', None)
                    stats['data_size'] = row.get('data_size', None)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get table stats for {database}.{table}: {e}")
            # Fallback to basic count
            try:
                count_query = f"SELECT COUNT(*) FROM {self.catalog}.{database}.{table}"
                count_result = self.execute_query(count_query)
                return {
                    'database': database,
                    'table': table,
                    'catalog': self.catalog,
                    'row_count': int(count_result.iloc[0, 0]) if not count_result.empty else None,
                    'error': str(e)
                }
            except Exception:
                return {
                    'database': database,
                    'table': table,
                    'catalog': self.catalog,
                    'error': str(e)
                }
    
    def get_column_stats(self, database: str, table: str, column: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific column."""
        try:
            # Get basic stats
            basic_stats = super().get_column_stats(database, table, column)
            
            # Get additional Presto-specific stats
            query = f"""
            SELECT 
                approx_distinct({column}) as approx_distinct_count,
                min({column}) as min_value,
                max({column}) as max_value
            FROM {self.catalog}.{database}.{table}
            """
            
            result = self.execute_query(query)
            if not result.empty:
                row = result.iloc[0]
                basic_stats.update({
                    'approx_distinct_count': row.get('approx_distinct_count', None),
                    'min_value': row.get('min_value', None),
                    'max_value': row.get('max_value', None)
                })
            
            return basic_stats
            
        except Exception as e:
            logger.error(f"Failed to get column stats for {database}.{table}.{column}: {e}")
            return super().get_column_stats(database, table, column)
    
    def get_catalogs(self) -> List[str]:
        """Get list of available catalogs."""
        query = "SHOW CATALOGS"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def get_functions(self) -> List[str]:
        """Get list of available functions."""
        query = "SHOW FUNCTIONS"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def analyze_query_plan(self, query: str) -> Dict[str, Any]:
        """Analyze query execution plan."""
        try:
            explain_query = f"EXPLAIN (FORMAT JSON) {query}"
            result = self.execute_query(explain_query)
            
            if not result.empty:
                plan_json = result.iloc[0, 0]
                return {
                    'query': query,
                    'plan': plan_json,
                    'analyzed': True
                }
            
            return {'query': query, 'analyzed': False}
            
        except Exception as e:
            logger.error(f"Failed to analyze query plan: {e}")
            return {'query': query, 'error': str(e)}
    
    def get_table_properties(self, database: str, table: str) -> Dict[str, Any]:
        """Get table properties and metadata."""
        try:
            query = f"SHOW CREATE TABLE {self.catalog}.{database}.{table}"
            result = self.execute_query(query)
            
            if not result.empty:
                create_statement = result.iloc[0, 0]
                
                properties = {
                    'create_statement': create_statement,
                    'partitioned': 'PARTITIONED BY' in create_statement.upper(),
                    'bucketed': 'CLUSTERED BY' in create_statement.upper(),
                    'file_format': self._extract_file_format(create_statement),
                    'compression': self._extract_compression(create_statement),
                    'location': self._extract_location(create_statement)
                }
                
                return properties
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get table properties for {database}.{table}: {e}")
            return {'error': str(e)}
    
    def _extract_file_format(self, create_statement: str) -> Optional[str]:
        """Extract file format from CREATE TABLE statement."""
        formats = ['PARQUET', 'ORC', 'AVRO', 'JSON', 'CSV', 'TEXTFILE', 'RCFILE', 'SEQUENCEFILE']
        create_upper = create_statement.upper()
        
        for format_name in formats:
            if format_name in create_upper:
                return format_name
        
        return None
    
    def _extract_compression(self, create_statement: str) -> Optional[str]:
        """Extract compression format from CREATE TABLE statement."""
        compressions = ['GZIP', 'SNAPPY', 'LZO', 'BZIP2', 'DEFLATE', 'LZ4']
        create_upper = create_statement.upper()
        
        for compression in compressions:
            if compression in create_upper:
                return compression
        
        return None
    
    def _extract_location(self, create_statement: str) -> Optional[str]:
        """Extract table location from CREATE TABLE statement."""
        try:
            if 'LOCATION' in create_statement.upper():
                parts = create_statement.split("'")
                for i, part in enumerate(parts):
                    if 'LOCATION' in parts[i-1].upper() if i > 0 else False:
                        return part
            return None
        except Exception:
            return None

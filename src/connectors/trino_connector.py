"""Trino connector for data lake access."""

from typing import List, Dict, Any, Optional
import pandas as pd
import trino
from loguru import logger

from .base import BaseConnector, ConnectionError, QueryError


class TrinoConnector(BaseConnector):
    """Connector for Trino."""
    
    def __init__(self, host: str, port: int, username: str, password: Optional[str] = None,
                 catalog: str = "hive", schema: str = "default", **kwargs):
        super().__init__(host, port, username, **kwargs)
        self.password = password
        self.catalog = catalog
        self.schema = schema
    
    def connect(self) -> None:
        """Establish connection to Trino."""
        try:
            auth = None
            if self.password:
                auth = trino.auth.BasicAuthentication(self.username, self.password)
            
            self.connection = trino.dbapi.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                catalog=self.catalog,
                schema=self.schema,
                auth=auth
            )
            self._connected = True
            logger.info(f"Connected to Trino at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Trino: {e}")
            raise ConnectionError(f"Failed to connect to Trino: {e}")
    
    def disconnect(self) -> None:
        """Close connection to Trino."""
        if self.connection:
            try:
                self.connection.close()
                self._connected = False
                logger.info("Disconnected from Trino")
            except Exception as e:
                logger.warning(f"Error during Trino disconnection: {e}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        if not self._connected:
            raise ConnectionError("Not connected to Trino")
        
        try:
            logger.debug(f"Executing Trino query: {query}")
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
        """Get list of available schemas (databases in Trino terminology)."""
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
            # Get basic table information
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
                column_name = row['column_name']
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
            
            # Get additional Trino-specific stats
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
    
    def analyze_table_performance(self, database: str, table: str) -> Dict[str, Any]:
        """Analyze table performance characteristics."""
        try:
            # Get table properties
            query = f"SHOW CREATE TABLE {self.catalog}.{database}.{table}"
            result = self.execute_query(query)
            
            create_statement = result.iloc[0, 0] if not result.empty else ""
            
            # Extract relevant information
            analysis = {
                'partitioned': 'PARTITIONED BY' in create_statement.upper(),
                'bucketed': 'CLUSTERED BY' in create_statement.upper(),
                'compressed': any(format_name in create_statement.upper() 
                                for format_name in ['PARQUET', 'ORC', 'GZIP', 'SNAPPY']),
                'file_format': self._extract_file_format(create_statement),
                'partition_columns': self._extract_partition_columns(create_statement)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze table performance for {database}.{table}: {e}")
            return {'error': str(e)}
    
    def _extract_file_format(self, create_statement: str) -> Optional[str]:
        """Extract file format from CREATE TABLE statement."""
        formats = ['PARQUET', 'ORC', 'AVRO', 'JSON', 'CSV', 'TEXTFILE']
        create_upper = create_statement.upper()
        
        for format_name in formats:
            if format_name in create_upper:
                return format_name
        
        return None
    
    def _extract_partition_columns(self, create_statement: str) -> List[str]:
        """Extract partition columns from CREATE TABLE statement."""
        try:
            if 'PARTITIONED BY' not in create_statement.upper():
                return []
            
            # Simple extraction - this could be made more robust
            parts = create_statement.upper().split('PARTITIONED BY')[1]
            if '(' in parts:
                partition_part = parts.split('(')[1].split(')')[0]
                # Extract column names (simplified)
                columns = [col.strip().split()[0] for col in partition_part.split(',')]
                return columns
            
            return []
            
        except Exception:
            return []

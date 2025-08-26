"""Schema analysis engine for understanding data structures and relationships."""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import re
from loguru import logger

from connectors.base import BaseConnector


@dataclass
class ColumnInfo:
    """Information about a table column."""
    name: str
    data_type: str
    nullable: bool = True
    comment: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    referenced_table: Optional[str] = None
    referenced_column: Optional[str] = None
    
    # Statistics
    distinct_count: Optional[int] = None
    null_count: Optional[int] = None
    null_percentage: Optional[float] = None
    min_value: Any = None
    max_value: Any = None
    avg_length: Optional[float] = None
    
    # Data quality indicators
    data_quality_score: Optional[float] = None
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class TableInfo:
    """Information about a table."""
    database: str
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    last_modified: Optional[str] = None
    
    # Partitioning and clustering
    partition_columns: List[str] = field(default_factory=list)
    clustering_columns: List[str] = field(default_factory=list)
    
    # Table properties
    file_format: Optional[str] = None
    compression: Optional[str] = None
    location: Optional[str] = None
    
    # Relationships
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    referenced_by: List[Dict[str, str]] = field(default_factory=list)
    
    # Analysis results
    table_type: Optional[str] = None  # fact, dimension, bridge, etc.
    data_quality_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)


class SchemaAnalyzer:
    """Analyzes database schemas to understand structure and relationships."""
    
    def __init__(self, connector: BaseConnector):
        self.connector = connector
        self.tables: Dict[str, TableInfo] = {}
        self.relationships: List[Dict[str, Any]] = []
        
    def analyze_database(self, database: str, 
                        tables: Optional[List[str]] = None,
                        sample_data: bool = True) -> Dict[str, Any]:
        """Analyze a complete database schema."""
        logger.info(f"Starting schema analysis for database: {database}")
        
        try:
            # Get list of tables to analyze
            if tables is None:
                tables = self.connector.get_tables(database)
            
            logger.info(f"Found {len(tables)} tables to analyze")
            
            # Analyze each table
            analyzed_tables = {}
            for table_name in tables:
                try:
                    table_info = self.analyze_table(database, table_name, sample_data)
                    analyzed_tables[table_name] = table_info
                    self.tables[f"{database}.{table_name}"] = table_info
                except Exception as e:
                    logger.error(f"Failed to analyze table {database}.{table_name}: {e}")
                    continue
            
            # Detect relationships between tables
            self.detect_relationships(database, list(analyzed_tables.keys()))
            
            # Generate database-level insights
            insights = self.generate_database_insights(database, analyzed_tables)
            
            return {
                'database': database,
                'tables': analyzed_tables,
                'relationships': self.relationships,
                'insights': insights,
                'summary': {
                    'total_tables': len(analyzed_tables),
                    'total_columns': sum(len(table.columns) for table in analyzed_tables.values()),
                    'total_rows': sum(table.row_count or 0 for table in analyzed_tables.values()),
                    'total_size_bytes': sum(table.size_bytes or 0 for table in analyzed_tables.values())
                }
            }
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            raise
    
    def analyze_table(self, database: str, table_name: str, 
                     sample_data: bool = True) -> TableInfo:
        """Analyze a single table."""
        logger.info(f"Analyzing table: {database}.{table_name}")
        
        # Get table schema
        schema = self.connector.get_table_schema(database, table_name)
        
        # Get table statistics
        stats = self.connector.get_table_stats(database, table_name)
        
        # Create table info
        table_info = TableInfo(
            database=database,
            name=table_name,
            row_count=stats.get('row_count'),
            size_bytes=stats.get('size_bytes'),
            last_modified=stats.get('last_modified'),
            location=stats.get('location')
        )
        
        # Analyze columns
        for col_schema in schema:
            column_info = self.analyze_column(
                database, table_name, col_schema, sample_data
            )
            table_info.columns.append(column_info)
        
        # Detect table type
        table_info.table_type = self.detect_table_type(table_info)
        
        # Calculate data quality score
        table_info.data_quality_score = self.calculate_table_quality_score(table_info)
        
        # Generate recommendations
        table_info.recommendations = self.generate_table_recommendations(table_info)
        
        return table_info
    
    def analyze_column(self, database: str, table_name: str, 
                      col_schema: Dict[str, Any], 
                      sample_data: bool = True) -> ColumnInfo:
        """Analyze a single column."""
        column_name = col_schema['column_name']
        
        column_info = ColumnInfo(
            name=column_name,
            data_type=col_schema['data_type'],
            nullable=col_schema.get('nullable', True),
            comment=col_schema.get('comment')
        )
        
        if sample_data:
            try:
                # Get column statistics
                col_stats = self.connector.get_column_stats(database, table_name, column_name)
                
                column_info.distinct_count = col_stats.get('distinct_count')
                column_info.null_count = col_stats.get('null_count')
                column_info.null_percentage = col_stats.get('null_percentage')
                
                # Get sample data for additional analysis
                sample_df = self.connector.get_sample_data(database, table_name, limit=1000)
                if not sample_df.empty and column_name in sample_df.columns:
                    self.analyze_column_data_quality(column_info, sample_df[column_name])
                
            except Exception as e:
                logger.warning(f"Could not get detailed stats for {database}.{table_name}.{column_name}: {e}")
        
        # Detect column patterns
        self.detect_column_patterns(column_info)
        
        return column_info
    
    def analyze_column_data_quality(self, column_info: ColumnInfo, data: pd.Series):
        """Analyze data quality for a column."""
        quality_issues = []
        
        # Check for null values
        null_pct = (data.isnull().sum() / len(data)) * 100
        if null_pct > 50:
            quality_issues.append(f"High null percentage: {null_pct:.1f}%")
        
        # Check for duplicates in potential key columns
        if self.is_potential_key_column(column_info.name):
            duplicate_pct = ((len(data) - data.nunique()) / len(data)) * 100
            if duplicate_pct > 10:
                quality_issues.append(f"High duplicate percentage in key column: {duplicate_pct:.1f}%")
        
        # Check data type consistency
        if column_info.data_type.lower().startswith('int'):
            non_numeric = data.dropna().apply(lambda x: not str(x).isdigit()).sum()
            if non_numeric > 0:
                quality_issues.append(f"Non-numeric values in integer column: {non_numeric}")
        
        # Check for unusual patterns
        if data.dtype == 'object':
            # Check for inconsistent formatting
            lengths = data.dropna().str.len()
            if lengths.std() > lengths.mean() * 0.5:
                quality_issues.append("Inconsistent string length formatting")
        
        column_info.quality_issues = quality_issues
        
        # Calculate quality score (0-1)
        base_score = 1.0
        base_score -= min(null_pct / 100 * 0.5, 0.3)  # Penalize high null percentage
        base_score -= len(quality_issues) * 0.1  # Penalize quality issues
        column_info.data_quality_score = max(0.0, base_score)
    
    def detect_column_patterns(self, column_info: ColumnInfo):
        """Detect common column patterns and purposes."""
        name_lower = column_info.name.lower()
        
        # Primary key detection
        if any(pattern in name_lower for pattern in ['id', '_id', 'key', 'pk']):
            if name_lower.endswith('_id') or name_lower == 'id':
                column_info.is_primary_key = True
        
        # Foreign key detection
        if name_lower.endswith('_id') and not column_info.is_primary_key:
            column_info.is_foreign_key = True
            # Try to infer referenced table
            table_name = name_lower.replace('_id', '')
            column_info.referenced_table = table_name
            column_info.referenced_column = 'id'
    
    def is_potential_key_column(self, column_name: str) -> bool:
        """Check if column name suggests it's a key column."""
        name_lower = column_name.lower()
        key_patterns = ['id', '_id', 'key', 'pk', 'code', 'number', 'num']
        return any(pattern in name_lower for pattern in key_patterns)
    
    def detect_table_type(self, table_info: TableInfo) -> str:
        """Detect if table is fact, dimension, bridge, etc."""
        # Simple heuristics for table type detection
        
        # Count different column types
        id_columns = sum(1 for col in table_info.columns if col.is_primary_key)
        fk_columns = sum(1 for col in table_info.columns if col.is_foreign_key)
        measure_columns = sum(1 for col in table_info.columns 
                            if col.data_type.lower() in ['int', 'bigint', 'float', 'double', 'decimal'])
        
        # Dimension table: few foreign keys, many descriptive columns
        if fk_columns <= 2 and len(table_info.columns) > 5:
            if any(col.name.lower() in ['name', 'description', 'title'] for col in table_info.columns):
                return 'dimension'
        
        # Fact table: many foreign keys, many measures
        if fk_columns >= 2 and measure_columns >= 2:
            return 'fact'
        
        # Bridge table: mostly foreign keys
        if fk_columns >= 2 and len(table_info.columns) <= fk_columns + 2:
            return 'bridge'
        
        # Lookup table: small, mostly descriptive
        if table_info.row_count and table_info.row_count < 1000:
            return 'lookup'
        
        return 'unknown'
    
    def detect_relationships(self, database: str, tables: List[str]):
        """Detect relationships between tables."""
        logger.info(f"Detecting relationships between {len(tables)} tables")
        
        relationships = []
        
        for table_name in tables:
            table_key = f"{database}.{table_name}"
            if table_key not in self.tables:
                continue
                
            table_info = self.tables[table_key]
            
            for column in table_info.columns:
                if column.is_foreign_key and column.referenced_table:
                    # Check if referenced table exists
                    referenced_table = column.referenced_table
                    if referenced_table in tables:
                        relationship = {
                            'type': 'foreign_key',
                            'from_table': table_name,
                            'from_column': column.name,
                            'to_table': referenced_table,
                            'to_column': column.referenced_column or 'id',
                            'confidence': 0.8  # Based on naming convention
                        }
                        relationships.append(relationship)
        
        # Detect potential relationships based on data analysis
        self.detect_data_based_relationships(database, tables, relationships)
        
        self.relationships = relationships
        logger.info(f"Detected {len(relationships)} relationships")
    
    def detect_data_based_relationships(self, database: str, tables: List[str], 
                                      relationships: List[Dict[str, Any]]):
        """Detect relationships based on actual data analysis."""
        # This would involve comparing column values across tables
        # For now, we'll implement a simplified version
        
        table_columns = {}
        for table_name in tables:
            table_key = f"{database}.{table_name}"
            if table_key in self.tables:
                table_columns[table_name] = [col.name for col in self.tables[table_key].columns]
        
        # Look for columns with similar names across tables
        for table1 in tables:
            for table2 in tables:
                if table1 >= table2:  # Avoid duplicate comparisons
                    continue
                
                if table1 not in table_columns or table2 not in table_columns:
                    continue
                
                cols1 = set(table_columns[table1])
                cols2 = set(table_columns[table2])
                
                # Find common column names
                common_cols = cols1.intersection(cols2)
                for col in common_cols:
                    if col.lower() in ['id', 'created_at', 'updated_at']:
                        continue  # Skip common system columns
                    
                    # This could be a relationship
                    relationship = {
                        'type': 'potential_join',
                        'table1': table1,
                        'column1': col,
                        'table2': table2,
                        'column2': col,
                        'confidence': 0.6,  # Lower confidence for data-based detection
                        'detected_by': 'common_column_name'
                    }
                    relationships.append(relationship)
    
    def calculate_table_quality_score(self, table_info: TableInfo) -> float:
        """Calculate overall data quality score for a table."""
        if not table_info.columns:
            return 0.0
        
        column_scores = [col.data_quality_score or 0.5 for col in table_info.columns]
        base_score = sum(column_scores) / len(column_scores)
        
        # Adjust based on table characteristics
        if table_info.row_count and table_info.row_count > 0:
            base_score += 0.1  # Bonus for having data
        
        if table_info.partition_columns:
            base_score += 0.05  # Bonus for partitioning
        
        # Penalize if no primary key detected
        has_pk = any(col.is_primary_key for col in table_info.columns)
        if not has_pk:
            base_score -= 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def generate_table_recommendations(self, table_info: TableInfo) -> List[str]:
        """Generate recommendations for table optimization."""
        recommendations = []
        
        # Primary key recommendations
        has_pk = any(col.is_primary_key for col in table_info.columns)
        if not has_pk:
            recommendations.append("Consider adding a primary key for better data integrity")
        
        # Partitioning recommendations
        if not table_info.partition_columns and table_info.row_count and table_info.row_count > 1000000:
            date_columns = [col.name for col in table_info.columns 
                          if 'date' in col.data_type.lower() or 'timestamp' in col.data_type.lower()]
            if date_columns:
                recommendations.append(f"Consider partitioning by date column: {date_columns[0]}")
        
        # Data quality recommendations
        high_null_columns = [col.name for col in table_info.columns 
                           if col.null_percentage and col.null_percentage > 50]
        if high_null_columns:
            recommendations.append(f"Review columns with high null percentages: {', '.join(high_null_columns)}")
        
        # Indexing recommendations
        if table_info.table_type == 'fact':
            fk_columns = [col.name for col in table_info.columns if col.is_foreign_key]
            if fk_columns:
                recommendations.append(f"Consider indexing foreign key columns: {', '.join(fk_columns)}")
        
        return recommendations
    
    def generate_database_insights(self, database: str, 
                                 tables: Dict[str, TableInfo]) -> Dict[str, Any]:
        """Generate high-level insights about the database."""
        insights = {
            'table_types': defaultdict(int),
            'data_quality': {
                'average_score': 0.0,
                'tables_needing_attention': []
            },
            'relationships': {
                'total_detected': len(self.relationships),
                'foreign_keys': 0,
                'potential_joins': 0
            },
            'optimization_opportunities': []
        }
        
        # Analyze table types
        for table in tables.values():
            insights['table_types'][table.table_type or 'unknown'] += 1
        
        # Calculate average data quality
        quality_scores = [table.data_quality_score for table in tables.values() 
                         if table.data_quality_score is not None]
        if quality_scores:
            insights['data_quality']['average_score'] = sum(quality_scores) / len(quality_scores)
            
            # Find tables needing attention
            for name, table in tables.items():
                if table.data_quality_score and table.data_quality_score < 0.7:
                    insights['data_quality']['tables_needing_attention'].append(name)
        
        # Analyze relationships
        for rel in self.relationships:
            if rel['type'] == 'foreign_key':
                insights['relationships']['foreign_keys'] += 1
            elif rel['type'] == 'potential_join':
                insights['relationships']['potential_joins'] += 1
        
        # Generate optimization opportunities
        large_unpartitioned = [name for name, table in tables.items()
                             if table.row_count and table.row_count > 1000000 
                             and not table.partition_columns]
        if large_unpartitioned:
            insights['optimization_opportunities'].append(
                f"Large unpartitioned tables: {', '.join(large_unpartitioned)}"
            )
        
        return insights

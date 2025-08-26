"""Recommendation engine for generating data modeling suggestions."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Recommendation:
    """Data class for a single recommendation."""
    title: str
    category: str
    priority: str  # 'high', 'medium', 'low'
    description: str
    implementation_effort: str  # 'low', 'medium', 'high'
    expected_benefit: str
    confidence: float
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None


class RecommendationEngine:
    """Engine for generating data modeling and optimization recommendations."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommendations = []
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Recommendation]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_results: Results from schema analysis, data profiling, etc.
            
        Returns:
            List of recommendations
        """
        logger.info("Generating recommendations from analysis results")
        
        recommendations = []
        
        # Generate schema-based recommendations
        if 'schema_analysis' in analysis_results:
            schema_recs = self._generate_schema_recommendations(
                analysis_results['schema_analysis']
            )
            recommendations.extend(schema_recs)
        
        # Generate data quality recommendations
        if 'data_profiling' in analysis_results:
            quality_recs = self._generate_quality_recommendations(
                analysis_results['data_profiling']
            )
            recommendations.extend(quality_recs)
        
        # Generate performance recommendations
        if 'table_stats' in analysis_results:
            perf_recs = self._generate_performance_recommendations(
                analysis_results['table_stats']
            )
            recommendations.extend(perf_recs)
        
        # Generate relationship-based recommendations
        if 'relationships' in analysis_results:
            rel_recs = self._generate_relationship_recommendations(
                analysis_results['relationships']
            )
            recommendations.extend(rel_recs)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _generate_schema_recommendations(self, schema_data: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations based on schema analysis."""
        recommendations = []
        
        for table_name, table_info in schema_data.items():
            columns = table_info.get('columns', [])
            
            # Check for missing primary keys
            has_pk = any(col.get('is_primary_key', False) for col in columns)
            if not has_pk:
                recommendations.append(Recommendation(
                    title=f"Add Primary Key to {table_name}",
                    category="schema_design",
                    priority="high",
                    description=f"Table {table_name} appears to lack a primary key, which is essential for data integrity and performance.",
                    implementation_effort="low",
                    expected_benefit="Improved data integrity and query performance",
                    confidence=0.8,
                    table_name=table_name,
                    technical_details={
                        "suggestion": "Add an auto-incrementing ID column or identify existing unique columns"
                    }
                ))
            
            # Check for wide tables (too many columns)
            if len(columns) > 50:
                recommendations.append(Recommendation(
                    title=f"Consider Normalizing Wide Table {table_name}",
                    category="normalization",
                    priority="medium",
                    description=f"Table {table_name} has {len(columns)} columns, which may indicate denormalization opportunities.",
                    implementation_effort="high",
                    expected_benefit="Improved maintainability and reduced storage",
                    confidence=0.6,
                    table_name=table_name,
                    technical_details={
                        "column_count": len(columns),
                        "suggestion": "Consider breaking into multiple related tables"
                    }
                ))
        
        return recommendations
    
    def _generate_quality_recommendations(self, profiling_data: Dict[str, Any]) -> List[Recommendation]:
        """Generate recommendations based on data quality analysis."""
        recommendations = []
        
        for table_name, table_profile in profiling_data.items():
            # Check for high null percentages
            for column_name, column_stats in table_profile.get('columns', {}).items():
                null_percentage = column_stats.get('null_percentage', 0)
                
                if null_percentage > 50:
                    recommendations.append(Recommendation(
                        title=f"High Null Values in {table_name}.{column_name}",
                        category="data_quality",
                        priority="medium",
                        description=f"Column {column_name} in {table_name} has {null_percentage:.1f}% null values.",
                        implementation_effort="medium",
                        expected_benefit="Improved data completeness and query reliability",
                        confidence=0.9,
                        table_name=table_name,
                        column_name=column_name,
                        technical_details={
                            "null_percentage": null_percentage,
                            "suggestion": "Investigate data collection process or consider default values"
                        }
                    ))
                
                # Check for potential data type issues
                data_type = column_stats.get('data_type', '').lower()
                if 'string' in data_type or 'varchar' in data_type:
                    unique_count = column_stats.get('unique_count', 0)
                    total_count = column_stats.get('count', 1)
                    
                    if unique_count / total_count < 0.1 and unique_count < 20:
                        recommendations.append(Recommendation(
                            title=f"Consider Enum/Category for {table_name}.{column_name}",
                            category="data_type_optimization",
                            priority="low",
                            description=f"Column {column_name} has only {unique_count} unique values, consider using enum or category type.",
                            implementation_effort="low",
                            expected_benefit="Reduced storage and improved performance",
                            confidence=0.7,
                            table_name=table_name,
                            column_name=column_name,
                            technical_details={
                                "unique_count": unique_count,
                                "total_count": total_count,
                                "suggestion": "Convert to enum or create lookup table"
                            }
                        ))
        
        return recommendations
    
    def _generate_performance_recommendations(self, stats_data: Dict[str, Any]) -> List[Recommendation]:
        """Generate performance-related recommendations."""
        recommendations = []
        
        for table_name, stats in stats_data.items():
            row_count = stats.get('row_count', 0)
            
            # Large table partitioning recommendations
            if row_count > 1000000:  # 1 million rows
                recommendations.append(Recommendation(
                    title=f"Consider Partitioning Large Table {table_name}",
                    category="partitioning",
                    priority="high",
                    description=f"Table {table_name} has {row_count:,} rows. Partitioning could improve query performance.",
                    implementation_effort="medium",
                    expected_benefit="Significantly improved query performance for date/time-based queries",
                    confidence=0.8,
                    table_name=table_name,
                    technical_details={
                        "row_count": row_count,
                        "suggestion": "Partition by date, region, or other frequently filtered columns"
                    }
                ))
            
            # File format recommendations
            size_bytes = stats.get('size_bytes', 0)
            if size_bytes > 100 * 1024 * 1024:  # 100MB
                recommendations.append(Recommendation(
                    title=f"Optimize File Format for {table_name}",
                    category="file_format",
                    priority="medium",
                    description=f"Large table {table_name} could benefit from columnar storage format.",
                    implementation_effort="medium",
                    expected_benefit="30-70% reduction in storage and improved query performance",
                    confidence=0.7,
                    table_name=table_name,
                    technical_details={
                        "size_bytes": size_bytes,
                        "suggestion": "Consider Parquet or ORC format with appropriate compression"
                    }
                ))
        
        return recommendations
    
    def _generate_relationship_recommendations(self, relationships_data: List[Dict[str, Any]]) -> List[Recommendation]:
        """Generate recommendations based on table relationships."""
        recommendations = []
        
        # Analyze relationship patterns
        table_connections = {}
        for rel in relationships_data:
            from_table = rel['from_table']
            to_table = rel['to_table']
            
            if from_table not in table_connections:
                table_connections[from_table] = {'outgoing': 0, 'incoming': 0}
            if to_table not in table_connections:
                table_connections[to_table] = {'outgoing': 0, 'incoming': 0}
            
            table_connections[from_table]['outgoing'] += 1
            table_connections[to_table]['incoming'] += 1
        
        # Identify potential fact and dimension tables
        for table_name, connections in table_connections.items():
            total_connections = connections['outgoing'] + connections['incoming']
            
            if connections['outgoing'] > 3:  # Many foreign keys = potential fact table
                recommendations.append(Recommendation(
                    title=f"Consider {table_name} as Fact Table in Star Schema",
                    category="dimensional_modeling",
                    priority="medium",
                    description=f"Table {table_name} has multiple foreign key relationships, suggesting it could be a fact table.",
                    implementation_effort="high",
                    expected_benefit="Improved analytics performance with star schema design",
                    confidence=0.6,
                    table_name=table_name,
                    technical_details={
                        "outgoing_relationships": connections['outgoing'],
                        "suggestion": "Design star schema with this as central fact table"
                    }
                ))
            
            elif connections['incoming'] > 2 and connections['outgoing'] == 0:  # Referenced by many = dimension
                recommendations.append(Recommendation(
                    title=f"Optimize {table_name} as Dimension Table",
                    category="dimensional_modeling",
                    priority="low",
                    description=f"Table {table_name} is referenced by multiple tables, suggesting it's a dimension table.",
                    implementation_effort="medium",
                    expected_benefit="Better organized dimensional model",
                    confidence=0.7,
                    table_name=table_name,
                    technical_details={
                        "incoming_relationships": connections['incoming'],
                        "suggestion": "Add surrogate keys and slowly changing dimension handling if needed"
                    }
                ))
        
        return recommendations
    
    def filter_recommendations(self, recommendations: List[Recommendation],
                             category: Optional[str] = None,
                             priority: Optional[str] = None,
                             table_name: Optional[str] = None) -> List[Recommendation]:
        """Filter recommendations by various criteria."""
        filtered = recommendations
        
        if category:
            filtered = [r for r in filtered if r.category == category]
        
        if priority:
            filtered = [r for r in filtered if r.priority == priority]
        
        if table_name:
            filtered = [r for r in filtered if r.table_name == table_name]
        
        return filtered
    
    def sort_recommendations(self, recommendations: List[Recommendation],
                           sort_by: str = "priority") -> List[Recommendation]:
        """Sort recommendations by various criteria."""
        if sort_by == "priority":
            priority_order = {"high": 0, "medium": 1, "low": 2}
            return sorted(recommendations, key=lambda r: priority_order.get(r.priority, 3))
        elif sort_by == "confidence":
            return sorted(recommendations, key=lambda r: r.confidence, reverse=True)
        elif sort_by == "category":
            return sorted(recommendations, key=lambda r: r.category)
        else:
            return recommendations

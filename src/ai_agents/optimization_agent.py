"""AI agent for performance optimization recommendations."""

from typing import Dict, List, Any, Optional
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class OptimizationAgent(BaseAgent):
    """AI agent that provides performance optimization recommendations."""
    
    def __init__(self):
        super().__init__("OptimizationAgent")
        
        self.system_prompt = """
You are a database performance optimization expert specializing in big data platforms 
like Hive, Trino, and Presto. Your role is to analyze database schemas, query patterns, 
and data characteristics to provide specific performance optimization recommendations.

Focus on:
1. Partitioning strategies for large tables
2. Clustering and bucketing recommendations
3. File format optimizations (Parquet, ORC, etc.)
4. Compression strategies
5. Indexing recommendations
6. Query optimization techniques
7. Resource allocation and configuration
8. Data layout optimization

Provide specific, technical recommendations with implementation details and 
expected performance improvements. Consider the trade-offs between storage, 
compute, and query performance.
"""
    
    def analyze(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Analyze data and provide optimization recommendations."""
        try:
            logger.info("OptimizationAgent analyzing for performance optimization")
            
            # Prepare the analysis prompt
            user_prompt = self._create_optimization_prompt(data, context)
            
            # Get recommendations from LLM
            llm_response = self._make_llm_request(self.system_prompt, user_prompt)
            
            # Parse recommendations
            recommendations = self._parse_optimization_recommendations(llm_response)
            
            # Add technical details to recommendations
            self._enhance_recommendations_with_technical_details(recommendations, data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, recommendations)
            
            return AgentResponse(
                success=True,
                recommendations=recommendations,
                reasoning=llm_response,
                confidence=confidence,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'performance_optimization',
                    'tables_analyzed': len(data.get('tables', {})),
                    'optimization_categories': list(set(rec['category'] for rec in recommendations))
                }
            )
            
        except Exception as e:
            logger.error(f"OptimizationAgent analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _create_optimization_prompt(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Create the optimization analysis prompt."""
        prompt = "Please analyze the following database schema for performance optimization opportunities:\n\n"
        
        # Add table information with focus on performance characteristics
        if 'tables' in data:
            prompt += "TABLES WITH PERFORMANCE CHARACTERISTICS:\n"
            for table_name, table_info in data['tables'].items():
                prompt += self._format_table_for_optimization(table_info)
                prompt += "\n"
        
        # Add query patterns if available
        if context and 'query_patterns' in context:
            prompt += f"QUERY PATTERNS:\n{context['query_patterns']}\n\n"
        
        # Add performance requirements
        if context and 'performance_requirements' in context:
            prompt += f"PERFORMANCE REQUIREMENTS:\n{context['performance_requirements']}\n\n"
        
        # Add data volume information
        if context and 'data_volume' in context:
            prompt += f"DATA VOLUME:\n{context['data_volume']}\n\n"
        
        prompt += """
Please provide specific optimization recommendations in these areas:

1. **Partitioning Strategy**: 
   - Recommend partitioning columns based on query patterns
   - Suggest partition granularity (daily, monthly, etc.)
   - Identify over-partitioning or under-partitioning issues

2. **File Format and Compression**:
   - Recommend optimal file formats (Parquet, ORC, Avro)
   - Suggest compression algorithms (Snappy, GZIP, LZ4)
   - Consider read vs write performance trade-offs

3. **Clustering and Bucketing**:
   - Recommend clustering columns for frequently joined tables
   - Suggest bucketing strategies for large fact tables
   - Optimize for common query patterns

4. **Indexing Strategy**:
   - Recommend indexes for frequently queried columns
   - Suggest composite indexes for multi-column queries
   - Balance index maintenance cost vs query performance

5. **Data Layout Optimization**:
   - Recommend column ordering for better compression
   - Suggest data distribution strategies
   - Optimize for scan performance

6. **Query Optimization**:
   - Identify potential query performance bottlenecks
   - Recommend query rewriting opportunities
   - Suggest materialized view candidates

For each recommendation, include:
- Specific implementation steps
- Expected performance improvement (quantified if possible)
- Implementation complexity and effort
- Potential trade-offs or risks
- Priority level (high/medium/low)
"""
        
        return prompt
    
    def _format_table_for_optimization(self, table_info: Dict[str, Any]) -> str:
        """Format table information focusing on optimization aspects."""
        formatted = f"Table: {table_info.get('database', 'unknown')}.{table_info.get('name', 'unknown')}\n"
        formatted += f"Rows: {table_info.get('row_count', 'unknown'):,}\n" if table_info.get('row_count') else "Rows: unknown\n"
        formatted += f"Size: {self._format_size(table_info.get('size_bytes'))}\n"
        formatted += f"File Format: {table_info.get('file_format', 'unknown')}\n"
        formatted += f"Compression: {table_info.get('compression', 'unknown')}\n"
        
        # Partitioning information
        partition_cols = table_info.get('partition_columns', [])
        if partition_cols:
            formatted += f"Partitioned by: {', '.join(partition_cols)}\n"
        else:
            formatted += "Not partitioned\n"
        
        # Clustering information
        clustering_cols = table_info.get('clustering_columns', [])
        if clustering_cols:
            formatted += f"Clustered by: {', '.join(clustering_cols)}\n"
        
        # Column information relevant to optimization
        formatted += "\nColumns (with optimization relevance):\n"
        for col in table_info.get('columns', []):
            col_line = f"- {col.get('name', 'unknown')} ({col.get('data_type', 'unknown')})"
            
            # Add cardinality information
            if col.get('unique_count') and col.get('total_count'):
                cardinality = col['unique_count'] / col['total_count']
                if cardinality < 0.01:
                    col_line += " [low cardinality]"
                elif cardinality > 0.9:
                    col_line += " [high cardinality]"
            
            # Add null percentage
            if col.get('null_percentage'):
                col_line += f" [null: {col['null_percentage']:.1f}%]"
            
            # Mark potential optimization columns
            col_name_lower = col.get('name', '').lower()
            if any(pattern in col_name_lower for pattern in ['date', 'time', 'created', 'updated']):
                col_line += " [potential partition key]"
            elif col.get('is_foreign_key'):
                col_line += " [potential clustering key]"
            
            formatted += col_line + "\n"
        
        return formatted
    
    def _format_size(self, size_bytes: Optional[int]) -> str:
        """Format size in human-readable format."""
        if not size_bytes:
            return "unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
    
    def _parse_optimization_recommendations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured optimization recommendations."""
        recommendations = self._parse_llm_recommendations(llm_response)
        
        # Enhance with optimization-specific categorization
        for rec in recommendations:
            rec['category'] = self._categorize_optimization_recommendation(rec['title'])
            rec['type'] = 'performance_optimization'
            
            # Extract performance impact if mentioned
            rec['performance_impact'] = self._extract_performance_impact(rec['description'])
            
            # Extract implementation complexity
            rec['implementation_complexity'] = self._extract_complexity(rec['description'])
        
        return recommendations
    
    def _categorize_optimization_recommendation(self, title: str) -> str:
        """Categorize optimization recommendation."""
        title_lower = title.lower()
        
        if 'partition' in title_lower:
            return 'partitioning'
        elif 'index' in title_lower:
            return 'indexing'
        elif 'compress' in title_lower or 'format' in title_lower:
            return 'storage_optimization'
        elif 'cluster' in title_lower or 'bucket' in title_lower:
            return 'data_layout'
        elif 'query' in title_lower:
            return 'query_optimization'
        elif 'cache' in title_lower or 'memory' in title_lower:
            return 'caching'
        else:
            return 'general_optimization'
    
    def _extract_performance_impact(self, description: str) -> str:
        """Extract expected performance impact from description."""
        desc_lower = description.lower()
        
        if any(term in desc_lower for term in ['significant', '50%', '2x', 'dramatic']):
            return 'high'
        elif any(term in desc_lower for term in ['moderate', '20%', '30%', 'noticeable']):
            return 'medium'
        elif any(term in desc_lower for term in ['minor', '10%', 'slight', 'small']):
            return 'low'
        else:
            return 'unknown'
    
    def _extract_complexity(self, description: str) -> str:
        """Extract implementation complexity from description."""
        desc_lower = description.lower()
        
        if any(term in desc_lower for term in ['complex', 'difficult', 'major', 'significant effort']):
            return 'high'
        elif any(term in desc_lower for term in ['moderate', 'medium', 'some effort']):
            return 'medium'
        elif any(term in desc_lower for term in ['simple', 'easy', 'straightforward', 'quick']):
            return 'low'
        else:
            return 'medium'
    
    def _enhance_recommendations_with_technical_details(self, recommendations: List[Dict[str, Any]], data: Dict[str, Any]):
        """Add technical implementation details to recommendations."""
        for rec in recommendations:
            category = rec.get('category', '')
            
            if category == 'partitioning':
                rec['technical_details'] = self._get_partitioning_details(rec, data)
            elif category == 'indexing':
                rec['technical_details'] = self._get_indexing_details(rec, data)
            elif category == 'storage_optimization':
                rec['technical_details'] = self._get_storage_optimization_details(rec, data)
            elif category == 'data_layout':
                rec['technical_details'] = self._get_data_layout_details(rec, data)
    
    def _get_partitioning_details(self, recommendation: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Get technical details for partitioning recommendations."""
        return {
            'implementation': 'Use PARTITIONED BY clause in CREATE TABLE or ALTER TABLE statements',
            'best_practices': 'Aim for partitions with 100MB-1GB of data each',
            'monitoring': 'Monitor partition pruning in query execution plans',
            'maintenance': 'Consider partition lifecycle management for time-based partitions'
        }
    
    def _get_indexing_details(self, recommendation: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Get technical details for indexing recommendations."""
        return {
            'implementation': 'Create indexes on frequently queried columns in WHERE and JOIN clauses',
            'best_practices': 'Consider composite indexes for multi-column queries',
            'monitoring': 'Monitor index usage statistics and maintenance overhead',
            'maintenance': 'Regular index maintenance and statistics updates required'
        }
    
    def _get_storage_optimization_details(self, recommendation: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Get technical details for storage optimization recommendations."""
        return {
            'implementation': 'Use STORED AS PARQUET or STORED AS ORC in table definitions',
            'compression_options': 'SNAPPY for balance, GZIP for maximum compression, LZ4 for speed',
            'monitoring': 'Monitor compression ratios and query performance impact',
            'migration': 'Use INSERT OVERWRITE to convert existing tables to optimized format'
        }
    
    def _get_data_layout_details(self, recommendation: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, str]:
        """Get technical details for data layout recommendations."""
        return {
            'implementation': 'Use CLUSTERED BY clause for bucketing, optimize column order',
            'best_practices': 'Cluster on frequently joined columns, order columns by query frequency',
            'monitoring': 'Monitor data skew and bucket distribution',
            'maintenance': 'Consider re-clustering for heavily updated tables'
        }
    
    def analyze_query_performance(self, query_patterns: List[str], schema_data: Dict[str, Any]) -> AgentResponse:
        """Analyze specific query patterns for optimization opportunities."""
        try:
            prompt = f"""
Analyze these query patterns for optimization opportunities:

QUERY PATTERNS:
{chr(10).join(f"{i+1}. {pattern}" for i, pattern in enumerate(query_patterns))}

SCHEMA CONTEXT:
{self._format_schema_analysis(schema_data)}

Please provide specific recommendations for:
1. Query rewriting opportunities
2. Index recommendations for these queries
3. Materialized view candidates
4. Partitioning optimizations for these access patterns
5. Join optimization strategies
"""
            
            llm_response = self._make_llm_request(self.system_prompt, prompt)
            recommendations = self._parse_optimization_recommendations(llm_response)
            
            return AgentResponse(
                success=True,
                recommendations=recommendations,
                reasoning=llm_response,
                confidence=0.8,  # High confidence for query-specific analysis
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'query_optimization',
                    'query_patterns_analyzed': len(query_patterns)
                }
            )
            
        except Exception as e:
            logger.error(f"Query performance analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )

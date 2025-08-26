"""Main analysis engine that orchestrates all components."""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time

from config import get_config
from connectors.base import BaseConnector
from connectors.hive_connector import HiveConnector
from connectors.trino_connector import TrinoConnector
from connectors.presto_connector import PrestoConnector
from analyzers.schema_analyzer import SchemaAnalyzer
from analyzers.data_profiler import DataProfiler
from ai_agents.workflow_engine import WorkflowEngine, WorkflowStage, WorkflowTask


@dataclass
class AnalysisRequest:
    """Request for data analysis."""
    connection_type: str  # 'hive', 'trino', 'presto'
    database: str
    tables: Optional[List[str]] = None
    include_data_profiling: bool = True
    include_ai_recommendations: bool = True
    sample_size: int = 10000
    context: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """Result of data analysis."""
    request: AnalysisRequest
    success: bool
    schema_analysis: Optional[Dict[str, Any]] = None
    data_profiles: Optional[Dict[str, Any]] = None
    ai_recommendations: Optional[List[Dict[str, Any]]] = None
    workflow_insights: Optional[Dict[str, Any]] = None
    agent_results: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class AnalysisEngine:
    """Main engine for orchestrating data analysis and recommendations."""
    
    def __init__(self):
        self.config = get_config()
        self.workflow_engine = WorkflowEngine(max_parallel_tasks=self.config.analysis.max_concurrent_tables)
        
    def create_connector(self, connection_type: str) -> BaseConnector:
        """Create appropriate connector based on type."""
        connection_type = connection_type.lower()
        
        if connection_type == 'hive':
            hive_config = self.config.data_lakes.get('hive')
            if not hive_config:
                raise ValueError("Hive configuration not found")
            
            return HiveConnector(
                host=hive_config.host,
                port=hive_config.port,
                username=hive_config.username,
                password=hive_config.password,
                database=hive_config.database or 'default'
            )
            
        elif connection_type == 'trino':
            trino_config = self.config.data_lakes.get('trino')
            if not trino_config:
                raise ValueError("Trino configuration not found")
            
            return TrinoConnector(
                host=trino_config.host,
                port=trino_config.port,
                username=trino_config.username,
                password=trino_config.password,
                catalog=trino_config.catalog or 'hive',
                schema=trino_config.schema or 'default'
            )
            
        elif connection_type == 'presto':
            presto_config = self.config.data_lakes.get('presto')
            if not presto_config:
                raise ValueError("Presto configuration not found")
            
            return PrestoConnector(
                host=presto_config.host,
                port=presto_config.port,
                username=presto_config.username,
                catalog=presto_config.catalog or 'hive',
                schema=presto_config.schema or 'default'
            )
            
        else:
            raise ValueError(f"Unsupported connection type: {connection_type}")
    
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform comprehensive data analysis."""
        start_time = time.time()
        logger.info(f"Starting analysis for {request.connection_type}://{request.database}")
        
        try:
            # Create connector
            connector = self.create_connector(request.connection_type)
            
            # Test connection
            with connector:
                if not connector.test_connection():
                    raise ConnectionError(f"Failed to connect to {request.connection_type}")
                
                # Perform analysis steps
                result = AnalysisResult(
                    request=request,
                    success=True,
                    execution_time=0.0,
                    metadata={
                        'connection_type': request.connection_type,
                        'database': request.database,
                        'analysis_timestamp': time.time()
                    }
                )
                
                # Step 1: Schema Analysis
                logger.info("Performing schema analysis...")
                result.schema_analysis = self._perform_schema_analysis(
                    connector, request.database, request.tables
                )
                
                # Step 2: Data Profiling (if requested)
                if request.include_data_profiling:
                    logger.info("Performing data profiling...")
                    result.data_profiles = self._perform_data_profiling(
                        connector, request.database, request.tables, request.sample_size
                    )
                
                # Step 3: AI Workflow Execution (if requested)
                if request.include_ai_recommendations:
                    logger.info("Executing AI agent workflow...")
                    workflow_result = self._execute_ai_workflow(
                        result.schema_analysis, result.data_profiles, request.context
                    )
                    result.ai_recommendations = workflow_result.consolidated_recommendations
                    result.workflow_insights = workflow_result.workflow_insights
                    result.agent_results = workflow_result.stage_results
                
                # Calculate execution time
                result.execution_time = time.time() - start_time
                
                logger.info(f"Analysis completed successfully in {result.execution_time:.2f} seconds")
                return result
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                request=request,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={'connection_type': request.connection_type, 'database': request.database}
            )
    
    def _perform_schema_analysis(self, connector: BaseConnector, database: str, 
                                tables: Optional[List[str]]) -> Dict[str, Any]:
        """Perform schema analysis."""
        analyzer = SchemaAnalyzer(connector)
        return analyzer.analyze_database(database, tables, sample_data=True)
    
    def _perform_data_profiling(self, connector: BaseConnector, database: str,
                               tables: Optional[List[str]], sample_size: int) -> Dict[str, Any]:
        """Perform data profiling."""
        profiler = DataProfiler(connector, sample_size)
        profiles = {}
        
        # Get tables to profile
        tables_to_profile = tables or connector.get_tables(database)
        
        # Profile tables (with potential parallelization)
        max_workers = min(self.config.analysis.max_concurrent_tables, len(tables_to_profile))
        
        if max_workers > 1:
            # Parallel profiling
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_table = {
                    executor.submit(profiler.profile_table, database, table): table
                    for table in tables_to_profile
                }
                
                for future in as_completed(future_to_table):
                    table_name = future_to_table[future]
                    try:
                        table_profile = future.result()
                        profiles[table_name] = self._serialize_table_profile(table_profile)
                    except Exception as e:
                        logger.error(f"Failed to profile table {table_name}: {e}")
                        profiles[table_name] = {'error': str(e)}
        else:
            # Sequential profiling
            for table_name in tables_to_profile:
                try:
                    table_profile = profiler.profile_table(database, table_name)
                    profiles[table_name] = self._serialize_table_profile(table_profile)
                except Exception as e:
                    logger.error(f"Failed to profile table {table_name}: {e}")
                    profiles[table_name] = {'error': str(e)}
        
        return profiles
    
    def _serialize_table_profile(self, table_profile) -> Dict[str, Any]:
        """Convert table profile to serializable dictionary."""
        # Convert dataclass to dict, handling nested objects
        result = {
            'database': table_profile.database,
            'table_name': table_profile.table_name,
            'total_rows': table_profile.total_rows,
            'total_columns': table_profile.total_columns,
            'completeness_score': table_profile.completeness_score,
            'consistency_score': table_profile.consistency_score,
            'overall_quality_score': table_profile.overall_quality_score,
            'potential_keys': table_profile.potential_keys,
            'potential_foreign_keys': table_profile.potential_foreign_keys,
            'table_classification': table_profile.table_classification,
            'recommended_partitioning': table_profile.recommended_partitioning,
            'recommended_indexing': table_profile.recommended_indexing,
            'table_level_issues': table_profile.table_level_issues,
            'column_profiles': {}
        }
        
        # Convert column profiles
        for col_name, col_profile in table_profile.column_profiles.items():
            result['column_profiles'][col_name] = {
                'column_name': col_profile.column_name,
                'data_type': col_profile.data_type,
                'total_count': col_profile.total_count,
                'null_count': col_profile.null_count,
                'null_percentage': col_profile.null_percentage,
                'unique_count': col_profile.unique_count,
                'unique_percentage': col_profile.unique_percentage,
                'quality_score': col_profile.quality_score,
                'quality_issues': col_profile.quality_issues,
                'potential_pii': col_profile.potential_pii,
                'potential_key': col_profile.potential_key,
                'data_classification': col_profile.data_classification,
                'most_frequent_values': col_profile.most_frequent_values,
                'common_patterns': col_profile.common_patterns,
                'anomalies': col_profile.anomalies
            }
            
            # Add numeric-specific fields
            if col_profile.mean is not None:
                result['column_profiles'][col_name].update({
                    'min_value': col_profile.min_value,
                    'max_value': col_profile.max_value,
                    'mean': col_profile.mean,
                    'median': col_profile.median,
                    'std_dev': col_profile.std_dev
                })
            
            # Add text-specific fields
            if col_profile.avg_length is not None:
                result['column_profiles'][col_name].update({
                    'min_length': col_profile.min_length,
                    'max_length': col_profile.max_length,
                    'avg_length': col_profile.avg_length
                })
        
        return result
    
    def _execute_ai_workflow(self, schema_analysis: Optional[Dict[str, Any]],
                           data_profiles: Optional[Dict[str, Any]],
                           context: Optional[Dict[str, Any]]):
        """Execute the AI agent workflow."""
        # Combine data for workflow
        workflow_data = {}
        if schema_analysis:
            workflow_data.update(schema_analysis)
        if data_profiles:
            workflow_data['data_profiles'] = data_profiles
        
        # Execute the workflow
        workflow_result = self.workflow_engine.execute_workflow(workflow_data, context)
        
        if not workflow_result.success:
            logger.error(f"Workflow execution failed: {workflow_result.error_message}")
        
        return workflow_result
    
    def analyze_multiple_databases(self, requests: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Analyze multiple databases in parallel."""
        logger.info(f"Starting analysis of {len(requests)} databases")
        
        results = []
        max_workers = min(self.config.analysis.max_concurrent_tables, len(requests))
        
        if max_workers > 1:
            # Parallel analysis
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_request = {
                    executor.submit(self.analyze, request): request
                    for request in requests
                }
                
                for future in as_completed(future_to_request):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        request = future_to_request[future]
                        logger.error(f"Failed to analyze {request.database}: {e}")
                        results.append(AnalysisResult(
                            request=request,
                            success=False,
                            error_message=str(e)
                        ))
        else:
            # Sequential analysis
            for request in requests:
                result = self.analyze(request)
                results.append(result)
        
        return results
    
    def test_connections(self) -> Dict[str, bool]:
        """Test all configured connections."""
        results = {}
        
        for conn_type in ['hive', 'trino', 'presto']:
            try:
                connector = self.create_connector(conn_type)
                results[conn_type] = connector.test_connection()
            except Exception as e:
                logger.error(f"Connection test failed for {conn_type}: {e}")
                results[conn_type] = False
        
        return results
    
    def get_available_databases(self, connection_type: str) -> List[str]:
        """Get list of available databases for a connection type."""
        try:
            connector = self.create_connector(connection_type)
            with connector:
                return connector.get_databases()
        except Exception as e:
            logger.error(f"Failed to get databases for {connection_type}: {e}")
            return []
    
    def get_available_tables(self, connection_type: str, database: str) -> List[str]:
        """Get list of available tables in a database."""
        try:
            connector = self.create_connector(connection_type)
            with connector:
                return connector.get_tables(database)
        except Exception as e:
            logger.error(f"Failed to get tables for {connection_type}.{database}: {e}")
            return []
    
    def analyze_with_custom_workflow(self, request: AnalysisRequest, 
                                   workflow_stages: List[str],
                                   parallel_stages: Optional[List[str]] = None) -> AnalysisResult:
        """Analyze with a custom AI agent workflow."""
        start_time = time.time()
        logger.info(f"Starting analysis with custom workflow: {workflow_stages}")
        
        try:
            # Create connector and perform basic analysis
            connector = self.create_connector(request.connection_type)
            
            with connector:
                if not connector.test_connection():
                    raise ConnectionError(f"Failed to connect to {request.connection_type}")
                
                # Initialize result
                result = AnalysisResult(
                    request=request,
                    success=True,
                    execution_time=0.0,
                    metadata={
                        'connection_type': request.connection_type,
                        'database': request.database,
                        'custom_workflow': workflow_stages,
                        'analysis_timestamp': time.time()
                    }
                )
                
                # Perform schema analysis and data profiling
                result.schema_analysis = self._perform_schema_analysis(
                    connector, request.database, request.tables
                )
                
                if request.include_data_profiling:
                    result.data_profiles = self._perform_data_profiling(
                        connector, request.database, request.tables, request.sample_size
                    )
                
                # Execute custom workflow
                if request.include_ai_recommendations:
                    custom_workflow = self.workflow_engine.create_custom_workflow(
                        workflow_stages, parallel_stages
                    )
                    
                    workflow_data = {}
                    if result.schema_analysis:
                        workflow_data.update(result.schema_analysis)
                    if result.data_profiles:
                        workflow_data['data_profiles'] = result.data_profiles
                    
                    workflow_result = self.workflow_engine.execute_workflow(
                        workflow_data, request.context, custom_workflow
                    )
                    
                    result.ai_recommendations = workflow_result.consolidated_recommendations
                    result.workflow_insights = workflow_result.workflow_insights
                    result.agent_results = workflow_result.stage_results
                
                result.execution_time = time.time() - start_time
                logger.info(f"Custom workflow analysis completed in {result.execution_time:.2f} seconds")
                return result
                
        except Exception as e:
            logger.error(f"Custom workflow analysis failed: {e}")
            return AnalysisResult(
                request=request,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={'connection_type': request.connection_type, 'database': request.database}
            )
    
    def analyze_discovery_only(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform discovery-only analysis."""
        return self.analyze_with_custom_workflow(request, ['discovery'])
    
    def analyze_quality_focused(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform quality-focused analysis."""
        return self.analyze_with_custom_workflow(
            request, 
            ['discovery', 'analysis'], 
            parallel_stages=[]
        )
    
    def analyze_modeling_focused(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform modeling-focused analysis."""
        return self.analyze_with_custom_workflow(
            request,
            ['discovery', 'modeling'],
            parallel_stages=['modeling']
        )
    
    def analyze_optimization_focused(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform optimization-focused analysis."""
        return self.analyze_with_custom_workflow(
            request,
            ['discovery', 'analysis', 'optimization'],
            parallel_stages=['optimization']
        )

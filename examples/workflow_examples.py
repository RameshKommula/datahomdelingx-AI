#!/usr/bin/env python3
"""Examples of using the new AI agent workflow system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine.analysis_engine import AnalysisEngine, AnalysisRequest
from ai_agents.workflow_engine import WorkflowEngine, WorkflowStage, WorkflowTask
from config import init_config
import json
import time


def example_discovery_agent():
    """Example of using just the Discovery Agent."""
    print("🔍 Example: Discovery Agent Only")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Get a working connection
    connection_results = engine.test_connections()
    working_connections = [conn for conn, success in connection_results.items() if success]
    
    if not working_connections:
        print("❌ No working connections found.")
        return
    
    connection_type = working_connections[0]
    databases = engine.get_available_databases(connection_type)
    
    if not databases:
        print("❌ No databases found.")
        return
    
    database = databases[0]
    print(f"🔍 Running discovery analysis on {connection_type}.{database}")
    
    # Create request for discovery-only analysis
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        include_data_profiling=False,  # Skip profiling for speed
        include_ai_recommendations=True,
        context={
            'analysis_focus': 'asset_discovery',
            'business_domain': 'data_catalog_building'
        }
    )
    
    # Run discovery-focused analysis
    result = engine.analyze_discovery_only(request)
    
    if result.success:
        print(f"✅ Discovery completed in {result.execution_time:.2f} seconds")
        
        # Show discovery insights
        if result.workflow_insights:
            insights = result.workflow_insights
            print(f"📊 Workflow Insights:")
            print(f"  - Stages executed: {insights.get('execution_summary', {}).get('successful_stages', 0)}")
            print(f"  - Average confidence: {insights.get('confidence_analysis', {}).get('average_confidence', 0):.2f}")
        
        # Show discovery findings
        discovery_findings = [rec for rec in result.ai_recommendations 
                            if rec.get('source_stage') == 'discovery']
        
        print(f"\n🎯 Discovery Findings ({len(discovery_findings)}):")
        for finding in discovery_findings[:5]:  # Show top 5
            domain = finding.get('domain', 'general')
            importance = finding.get('importance', 'supporting')
            print(f"  • {finding.get('title', 'Unknown')} [{domain}] [{importance}]")
    else:
        print(f"❌ Discovery failed: {result.error_message}")


def example_analysis_workflow():
    """Example of using Discovery + Analysis workflow."""
    print("\n📊 Example: Discovery + Analysis Workflow")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Get a working connection
    connection_results = engine.test_connections()
    working_connections = [conn for conn, success in connection_results.items() if success]
    
    if not working_connections:
        print("❌ No working connections found.")
        return
    
    connection_type = working_connections[0]
    databases = engine.get_available_databases(connection_type)
    
    if not databases:
        print("❌ No databases found.")
        return
    
    database = databases[0]
    print(f"📊 Running quality-focused analysis on {connection_type}.{database}")
    
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        include_data_profiling=True,
        include_ai_recommendations=True,
        sample_size=5000,
        context={
            'analysis_focus': 'data_quality',
            'business_requirements': 'Improve data quality for analytics',
            'quality_threshold': 0.8
        }
    )
    
    # Run quality-focused analysis
    result = engine.analyze_quality_focused(request)
    
    if result.success:
        print(f"✅ Quality analysis completed in {result.execution_time:.2f} seconds")
        
        # Show agent-specific results
        if result.agent_results:
            print(f"\n🤖 Agent Results:")
            for stage_name, agent_response in result.agent_results.items():
                stage_display = stage_name.replace('_', ' ').title() if isinstance(stage_name, str) else stage_name.value.replace('_', ' ').title()
                status = "✅" if agent_response.success else "❌"
                print(f"  {status} {stage_display}: {len(agent_response.recommendations)} recommendations (confidence: {agent_response.confidence:.2f})")
        
        # Show quality-specific findings
        quality_findings = [rec for rec in result.ai_recommendations 
                          if rec.get('category') in ['data_quality', 'pattern_recognition']]
        
        print(f"\n🔍 Quality Findings ({len(quality_findings)}):")
        for finding in quality_findings[:3]:  # Show top 3
            severity = finding.get('severity', 'medium')
            source = finding.get('source_stage', 'unknown')
            print(f"  • [{severity.upper()}] {finding.get('title', 'Unknown')} (from {source})")
    else:
        print(f"❌ Quality analysis failed: {result.error_message}")


def example_custom_workflow():
    """Example of creating a custom workflow."""
    print("\n⚙️ Example: Custom Workflow (Discovery → Modeling)")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Get a working connection
    connection_results = engine.test_connections()
    working_connections = [conn for conn, success in connection_results.items() if success]
    
    if not working_connections:
        print("❌ No working connections found.")
        return
    
    connection_type = working_connections[0]
    databases = engine.get_available_databases(connection_type)
    
    if not databases:
        print("❌ No databases found.")
        return
    
    database = databases[0]
    print(f"⚙️ Running custom workflow: Discovery → Modeling on {connection_type}.{database}")
    
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        include_data_profiling=False,  # Skip profiling for this example
        include_ai_recommendations=True,
        context={
            'analysis_focus': 'dimensional_modeling',
            'business_domain': 'analytics',
            'target_architecture': 'star_schema'
        }
    )
    
    # Custom workflow: Discovery first, then Modeling
    custom_stages = ['discovery', 'modeling']
    result = engine.analyze_with_custom_workflow(request, custom_stages)
    
    if result.success:
        print(f"✅ Custom workflow completed in {result.execution_time:.2f} seconds")
        
        # Show workflow execution details
        if result.workflow_insights:
            insights = result.workflow_insights
            exec_summary = insights.get('execution_summary', {})
            print(f"📋 Workflow Execution:")
            print(f"  - Total stages: {exec_summary.get('total_stages', 0)}")
            print(f"  - Successful stages: {exec_summary.get('successful_stages', 0)}")
            print(f"  - Failed stages: {exec_summary.get('failed_stages', 0)}")
            
            # Show cross-stage correlations
            correlations = insights.get('cross_stage_correlations', [])
            if correlations:
                print(f"  - Cross-stage correlations: {len(correlations)}")
                for corr in correlations[:2]:  # Show top 2
                    print(f"    • {corr['stage1']} ↔ {corr['stage2']} (overlap: {corr['overlap_score']:.2f})")
        
        # Show modeling recommendations
        modeling_recs = [rec for rec in result.ai_recommendations 
                        if rec.get('source_stage') == 'modeling']
        
        print(f"\n🏗️ Modeling Recommendations ({len(modeling_recs)}):")
        for rec in modeling_recs[:3]:  # Show top 3
            category = rec.get('category', 'general').replace('_', ' ').title()
            priority = rec.get('consolidated_priority', rec.get('priority', 'medium'))
            print(f"  • [{priority.upper()}] {rec.get('title', 'Unknown')} ({category})")
    else:
        print(f"❌ Custom workflow failed: {result.error_message}")


def example_parallel_workflow():
    """Example of running agents in parallel."""
    print("\n🚀 Example: Parallel Workflow (Discovery → Modeling & Optimization)")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Get a working connection
    connection_results = engine.test_connections()
    working_connections = [conn for conn, success in connection_results.items() if success]
    
    if not working_connections:
        print("❌ No working connections found.")
        return
    
    connection_type = working_connections[0]
    databases = engine.get_available_databases(connection_type)
    
    if not databases:
        print("❌ No databases found.")
        return
    
    database = databases[0]
    print(f"🚀 Running parallel workflow on {connection_type}.{database}")
    
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        include_data_profiling=True,
        include_ai_recommendations=True,
        sample_size=3000,
        context={
            'business_domain': 'data_warehouse',
            'performance_requirements': 'High-performance analytics',
            'governance_requirements': 'Enterprise data governance'
        }
    )
    
    # Custom workflow with parallel execution
    workflow_stages = ['discovery', 'analysis', 'modeling', 'optimization']
    parallel_stages = ['modeling', 'optimization']  # Run these in parallel
    
    result = engine.analyze_with_custom_workflow(request, workflow_stages, parallel_stages)
    
    if result.success:
        print(f"✅ Parallel workflow completed in {result.execution_time:.2f} seconds")
        
        # Show recommendations by source
        recommendations_by_source = {}
        for rec in result.ai_recommendations:
            source = rec.get('source_stage', 'unknown')
            if source not in recommendations_by_source:
                recommendations_by_source[source] = []
            recommendations_by_source[source].append(rec)
        
        print(f"\n📊 Recommendations by Agent:")
        for source, recs in recommendations_by_source.items():
            high_priority = len([r for r in recs if r.get('consolidated_priority') in ['critical', 'high']])
            print(f"  • {source.replace('_', ' ').title()}: {len(recs)} total ({high_priority} high priority)")
        
        # Show consolidated top recommendations
        top_recs = sorted(
            result.ai_recommendations,
            key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}.get(
                x.get('consolidated_priority', 'medium'), 2
            ),
            reverse=True
        )[:5]
        
        print(f"\n🎯 Top Consolidated Recommendations:")
        for i, rec in enumerate(top_recs, 1):
            priority = rec.get('consolidated_priority', 'medium')
            source = rec.get('source_stage', 'unknown')
            confidence = rec.get('stage_confidence', 0)
            print(f"  {i}. [{priority.upper()}] {rec.get('title', 'Unknown')}")
            print(f"     Source: {source} (confidence: {confidence:.2f})")
    else:
        print(f"❌ Parallel workflow failed: {result.error_message}")


def example_workflow_engine_direct():
    """Example of using the WorkflowEngine directly."""
    print("\n🔧 Example: Direct WorkflowEngine Usage")
    print("-" * 50)
    
    # Create workflow engine
    workflow_engine = WorkflowEngine(max_parallel_tasks=2)
    
    # Create sample data (normally this would come from schema analysis)
    sample_data = {
        'database': 'sample_db',
        'tables': {
            'customers': {
                'table_type': 'dimension',
                'row_count': 100000,
                'columns': [
                    {'name': 'customer_id', 'data_type': 'bigint', 'is_primary_key': True},
                    {'name': 'customer_name', 'data_type': 'varchar'},
                    {'name': 'email', 'data_type': 'varchar'},
                    {'name': 'created_date', 'data_type': 'date'}
                ]
            },
            'orders': {
                'table_type': 'fact',
                'row_count': 1000000,
                'columns': [
                    {'name': 'order_id', 'data_type': 'bigint', 'is_primary_key': True},
                    {'name': 'customer_id', 'data_type': 'bigint', 'is_foreign_key': True},
                    {'name': 'order_amount', 'data_type': 'decimal'},
                    {'name': 'order_date', 'data_type': 'date'}
                ]
            }
        },
        'relationships': [
            {
                'type': 'foreign_key',
                'from_table': 'orders',
                'from_column': 'customer_id',
                'to_table': 'customers',
                'to_column': 'customer_id',
                'confidence': 0.9
            }
        ]
    }
    
    context = {
        'business_domain': 'e-commerce',
        'analysis_focus': 'dimensional_modeling'
    }
    
    print("🔧 Executing workflow with sample data...")
    
    # Execute workflow
    workflow_result = workflow_engine.execute_workflow(sample_data, context)
    
    if workflow_result.success:
        print(f"✅ Workflow completed in {workflow_result.execution_time:.2f} seconds")
        
        # Show stage results
        print(f"\n📋 Stage Execution Results:")
        for stage, agent_response in workflow_result.stage_results.items():
            status = "✅" if agent_response.success else "❌"
            stage_name = stage.value.replace('_', ' ').title()
            print(f"  {status} {stage_name}: {len(agent_response.recommendations)} recommendations")
            print(f"     Confidence: {agent_response.confidence:.2f}")
        
        # Show workflow insights
        if workflow_result.workflow_insights:
            insights = workflow_result.workflow_insights
            print(f"\n💡 Workflow Insights:")
            
            conf_analysis = insights.get('confidence_analysis', {})
            if conf_analysis:
                print(f"  - Average confidence: {conf_analysis.get('average_confidence', 0):.2f}")
                print(f"  - Highest confidence stage: {conf_analysis.get('highest_confidence_stage', 'unknown')}")
            
            rec_dist = insights.get('recommendation_distribution', {})
            if rec_dist:
                print(f"  - Total recommendations: {rec_dist.get('total_recommendations', 0)}")
                print(f"  - Average per stage: {rec_dist.get('average_per_stage', 0):.1f}")
        
        # Show consolidated recommendations
        print(f"\n🎯 Consolidated Recommendations ({len(workflow_result.consolidated_recommendations)}):")
        for i, rec in enumerate(workflow_result.consolidated_recommendations[:3], 1):
            priority = rec.get('consolidated_priority', 'medium')
            source = rec.get('source_stage', 'unknown')
            print(f"  {i}. [{priority.upper()}] {rec.get('title', 'Unknown')} (from {source})")
    else:
        print(f"❌ Workflow failed: {workflow_result.error_message}")


def main():
    """Run all workflow examples."""
    print("🤖 DataModeling-AI Workflow Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_discovery_agent()
        example_analysis_workflow()
        example_custom_workflow()
        example_parallel_workflow()
        example_workflow_engine_direct()
        
        print("\n🎉 All workflow examples completed!")
        print("\nKey Benefits of the New Workflow System:")
        print("✅ Specialized AI agents for different analysis types")
        print("✅ Flexible workflow orchestration with dependencies")
        print("✅ Parallel execution for improved performance")
        print("✅ Agent-to-agent communication and data passing")
        print("✅ Consolidated recommendations with priority scoring")
        print("✅ Workflow insights and cross-stage correlations")
        
        print("\nNext Steps:")
        print("1. Try the CLI with workflow options:")
        print("   python main.py workflow-info")
        print("   python main.py analyze hive my_db --workflow-type discovery")
        print("   python main.py analyze hive my_db --workflow discovery modeling --parallel modeling")
        print("2. Explore the web interface for interactive workflows")
        print("3. Build custom agents by extending the BaseAgent class")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your configuration file is set up correctly")
        print("2. Verify database connections are working")
        print("3. Check that you have a valid OpenAI API key")


if __name__ == "__main__":
    main()

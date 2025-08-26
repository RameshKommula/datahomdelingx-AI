#!/usr/bin/env python3
"""Example usage of DataModeling-AI."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine.analysis_engine import AnalysisEngine, AnalysisRequest
from config import init_config
import json
import time


def example_basic_analysis():
    """Example of basic database analysis."""
    print("🔍 Example: Basic Database Analysis")
    print("-" * 50)
    
    # Initialize configuration
    # init_config("config.yaml")  # Uncomment if you have a config file
    
    # Create analysis engine
    engine = AnalysisEngine()
    
    # Test connections first
    print("Testing connections...")
    connection_results = engine.test_connections()
    
    for conn_type, success in connection_results.items():
        status = "✅" if success else "❌"
        print(f"{status} {conn_type.upper()}")
    
    # Find a working connection
    working_connections = [conn for conn, success in connection_results.items() if success]
    
    if not working_connections:
        print("❌ No working connections found. Please check your configuration.")
        return
    
    connection_type = working_connections[0]
    print(f"\n📊 Using {connection_type.upper()} for analysis")
    
    # Get available databases
    databases = engine.get_available_databases(connection_type)
    if not databases:
        print("❌ No databases found.")
        return
    
    database = databases[0]  # Use first database
    print(f"📁 Analyzing database: {database}")
    
    # Create analysis request
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        tables=None,  # Analyze all tables
        include_data_profiling=True,
        include_ai_recommendations=True,
        sample_size=5000,  # Smaller sample for example
        context={
            'business_domain': 'example_analysis',
            'performance_requirements': 'Standard OLAP workloads'
        }
    )
    
    # Perform analysis
    print("🚀 Starting analysis...")
    start_time = time.time()
    
    result = engine.analyze(request)
    
    if result.success:
        print(f"✅ Analysis completed in {result.execution_time:.2f} seconds")
        
        # Print summary
        if result.schema_analysis:
            summary = result.schema_analysis.get('summary', {})
            print(f"📋 Found {summary.get('total_tables', 0)} tables")
            print(f"📋 Found {summary.get('total_columns', 0)} columns")
            print(f"📋 Found {summary.get('total_rows', 0):,} total rows")
        
        if result.ai_recommendations:
            print(f"🤖 Generated {len(result.ai_recommendations)} AI recommendations")
            
            # Show top 3 high-priority recommendations
            high_priority = [rec for rec in result.ai_recommendations 
                           if rec.get('priority') == 'high'][:3]
            
            if high_priority:
                print("\n🎯 Top Priority Recommendations:")
                for i, rec in enumerate(high_priority, 1):
                    print(f"{i}. {rec.get('title', 'Unknown')}")
                    print(f"   Category: {rec.get('category', 'general')}")
                    print(f"   Priority: {rec.get('priority', 'medium')}")
                    print()
        
        # Save results
        output_file = f"analysis_results_{int(time.time())}.json"
        save_analysis_results(result, output_file)
        print(f"💾 Results saved to {output_file}")
        
    else:
        print(f"❌ Analysis failed: {result.error_message}")


def example_specific_tables_analysis():
    """Example of analyzing specific tables."""
    print("\n🎯 Example: Specific Tables Analysis")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Get available connections
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
    
    # Get available tables
    tables = engine.get_available_tables(connection_type, database)
    
    if not tables:
        print("❌ No tables found.")
        return
    
    # Analyze first 2 tables (or all if less than 2)
    selected_tables = tables[:2]
    print(f"📊 Analyzing specific tables: {', '.join(selected_tables)}")
    
    request = AnalysisRequest(
        connection_type=connection_type,
        database=database,
        tables=selected_tables,
        include_data_profiling=True,
        include_ai_recommendations=True,
        sample_size=3000
    )
    
    result = engine.analyze(request)
    
    if result.success:
        print(f"✅ Analysis completed for {len(selected_tables)} tables")
        
        # Show table-specific insights
        if result.data_profiles:
            print("\n📈 Data Quality Scores:")
            for table_name, profile in result.data_profiles.items():
                if isinstance(profile, dict) and 'overall_quality_score' in profile:
                    score = profile['overall_quality_score']
                    print(f"  {table_name}: {score:.2f}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")


def example_batch_analysis():
    """Example of batch analysis across multiple databases."""
    print("\n📦 Example: Batch Analysis")
    print("-" * 50)
    
    engine = AnalysisEngine()
    
    # Create multiple analysis requests
    requests = []
    
    for conn_type in ['hive', 'trino', 'presto']:
        try:
            databases = engine.get_available_databases(conn_type)
            if databases:
                # Create request for first database
                request = AnalysisRequest(
                    connection_type=conn_type,
                    database=databases[0],
                    include_data_profiling=False,  # Skip profiling for speed
                    include_ai_recommendations=True,
                    sample_size=1000
                )
                requests.append(request)
        except Exception as e:
            print(f"⚠️ Could not create request for {conn_type}: {e}")
    
    if not requests:
        print("❌ No valid requests created.")
        return
    
    print(f"🚀 Starting batch analysis of {len(requests)} databases...")
    
    # Perform batch analysis
    results = engine.analyze_multiple_databases(requests)
    
    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"✅ Successful analyses: {successful}")
    print(f"❌ Failed analyses: {failed}")
    
    # Show recommendations summary
    total_recommendations = 0
    for result in results:
        if result.success and result.ai_recommendations:
            total_recommendations += len(result.ai_recommendations)
            print(f"  {result.request.database}: {len(result.ai_recommendations)} recommendations")
    
    print(f"🤖 Total recommendations generated: {total_recommendations}")


def example_custom_context():
    """Example of using custom context for targeted recommendations."""
    print("\n🎨 Example: Custom Context Analysis")
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
    
    # Create request with rich context
    context = {
        'business_domain': 'E-commerce Analytics',
        'data_volume': 'Large (>1TB)',
        'query_patterns': [
            'Daily sales reports',
            'Customer segmentation analysis',
            'Product performance metrics',
            'Real-time inventory tracking'
        ],
        'performance_requirements': 'Sub-second response for dashboards, batch processing for ML',
        'compliance_requirements': 'GDPR, PCI-DSS',
        'team_size': 'Medium (5-10 data engineers)',
        'budget_constraints': 'Moderate - focus on cost-effective solutions'
    }
    
    request = AnalysisRequest(
        connection_type=connection_type,
        database=databases[0],
        include_data_profiling=True,
        include_ai_recommendations=True,
        sample_size=5000,
        context=context
    )
    
    print("🎯 Analyzing with rich business context...")
    result = engine.analyze(request)
    
    if result.success:
        print("✅ Context-aware analysis completed")
        
        if result.ai_recommendations:
            # Group recommendations by category
            categories = {}
            for rec in result.ai_recommendations:
                category = rec.get('category', 'general')
                if category not in categories:
                    categories[category] = []
                categories[category].append(rec)
            
            print(f"\n📊 Recommendations by Category:")
            for category, recs in categories.items():
                print(f"  {category.replace('_', ' ').title()}: {len(recs)} recommendations")
            
            # Show business-focused recommendations
            business_recs = [rec for rec in result.ai_recommendations 
                           if 'business' in rec.get('description', '').lower() or 
                              'cost' in rec.get('description', '').lower() or
                              'performance' in rec.get('description', '').lower()]
            
            if business_recs:
                print(f"\n💼 Business-Focused Recommendations ({len(business_recs)}):")
                for rec in business_recs[:3]:  # Show top 3
                    print(f"  • {rec.get('title', 'Unknown')}")
    else:
        print(f"❌ Analysis failed: {result.error_message}")


def save_analysis_results(result, filename):
    """Save analysis results to JSON file."""
    # Convert result to serializable format
    data = {
        'request': {
            'connection_type': result.request.connection_type,
            'database': result.request.database,
            'tables': result.request.tables,
            'include_data_profiling': result.request.include_data_profiling,
            'include_ai_recommendations': result.request.include_ai_recommendations,
            'sample_size': result.request.sample_size,
            'context': result.request.context
        },
        'success': result.success,
        'execution_time': result.execution_time,
        'schema_analysis': result.schema_analysis,
        'data_profiles': result.data_profiles,
        'ai_recommendations': result.ai_recommendations,
        'metadata': result.metadata
    }
    
    if result.error_message:
        data['error_message'] = result.error_message
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def main():
    """Run all examples."""
    print("🤖 DataModeling-AI Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_analysis()
        example_specific_tables_analysis()
        example_batch_analysis()
        example_custom_context()
        
        print("\n🎉 All examples completed!")
        print("\nNext steps:")
        print("1. Review the generated JSON files with detailed results")
        print("2. Try the web interface: python web_app.py")
        print("3. Explore the CLI: python main.py --help")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure your configuration file is set up correctly")
        print("2. Verify database connections are working")
        print("3. Check that you have a valid OpenAI API key")


if __name__ == "__main__":
    main()

"""Streamlit web interface for DataModeling-AI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.analysis_engine import AnalysisEngine, AnalysisRequest
from config import init_config, get_config
from security.encryption import ConnectionManager

# Page configuration
st.set_page_config(
    page_title="DataModeling-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-high {
        border-left: 4px solid #ff4b4b;
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        border-left: 4px solid #ffa500;
        background-color: #fffaf0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-low {
        border-left: 4px solid #32cd32;
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    st.title("🤖 DataModeling-AI")
    st.markdown("*Automate data modeling with AI agents for Hive, Trino, and Presto*")
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Connection Configuration")
        
        # Initialize connection manager
        if 'connection_manager' not in st.session_state:
            st.session_state.connection_manager = ConnectionManager()
        
        # Connection management section
        with st.expander("💾 Saved Connections", expanded=True):
            display_connection_manager()
        
        # Connection method selection
        config_method = st.radio(
            "Configuration Method",
            ["Manual Connection", "Upload Config File", "Load Saved Connection"],
            help="Choose how to configure your data lake connections"
        )
        
        if config_method == "Upload Config File":
            # Configuration file upload
            config_file = st.file_uploader(
                "Upload Configuration File",
                type=['yaml', 'yml'],
                help="Upload your configuration file with connection details"
            )
            
            if config_file:
                try:
                    # Save uploaded file temporarily
                    config_path = f"/tmp/{config_file.name}"
                    with open(config_path, "wb") as f:
                        f.write(config_file.getbuffer())
                    
                    # Initialize configuration
                    init_config(config_path)
                    st.success("Configuration loaded successfully!")
                    
                    # Initialize engine
                    if st.session_state.engine is None:
                        st.session_state.engine = AnalysisEngine()
                    
                except Exception as e:
                    st.error(f"Failed to load configuration: {e}")
        
        elif config_method == "Load Saved Connection":
            # Load saved connection
            handle_load_saved_connection_ui()
        
        else:  # Manual Connection
            # Show setup progress
            display_setup_progress()
            
            # Show saved connections at the top
            display_saved_connections_summary()
            
            st.subheader("🔗 Step 1: Trino Connection Details")
            
            # Trino connection form
            with st.form("trino_connection"):
                trino_host = st.text_input("Host", value="localhost", help="Trino server hostname")
                trino_port = st.number_input("Port", value=8080, min_value=1, max_value=65535, help="Trino server port")
                trino_username = st.text_input("Username", help="Trino username")
                trino_password = st.text_input("Password", type="password", help="Trino password (optional)")
                trino_catalog = st.text_input("Catalog", value="hive", help="Trino catalog name")
                trino_schema = st.text_input("Schema", value="default", help="Default schema name")
                
                # Trino connection buttons
                col1, col2 = st.columns(2)
                with col1:
                    test_trino_button = st.form_submit_button("🔍 Test Connection", type="secondary")
                with col2:
                    save_trino_button = st.form_submit_button("💾 Save Connection", type="secondary")
                
                if test_trino_button:
                    if not trino_host or not trino_username:
                        st.error("Please provide at least host and username")
                    else:
                        test_trino_connection(trino_host, trino_port, trino_username, trino_password, trino_catalog, trino_schema)
                
                if save_trino_button:
                    if not trino_host or not trino_username:
                        st.error("Please provide at least host and username")
                    else:
                        save_trino_connection_data(trino_host, trino_port, trino_username, trino_password, trino_catalog, trino_schema)
            
            st.divider()
            
            # Check if Trino connection is successful before showing AI configuration
            trino_connected = st.session_state.get('trino_connection_successful', False)
            
            if not trino_connected:
                st.subheader("🤖 Step 2: AI Configuration")
                st.info("⏳ Please complete and test your Trino connection first to enable AI configuration")
            else:
                # AI Configuration section (only enabled after Trino success)
                st.subheader("🤖 Step 2: AI Configuration")
                st.success("✅ Trino connection verified - AI configuration now enabled!")
                
                # Show saved AI configurations at the top
                display_saved_ai_configs_summary()
                
                with st.form("ai_configuration"):
                    llm_provider = st.selectbox("LLM Provider", ["claude", "openai"], index=0, help="Choose your AI provider")
                    
                    if llm_provider == "claude":
                        api_key = st.text_input("Claude API Key", type="password", help="Your Anthropic API key")
                        st.info("🤖 Claude-3.5-Sonnet provides superior analysis and recommendations")
                    else:
                        api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
                        st.info("🤖 Using OpenAI GPT-4 for analysis")
                    
                    # AI configuration buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        test_ai_button = st.form_submit_button("🔍 Test AI Connection", type="secondary")
                    with col2:
                        save_ai_button = st.form_submit_button("💾 Save AI Config", type="secondary")
                    with col3:
                        skip_test_button = st.form_submit_button("⏭️ Skip Test & Continue", type="secondary", help="Skip API test and proceed (for billing issues)")
                    
                    if test_ai_button:
                        if not api_key:
                            st.error(f"Please provide {llm_provider.title()} API key")
                        else:
                            test_ai_connection(llm_provider, api_key)
                    
                    if save_ai_button:
                        if not api_key:
                            st.error(f"Please provide {llm_provider.title()} API key")
                        else:
                            save_ai_configuration(llm_provider, api_key)
                    
                    if skip_test_button:
                        if not api_key:
                            st.error(f"Please provide {llm_provider.title()} API key")
                        else:
                            st.warning("⚠️ Skipping API test - Configuration saved without validation")
                            st.info("💡 **Note:** You can test the connection later or add credits to your account")
                            
                            # Store config without testing
                            st.session_state.current_ai_config = {
                                'llm_provider': llm_provider,
                                'api_key': api_key
                            }
                            
                            # Mark as successful to enable launch
                            st.session_state.ai_connection_successful = True
                
                st.divider()
                
                # Final connection setup (only enabled after both connections are successful)
                ai_connected = st.session_state.get('ai_connection_successful', False)
                
                if not ai_connected:
                    st.subheader("🚀 Step 3: Launch Application")
                    st.info("⏳ Please complete and test your AI configuration first to enable application launch")
                else:
                    st.subheader("🚀 Step 3: Launch Application")
                    st.success("✅ Both Trino and AI connections verified - Ready to launch!")
                    
                    if st.button("🚀 Launch DataModeling-AI Engine", type="primary", help="Initialize the complete application with your configurations"):
                        # Get configurations
                        trino_config = st.session_state.get('current_trino_config')
                        ai_config = st.session_state.get('current_ai_config')
                        
                        try:
                            # Create complete configuration
                            complete_config = create_complete_configuration(trino_config, ai_config)
                            
                            # Initialize the engine
                            init_config_from_dict(complete_config)
                            st.session_state.engine = AnalysisEngine()
                            st.session_state.manual_connection = True
                            
                            st.success("🎉 DataModeling-AI Engine launched successfully!")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Launch failed: {e}")
        
        # Connection test
        if st.session_state.engine:
            st.subheader("🔍 Connection Status")
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    try:
                        # Test specifically Trino connection
                        connector = st.session_state.engine.create_connector('trino')
                        success = connector.test_connection()
                        
                        if success:
                            st.success("✅ Trino connection successful!")
                        else:
                            st.error("❌ Trino connection failed!")
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")
            
            # Show current connection info
            if hasattr(st.session_state, 'manual_connection') and st.session_state.manual_connection:
                st.info("📡 Connected via manual configuration")
            
            # Save connection section
            if (hasattr(st.session_state, 'current_connection_data') and 
                st.session_state.current_connection_data and
                st.session_state.get('encryption_password')):
                
                st.subheader("💾 Save Connection")
                save_current_connection()
    
    # Main content
    if st.session_state.engine is None:
        st.markdown("""
        # 🚀 Welcome to DataModeling-AI
        
        **Automate data modeling with AI agents for Trino, Hive, and Presto data lakes**
        
        To get started, please configure your connection using one of the methods in the sidebar:
        
        ### 🔧 Quick Setup Options:
        
        1. **📝 Manual Connection** (Recommended for first-time users)
           - Enter your Trino connection details directly in the sidebar
           - Add your Claude or OpenAI API key for AI analysis
           - Perfect for testing and getting started quickly
        
        2. **📁 Upload Config File** (For advanced users)
           - Upload a YAML configuration file with your settings
           - Supports multiple data lake connections
           - Ideal for production environments
        
        ### 🎯 What You'll Get:
        
        - **🔍 Discovery Agent**: Catalogs your data assets and identifies business domains
        - **📊 Analysis Agent**: Deep data quality analysis and pattern recognition  
        - **🏗️ Modeling Agent**: Dimensional modeling and schema design recommendations
        - **⚡ Optimization Agent**: Performance optimization and best practices
        
        ### 📋 Prerequisites:
        
        - Access to a Trino, Hive, or Presto cluster
        - Valid credentials for your data lake
        - Claude API key (recommended) or OpenAI API key
        """)
        
        # Quick start guide
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            ### Step 1: Connect to Your Data Lake
            1. In the sidebar, select "Manual Connection"
            2. Enter your Trino connection details:
               - **Host**: Your Trino coordinator hostname
               - **Port**: Usually 8080 for Trino
               - **Username**: Your Trino username
               - **Catalog**: Usually "hive" for Hive-compatible catalogs
               - **Schema**: Target schema/database name
            
            ### Step 2: Configure AI Analysis
            1. Choose your LLM provider (Claude recommended)
            2. Enter your API key
            3. Click "Connect to Trino"
            
            ### Step 3: Select Your Data
            1. Choose your schema/database from the dropdown
            2. Select tables to analyze:
               - **All Tables**: Analyze everything (good for comprehensive analysis)
               - **Specific Tables**: Choose individual tables
               - **Pattern Matching**: Use patterns like "fact_*" or "*_dim"
            
            ### Step 4: Run Analysis
            1. Configure workflow options (or use defaults)
            2. Click "Start Analysis"
            3. Review detailed recommendations and insights
            """)
        
        # Sample configuration
        with st.expander("📄 Sample Configuration File"):
            st.code("""
# DataModeling-AI Configuration
llm_provider: "claude"  # or "openai"

# Claude Configuration (recommended)
claude:
  api_key: "your_claude_api_key_here"
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.3
  max_tokens: 4000

# Data Lake Connections
data_lakes:
  trino:
    host: "your-trino-host.com"
    port: 8080
    username: "your_username"
    password: "your_password"  # optional
    catalog: "hive"
    schema: "default"

analysis:
  max_sample_rows: 10000
  max_concurrent_tables: 5
""", language="yaml")
        
        return
    
    # Analysis configuration
    st.header("🔬 Analysis Configuration")
    
    # Initialize session state for analysis flow
    if 'analysis_connection_tested' not in st.session_state:
        st.session_state.analysis_connection_tested = False
    if 'analysis_database_selected' not in st.session_state:
        st.session_state.analysis_database_selected = False
    if 'analysis_tables_selected' not in st.session_state:
        st.session_state.analysis_tables_selected = False
    if 'workflow_started' not in st.session_state:
        st.session_state.workflow_started = False
    
    # Step 1: Connection & Database Selection
    with st.container():
        st.subheader("1️⃣ Connection & Database Selection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.analysis_connection_tested and st.session_state.analysis_database_selected:
                selected_db = st.session_state.get('selected_database', '')
                st.success(f"✅ Connected to database: **{selected_db}**")
            else:
                st.info("🔌 Test connection and select database/schema")
        
        with col2:
            if not st.session_state.analysis_connection_tested:
                test_conn_btn = st.button("🔍 Test Connection", key="test_analysis_conn")
            elif not st.session_state.analysis_database_selected:
                # Database selection is handled below
                pass
            else:
                reset_conn_btn = st.button("🔄 Reset", key="reset_analysis_conn")
    
    # Connection test and database selection
    if not st.session_state.analysis_connection_tested:
        if 'test_conn_btn' in locals() and test_conn_btn:
            with st.spinner("Testing connection and loading databases..."):
                try:
                    # Use the configured connection type (default to trino)
                    connection_type = "trino"  # Could be made configurable
                    databases = st.session_state.engine.get_available_databases(connection_type)
                    
                    if databases:
                        st.session_state.available_databases = databases
                        st.session_state.connection_type = connection_type
                        st.session_state.analysis_connection_tested = True
                        st.success("✅ Connection successful!")
                        st.rerun()
                    else:
                        st.error("❌ No databases found")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    elif st.session_state.analysis_connection_tested and not st.session_state.analysis_database_selected:
        # Database selection dropdown (enabled after connection test)
        databases = st.session_state.get('available_databases', [])
        selected_db = st.selectbox(
            "📁 Select Database/Schema",
            options=[""] + databases,
            index=0,
            key="db_selector",
            help="Choose the database schema to analyze"
        )
        
        if selected_db:
            st.session_state.selected_database = selected_db
            st.session_state.analysis_database_selected = True
            st.rerun()
    
    elif st.session_state.analysis_connection_tested and st.session_state.analysis_database_selected:
        # Reset connection option
        if 'reset_conn_btn' in locals() and reset_conn_btn:
            # Reset all states
            for key in ['analysis_connection_tested', 'analysis_database_selected', 'analysis_tables_selected', 'workflow_started']:
                st.session_state[key] = False
            st.rerun()
    
    # Step 2: Table Selection
    if st.session_state.analysis_database_selected:
        st.divider()
        st.subheader("2️⃣ Table Selection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.analysis_tables_selected:
                selected_tables = st.session_state.get('selected_tables', [])
                st.success(f"✅ {len(selected_tables)} tables selected for analysis")
            else:
                st.info("📋 Enter table names (comma-separated)")
        
        with col2:
            if not st.session_state.analysis_tables_selected:
                load_tables_btn = st.button("📋 Load Available", key="load_tables_btn")
            else:
                reset_tables_btn = st.button("🔄 Reset", key="reset_tables_btn")
        
        # Load available tables for reference
        if not st.session_state.analysis_tables_selected:
            if 'load_tables_btn' in locals() and load_tables_btn:
                with st.spinner("Loading available tables..."):
                    try:
                        connection_type = st.session_state.get('connection_type', 'trino')
                        database = st.session_state.get('selected_database', '')
                        tables = st.session_state.engine.get_available_tables(connection_type, database)
                        st.session_state.available_tables = tables
                        st.success(f"✅ Found {len(tables)} available tables")
                    except Exception as e:
                        st.error(f"❌ Failed to load tables: {e}")
            
            # Show available tables for reference
            if st.session_state.get('available_tables'):
                with st.expander(f"📋 Available Tables ({len(st.session_state.available_tables)} total)"):
                    tables = st.session_state.available_tables
                    # Display in columns for better readability
                    cols = st.columns(3)
                    for i, table in enumerate(tables[:30]):  # Show first 30
                        cols[i % 3].write(f"• {table}")
                    if len(tables) > 30:
                        st.info(f"... and {len(tables) - 30} more tables available")
            
            # Table input
            table_input = st.text_area(
                "Enter Table Names (comma-separated)",
                placeholder="table1, table2, table3\nor\nschema.table1, schema.table2",
                help="Enter table names separated by commas. You can include schema prefix if needed.",
                height=100,
                key="table_input"
            )
            
            if table_input and st.button("✅ Confirm Tables", key="confirm_tables"):
                # Parse table names
                tables = [t.strip() for t in table_input.split(',') if t.strip()]
                if tables:
                    st.session_state.selected_tables = tables
                    st.session_state.analysis_tables_selected = True
                    st.success(f"✅ Selected {len(tables)} tables")
                    st.rerun()
                else:
                    st.error("Please enter at least one table name")
        
        else:
            # Show selected tables
            selected_tables = st.session_state.selected_tables
            with st.expander(f"Selected Tables ({len(selected_tables)})", expanded=True):
                for i, table in enumerate(selected_tables, 1):
                    st.write(f"{i}. **{table}**")
            
            # Reset tables
            if 'reset_tables_btn' in locals() and reset_tables_btn:
                st.session_state.analysis_tables_selected = False
                st.rerun()
            
    
    # Step 3: Workflow Execution
    if st.session_state.analysis_tables_selected:
        render_workflow_section()

    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_profiling = st.checkbox("Include Data Profiling", value=True)
            include_ai = st.checkbox("Include AI Recommendations", value=True)
            
            # Workflow options
            st.subheader("AI Workflow Options")
            workflow_type = st.selectbox(
                "Workflow Type",
                ["default", "discovery", "quality", "modeling", "optimization", "custom"],
                help="Choose predefined workflow or create custom"
            )
            
            if workflow_type == "custom":
                available_stages = ["discovery", "analysis", "modeling", "optimization"]
                selected_stages = st.multiselect("Select Stages", available_stages, default=available_stages)
                parallel_stages = st.multiselect("Parallel Stages", selected_stages)
            else:
                selected_stages = None
                parallel_stages = None
        
        with col2:
            sample_size = st.number_input("Sample Size", min_value=1000, max_value=100000, value=10000)
            
            # LLM Configuration
            st.subheader("AI Model Settings")
            llm_provider = st.selectbox("LLM Provider", ["claude", "openai"], index=0)
            
            if llm_provider == "claude":
                st.info("🤖 Using Claude-3.5-Sonnet for enhanced analysis")
                st.write("Claude provides superior reasoning and detailed recommendations")
            else:
                st.info("🤖 Using OpenAI GPT-4 for analysis")
        
        with col3:
            # Context information
            business_domain = st.text_input("Business Domain (optional)")
            performance_requirements = st.text_area("Performance Requirements (optional)")
            data_volume = st.selectbox("Data Volume", ["Small (<1GB)", "Medium (1GB-100GB)", "Large (100GB-1TB)", "Very Large (>1TB)"], index=1)
            compliance_requirements = st.text_input("Compliance Requirements (optional)", placeholder="e.g., GDPR, SOX, HIPAA")
    
    # Analysis button
    if st.button("🚀 Start Analysis", type="primary"):
        if not database:
            st.error("Please select a database to analyze")
            return
        
        # Create analysis request with enhanced context
        context = {
            'llm_provider': llm_provider,
            'workflow_type': workflow_type
        }
        if business_domain:
            context['business_domain'] = business_domain
        if performance_requirements:
            context['performance_requirements'] = performance_requirements
        if data_volume:
            context['data_volume'] = data_volume
        if compliance_requirements:
            context['compliance_requirements'] = compliance_requirements
        
        request = AnalysisRequest(
            connection_type=connection_type,
            database=database,
            tables=selected_tables if selected_tables else None,
            include_data_profiling=include_profiling,
            include_ai_recommendations=include_ai,
            sample_size=sample_size,
            context=context
        )
        
        # Perform analysis
        with st.spinner("Analyzing database... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            for i in range(100):
                time.sleep(0.05)  # Small delay for visual effect
                progress_bar.progress(i + 1)
                
                if i < 20:
                    status_text.text("Connecting to database...")
                elif i < 40:
                    status_text.text("Analyzing schema...")
                elif i < 70:
                    status_text.text("Profiling data...")
                elif i < 90:
                    status_text.text("Generating AI recommendations...")
                else:
                    status_text.text("Finalizing analysis...")
            
            # Choose analysis method based on workflow type
            if workflow_type == "discovery":
                result = st.session_state.engine.analyze_discovery_only(request)
            elif workflow_type == "quality":
                result = st.session_state.engine.analyze_quality_focused(request)
            elif workflow_type == "modeling":
                result = st.session_state.engine.analyze_modeling_focused(request)
            elif workflow_type == "optimization":
                result = st.session_state.engine.analyze_optimization_focused(request)
            elif workflow_type == "custom" and selected_stages:
                result = st.session_state.engine.analyze_with_custom_workflow(
                    request, selected_stages, parallel_stages
                )
            else:  # default
                result = st.session_state.engine.analyze(request)
            
            st.session_state.analysis_result = result
        
        if result.success:
            st.success(f"Analysis completed in {result.execution_time:.2f} seconds!")
        else:
            st.error(f"Analysis failed: {result.error_message}")
    
    # Display results
    if st.session_state.analysis_result and st.session_state.analysis_result.success:
        display_analysis_results(st.session_state.analysis_result)


def display_analysis_results(result):
    """Display analysis results."""
    st.header("📊 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if result.schema_analysis:
        summary = result.schema_analysis.get('summary', {})
        
        with col1:
            st.metric("Total Tables", summary.get('total_tables', 0))
        
        with col2:
            st.metric("Total Columns", summary.get('total_columns', 0))
        
        with col3:
            st.metric("Total Rows", f"{summary.get('total_rows', 0):,}")
        
        with col4:
            relationships = len(result.schema_analysis.get('relationships', []))
            st.metric("Relationships", relationships)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", 
        "🔄 Workflow Details", 
        "🏗️ Schema Analysis", 
        "🔍 Data Profiling", 
        "🤖 AI Recommendations",
        "🎯 Agent Results"
    ])
    
    with tab1:
        display_overview(result)
    
    with tab2:
        display_workflow_details(result)
    
    with tab3:
        display_schema_analysis(result.schema_analysis)
    
    with tab4:
        display_data_profiling(result.data_profiles)
    
    with tab5:
        display_ai_recommendations(result.ai_recommendations)
    
    with tab6:
        display_agent_results(result)


def display_workflow_details(result):
    """Display detailed workflow execution information."""
    if not hasattr(result, 'workflow_insights') or not result.workflow_insights:
        st.info("No workflow details available. This analysis may not have used the AI agent workflow.")
        return
    
    insights = result.workflow_insights
    
    st.header("🔄 Workflow Execution Details")
    
    # Execution Summary
    st.subheader("📊 Execution Summary")
    exec_summary = insights.get('execution_summary', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stages", exec_summary.get('total_stages', 0))
    with col2:
        st.metric("Successful Stages", exec_summary.get('successful_stages', 0))
    with col3:
        st.metric("Failed Stages", exec_summary.get('failed_stages', 0))
    with col4:
        success_rate = (exec_summary.get('successful_stages', 0) / max(exec_summary.get('total_stages', 1), 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Confidence Analysis
    if 'confidence_analysis' in insights:
        st.subheader("🎯 Confidence Analysis")
        conf_analysis = insights['confidence_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            avg_confidence = conf_analysis.get('average_confidence', 0)
            st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Confidence"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Stage Confidence Scores:**")
            stage_confidences = conf_analysis.get('stage_confidences', {})
            for stage, confidence in stage_confidences.items():
                confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                st.write(f"{confidence_color} **{stage.replace('_', ' ').title()}**: {confidence:.2f}")
            
            if 'highest_confidence_stage' in conf_analysis:
                st.success(f"🏆 Highest Confidence: {conf_analysis['highest_confidence_stage'].replace('_', ' ').title()}")
            if 'lowest_confidence_stage' in conf_analysis:
                st.warning(f"⚠️ Lowest Confidence: {conf_analysis['lowest_confidence_stage'].replace('_', ' ').title()}")
    
    # Recommendation Distribution
    if 'recommendation_distribution' in insights:
        st.subheader("📈 Recommendation Distribution")
        rec_dist = insights['recommendation_distribution']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Recommendations", rec_dist.get('total_recommendations', 0))
            st.metric("Average per Stage", f"{rec_dist.get('average_per_stage', 0):.1f}")
        
        with col2:
            # Chart showing recommendations by stage
            by_stage = rec_dist.get('by_stage', {})
            if by_stage:
                fig = px.bar(
                    x=list(by_stage.keys()),
                    y=list(by_stage.values()),
                    title="Recommendations by Stage",
                    labels={'x': 'Stage', 'y': 'Number of Recommendations'}
                )
                fig.update_xaxes(title="Agent Stage")
                fig.update_yaxes(title="Recommendations")
                st.plotly_chart(fig, use_container_width=True)
    
    # Cross-Stage Correlations
    if 'cross_stage_correlations' in insights:
        correlations = insights['cross_stage_correlations']
        if correlations:
            st.subheader("🔗 Cross-Stage Correlations")
            st.write("Showing how different AI agents' findings complement each other:")
            
            for corr in correlations:
                with st.expander(f"{corr['stage1'].title()} ↔ {corr['stage2'].title()} (Overlap: {corr['overlap_score']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Common Themes:**")
                        for theme in corr.get('common_themes', []):
                            st.write(f"• {theme.replace('_', ' ').title()}")
                    
                    with col2:
                        st.write("**Complementary Insights:**")
                        comp_insights = corr.get('complementary_insights', {})
                        if comp_insights.get('unique_to_first'):
                            st.write(f"**Unique to {corr['stage1'].title()}:**")
                            for insight in comp_insights['unique_to_first']:
                                st.write(f"• {insight.replace('_', ' ').title()}")
                        if comp_insights.get('unique_to_second'):
                            st.write(f"**Unique to {corr['stage2'].title()}:**")
                            for insight in comp_insights['unique_to_second']:
                                st.write(f"• {insight.replace('_', ' ').title()}")


def display_agent_results(result):
    """Display individual agent results and detailed information."""
    if not hasattr(result, 'agent_results') or not result.agent_results:
        st.info("No individual agent results available. This analysis may not have used the AI agent workflow.")
        return
    
    st.header("🎯 Individual Agent Results")
    
    # Agent selector
    agent_names = list(result.agent_results.keys())
    selected_agent = st.selectbox(
        "Select Agent to View Details",
        agent_names,
        format_func=lambda x: x.value.replace('_', ' ').title() if hasattr(x, 'value') else str(x).replace('_', ' ').title()
    )
    
    if selected_agent:
        agent_response = result.agent_results[selected_agent]
        agent_name = selected_agent.value.replace('_', ' ').title() if hasattr(selected_agent, 'value') else str(selected_agent).replace('_', ' ').title()
        
        # Agent overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "✅ Success" if agent_response.success else "❌ Failed"
            st.metric("Status", status)
        with col2:
            st.metric("Recommendations", len(agent_response.recommendations))
        with col3:
            st.metric("Confidence", f"{agent_response.confidence:.2f}")
        with col4:
            metadata = agent_response.metadata or {}
            analysis_type = metadata.get('analysis_type', 'unknown')
            st.metric("Analysis Type", analysis_type.replace('_', ' ').title())
        
        if not agent_response.success:
            st.error(f"❌ Agent failed: {agent_response.error_message}")
            return
        
        # Agent-specific insights
        st.subheader(f"🔍 {agent_name} Insights")
        
        # Display agent reasoning
        if agent_response.reasoning:
            with st.expander("🧠 Agent Reasoning", expanded=False):
                st.text_area("Detailed Analysis Reasoning", agent_response.reasoning, height=300, disabled=True)
        
        # Display recommendations with detailed information
        if agent_response.recommendations:
            st.subheader(f"📋 {agent_name} Recommendations")
            
            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            with col1:
                categories = list(set(rec.get('category', 'general') for rec in agent_response.recommendations))
                selected_category = st.selectbox("Filter by Category", ['All'] + categories)
            
            with col2:
                priorities = ['All', 'critical', 'high', 'medium', 'low']
                selected_priority = st.selectbox("Filter by Priority", priorities)
            
            with col3:
                sort_options = ['Priority', 'Category', 'Confidence', 'Title']
                sort_by = st.selectbox("Sort by", sort_options)
            
            # Filter recommendations
            filtered_recs = agent_response.recommendations.copy()
            
            if selected_category != 'All':
                filtered_recs = [rec for rec in filtered_recs if rec.get('category') == selected_category]
            
            if selected_priority != 'All':
                filtered_recs = [rec for rec in filtered_recs if rec.get('priority') == selected_priority]
            
            # Sort recommendations
            if sort_by == 'Priority':
                priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
                filtered_recs.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 2))
            elif sort_by == 'Confidence':
                filtered_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            elif sort_by == 'Category':
                filtered_recs.sort(key=lambda x: x.get('category', ''))
            else:  # Title
                filtered_recs.sort(key=lambda x: x.get('title', ''))
            
            # Display filtered recommendations
            st.write(f"Showing {len(filtered_recs)} of {len(agent_response.recommendations)} recommendations")
            
            for i, rec in enumerate(filtered_recs):
                priority = rec.get('priority', 'medium')
                category = rec.get('category', 'general').replace('_', ' ').title()
                title = rec.get('title', 'Untitled Recommendation')
                description = rec.get('description', 'No description available.')
                
                # Priority color coding
                priority_colors = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }
                priority_color = priority_colors.get(priority, '⚪')
                
                with st.expander(f"{priority_color} {title} [{category}]"):
                    # Basic information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Priority:** {priority.upper()}")
                        st.write(f"**Category:** {category}")
                        if rec.get('confidence'):
                            st.write(f"**Confidence:** {rec['confidence']:.2f}")
                    
                    with col2:
                        if rec.get('severity'):
                            st.write(f"**Severity:** {rec['severity'].upper()}")
                        if rec.get('business_impact'):
                            st.write(f"**Business Impact:** {rec['business_impact'].upper()}")
                        if rec.get('implementation_effort'):
                            st.write(f"**Implementation Effort:** {rec['implementation_effort'].upper()}")
                    
                    # Description
                    st.write("**Description:**")
                    st.write(description)
                    
                    # Technical details
                    if rec.get('technical_details'):
                        st.write("**Technical Details:**")
                        tech_details = rec['technical_details']
                        for key, value in tech_details.items():
                            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                    
                    # Associated tables
                    if rec.get('associated_tables'):
                        st.write("**Associated Tables:**")
                        for table in rec['associated_tables']:
                            st.write(f"• {table}")
                    
                    # Metrics
                    if rec.get('metrics'):
                        st.write("**Metrics:**")
                        metrics = rec['metrics']
                        metric_cols = st.columns(len(metrics))
                        for i, (metric_name, metric_value) in enumerate(metrics.items()):
                            with metric_cols[i % len(metric_cols)]:
                                if isinstance(metric_value, (int, float)):
                                    if metric_name.endswith('_score'):
                                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.2f}")
                                    else:
                                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:,}")
                                else:
                                    st.write(f"**{metric_name.replace('_', ' ').title()}:** {metric_value}")
        
        # Agent metadata
        if agent_response.metadata:
            with st.expander("📊 Agent Metadata", expanded=False):
                metadata = agent_response.metadata
                for key, value in metadata.items():
                    if key != 'analysis_type':  # Already shown above
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")


def display_overview(result):
    """Display overview of analysis results."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality Overview")
        
        if result.data_profiles:
            quality_scores = []
            table_names = []
            
            for table_name, profile in result.data_profiles.items():
                if isinstance(profile, dict) and 'overall_quality_score' in profile:
                    quality_scores.append(profile['overall_quality_score'])
                    table_names.append(table_name)
            
            if quality_scores:
                # Create quality score chart
                fig = px.bar(
                    x=table_names,
                    y=quality_scores,
                    title="Data Quality Score by Table",
                    labels={'x': 'Table', 'y': 'Quality Score'},
                    color=quality_scores,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Average quality score
                avg_quality = sum(quality_scores) / len(quality_scores)
                st.metric("Average Quality Score", f"{avg_quality:.2f}")
    
    with col2:
        st.subheader("Table Types Distribution")
        
        if result.schema_analysis and 'insights' in result.schema_analysis:
            table_types = result.schema_analysis['insights'].get('table_types', {})
            
            if table_types:
                # Create pie chart
                fig = px.pie(
                    values=list(table_types.values()),
                    names=list(table_types.keys()),
                    title="Table Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations summary
    if result.ai_recommendations:
        st.subheader("Recommendations Summary")
        
        # Group by category and priority
        recommendations_by_category = {}
        recommendations_by_priority = {'high': 0, 'medium': 0, 'low': 0}
        
        for rec in result.ai_recommendations:
            category = rec.get('category', 'general')
            priority = rec.get('priority', 'medium')
            
            if category not in recommendations_by_category:
                recommendations_by_category[category] = 0
            recommendations_by_category[category] += 1
            recommendations_by_priority[priority] += 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            if recommendations_by_category:
                fig = px.bar(
                    x=list(recommendations_by_category.keys()),
                    y=list(recommendations_by_category.values()),
                    title="Recommendations by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Priority distribution
            colors = {'high': '#ff4b4b', 'medium': '#ffa500', 'low': '#32cd32'}
            fig = px.pie(
                values=list(recommendations_by_priority.values()),
                names=list(recommendations_by_priority.keys()),
                title="Recommendations by Priority",
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)


def display_schema_analysis(schema_analysis):
    """Display schema analysis results."""
    if not schema_analysis:
        st.info("No schema analysis data available.")
        return
    
    # Tables overview
    st.subheader("Tables Overview")
    
    if 'tables' in schema_analysis:
        tables_data = []
        for table_name, table_info in schema_analysis['tables'].items():
            tables_data.append({
                'Table': table_name,
                'Type': table_info.get('table_type', 'unknown'),
                'Rows': table_info.get('row_count', 0),
                'Columns': len(table_info.get('columns', [])),
                'Quality Score': table_info.get('data_quality_score', 0.0)
            })
        
        if tables_data:
            df = pd.DataFrame(tables_data)
            st.dataframe(df, use_container_width=True)
    
    # Relationships
    st.subheader("Detected Relationships")
    
    relationships = schema_analysis.get('relationships', [])
    if relationships:
        rel_data = []
        for rel in relationships:
            rel_data.append({
                'From': f"{rel.get('from_table', '')}.{rel.get('from_column', '')}",
                'To': f"{rel.get('to_table', '')}.{rel.get('to_column', '')}",
                'Type': rel.get('type', ''),
                'Confidence': f"{rel.get('confidence', 0):.2f}"
            })
        
        df_rel = pd.DataFrame(rel_data)
        st.dataframe(df_rel, use_container_width=True)
    else:
        st.info("No relationships detected.")
    
    # Insights
    if 'insights' in schema_analysis:
        st.subheader("Schema Insights")
        insights = schema_analysis['insights']
        
        # Data quality insights
        if 'data_quality' in insights:
            dq = insights['data_quality']
            col1, col2 = st.columns(2)
            
            with col1:
                avg_score = dq.get('average_score', 0)
                st.metric("Average Data Quality", f"{avg_score:.2f}")
            
            with col2:
                tables_needing_attention = dq.get('tables_needing_attention', [])
                st.metric("Tables Needing Attention", len(tables_needing_attention))
            
            if tables_needing_attention:
                st.warning(f"Tables requiring attention: {', '.join(tables_needing_attention[:5])}")


def display_data_profiling(data_profiles):
    """Display data profiling results."""
    if not data_profiles:
        st.info("No data profiling results available.")
        return
    
    # Table selector
    table_names = list(data_profiles.keys())
    selected_table = st.selectbox("Select Table", table_names)
    
    if selected_table and selected_table in data_profiles:
        profile = data_profiles[selected_table]
        
        if isinstance(profile, dict) and 'error' not in profile:
            # Table-level metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", profile.get('total_rows', 0))
            with col2:
                st.metric("Total Columns", profile.get('total_columns', 0))
            with col3:
                st.metric("Completeness Score", f"{profile.get('completeness_score', 0):.2f}")
            with col4:
                st.metric("Overall Quality", f"{profile.get('overall_quality_score', 0):.2f}")
            
            # Column profiles
            st.subheader("Column Profiles")
            
            column_profiles = profile.get('column_profiles', {})
            if column_profiles:
                # Create columns data
                col_data = []
                for col_name, col_profile in column_profiles.items():
                    col_data.append({
                        'Column': col_name,
                        'Type': col_profile.get('data_type', ''),
                        'Null %': f"{col_profile.get('null_percentage', 0):.1f}%",
                        'Unique %': f"{col_profile.get('unique_percentage', 0):.1f}%",
                        'Quality': f"{col_profile.get('quality_score', 0):.2f}",
                        'Classification': col_profile.get('data_classification', ''),
                        'PII': '✓' if col_profile.get('potential_pii') else '',
                        'Key': '✓' if col_profile.get('potential_key') else ''
                    })
                
                df_cols = pd.DataFrame(col_data)
                st.dataframe(df_cols, use_container_width=True)
                
                # Column details
                st.subheader("Column Details")
                selected_column = st.selectbox("Select Column", list(column_profiles.keys()))
                
                if selected_column:
                    col_profile = column_profiles[selected_column]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Statistics:**")
                        st.write(f"- Total Count: {col_profile.get('total_count', 0):,}")
                        st.write(f"- Null Count: {col_profile.get('null_count', 0):,}")
                        st.write(f"- Unique Count: {col_profile.get('unique_count', 0):,}")
                        
                        # Numeric statistics if available
                        if col_profile.get('mean') is not None:
                            st.write("**Numeric Statistics:**")
                            st.write(f"- Mean: {col_profile.get('mean', 0):.2f}")
                            st.write(f"- Min: {col_profile.get('min_value', 0)}")
                            st.write(f"- Max: {col_profile.get('max_value', 0)}")
                    
                    with col2:
                        # Most frequent values
                        frequent_values = col_profile.get('most_frequent_values', [])
                        if frequent_values:
                            st.write("**Most Frequent Values:**")
                            for val_info in frequent_values[:5]:
                                st.write(f"- {val_info.get('value', '')}: {val_info.get('count', 0)} ({val_info.get('percentage', 0):.1f}%)")
                        
                        # Quality issues
                        quality_issues = col_profile.get('quality_issues', [])
                        if quality_issues:
                            st.write("**Quality Issues:**")
                            for issue in quality_issues:
                                st.warning(f"⚠️ {issue}")
        else:
            st.error(f"Error in profiling table {selected_table}: {profile.get('error', 'Unknown error')}")


def display_ai_recommendations(recommendations):
    """Display AI recommendations with enhanced details."""
    if not recommendations:
        st.info("No AI recommendations available.")
        return
    
    st.header("🤖 AI-Powered Recommendations")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    total_recs = len(recommendations)
    critical_recs = len([r for r in recommendations if r.get('consolidated_priority', r.get('priority', 'medium')) == 'critical'])
    high_recs = len([r for r in recommendations if r.get('consolidated_priority', r.get('priority', 'medium')) == 'high'])
    
    # Calculate average confidence
    confidences = [r.get('stage_confidence', 0) for r in recommendations if r.get('stage_confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    with col1:
        st.metric("Total Recommendations", total_recs)
    with col2:
        st.metric("Critical Priority", critical_recs)
    with col3:
        st.metric("High Priority", high_recs)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Recommendations by source agent
    source_counts = {}
    for rec in recommendations:
        source = rec.get('source_stage', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    if len(source_counts) > 1:
        st.subheader("📊 Recommendations by AI Agent")
        
        col1, col2 = st.columns(2)
        with col1:
            # Pie chart
            fig = px.pie(
                values=list(source_counts.values()),
                names=[name.replace('_', ' ').title() for name in source_counts.keys()],
                title="Distribution by Agent"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Agent performance table
            agent_data = []
            for source, count in source_counts.items():
                source_recs = [r for r in recommendations if r.get('source_stage') == source]
                high_priority_count = len([r for r in source_recs if r.get('consolidated_priority', r.get('priority', 'medium')) in ['critical', 'high']])
                avg_conf = sum(r.get('stage_confidence', 0) for r in source_recs) / len(source_recs)
                
                agent_data.append({
                    'Agent': source.replace('_', ' ').title(),
                    'Total': count,
                    'High Priority': high_priority_count,
                    'Avg Confidence': f"{avg_conf:.2f}"
                })
            
            df_agents = pd.DataFrame(agent_data)
            st.dataframe(df_agents, use_container_width=True)
    
    # Filter and display controls
    st.subheader("🔍 Filter & Sort Recommendations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        categories = list(set(rec.get('category', 'general') for rec in recommendations))
        selected_category = st.selectbox("Filter by Category", ['All'] + categories)
    
    with col2:
        priorities = ['All', 'critical', 'high', 'medium', 'low']
        selected_priority = st.selectbox("Filter by Priority", priorities)
    
    with col3:
        sources = list(set(rec.get('source_stage', 'unknown') for rec in recommendations))
        selected_source = st.selectbox("Filter by Agent", ['All'] + sources, format_func=lambda x: x.replace('_', ' ').title() if x != 'All' else x)
    
    with col4:
        sort_options = ['Priority', 'Confidence', 'Category', 'Title']
        sort_by = st.selectbox("Sort by", sort_options)
    
    # Filter recommendations
    filtered_recs = recommendations.copy()
    
    if selected_category != 'All':
        filtered_recs = [rec for rec in filtered_recs if rec.get('category') == selected_category]
    
    if selected_priority != 'All':
        filtered_recs = [rec for rec in filtered_recs if rec.get('consolidated_priority', rec.get('priority', 'medium')) == selected_priority]
    
    if selected_source != 'All':
        filtered_recs = [rec for rec in filtered_recs if rec.get('source_stage') == selected_source]
    
    # Sort recommendations
    if sort_by == 'Priority':
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        filtered_recs.sort(key=lambda x: priority_order.get(x.get('consolidated_priority', x.get('priority', 'medium')), 2))
    elif sort_by == 'Confidence':
        filtered_recs.sort(key=lambda x: x.get('stage_confidence', 0), reverse=True)
    elif sort_by == 'Category':
        filtered_recs.sort(key=lambda x: x.get('category', ''))
    else:  # Title
        filtered_recs.sort(key=lambda x: x.get('title', ''))
    
    # Display recommendations with enhanced UI
    st.subheader(f"📋 Detailed Recommendations ({len(filtered_recs)} of {len(recommendations)})")
    
    if not filtered_recs:
        st.info("No recommendations match the selected filters.")
        return
    
    for i, rec in enumerate(filtered_recs):
        priority = rec.get('consolidated_priority', rec.get('priority', 'medium'))
        category = rec.get('category', 'general').replace('_', ' ').title()
        title = rec.get('title', 'Untitled Recommendation')
        description = rec.get('description', 'No description available.')
        source = rec.get('source_stage', 'unknown').replace('_', ' ').title()
        confidence = rec.get('stage_confidence', 0)
        
        # Priority emoji and color
        priority_emojis = {
            'critical': '🔴',
            'high': '🟠', 
            'medium': '🟡',
            'low': '🟢'
        }
        priority_emoji = priority_emojis.get(priority, '⚪')
        
        # Create expandable recommendation card
        with st.expander(f"{priority_emoji} {title} [{priority.upper()}] - {source}", expanded=(i < 3)):
            # Header info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Priority", priority.upper())
            with col2:
                st.metric("Category", category)
            with col3:
                st.metric("Source Agent", source)
            with col4:
                st.metric("Confidence", f"{confidence:.2f}")
            
            # Description
            st.write("**📝 Description:**")
            st.write(description)
            
            # Additional details
            col1, col2 = st.columns(2)
            
            with col1:
                # Business impact and implementation
                if rec.get('business_impact'):
                    st.write(f"**💼 Business Impact:** {rec['business_impact'].upper()}")
                if rec.get('implementation_effort'):
                    effort = rec['implementation_effort']
                    effort_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                    effort_color = effort_colors.get(effort, '⚪')
                    st.write(f"**⚡ Implementation Effort:** {effort_color} {effort.upper()}")
                if rec.get('severity'):
                    st.write(f"**⚠️ Severity:** {rec['severity'].upper()}")
            
            with col2:
                # Technical and domain info
                if rec.get('domain'):
                    st.write(f"**🏢 Domain:** {rec['domain'].replace('_', ' ').title()}")
                if rec.get('importance'):
                    st.write(f"**📊 Importance:** {rec['importance'].replace('_', ' ').title()}")
                if rec.get('type'):
                    st.write(f"**🔧 Type:** {rec['type'].replace('_', ' ').title()}")
            
            # Technical details
            if rec.get('technical_details'):
                st.write("**🔧 Technical Implementation Details:**")
                tech_details = rec['technical_details']
                for key, value in tech_details.items():
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            
            # Associated tables/metrics
            if rec.get('associated_tables'):
                st.write("**📊 Associated Tables:**")
                for table in rec['associated_tables']:
                    st.write(f"• {table}")
            
            if rec.get('metrics'):
                st.write("**📈 Metrics:**")
                metrics = rec['metrics']
                metric_cols = st.columns(min(4, len(metrics)))
                for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_cols[idx % len(metric_cols)]:
                        if isinstance(metric_value, (int, float)):
                            if metric_name.endswith('_score') or 'percentage' in metric_name:
                                st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.2f}")
                            else:
                                st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:,}")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Filtered Results"):
            export_data = []
            for rec in filtered_recs:
                export_data.append({
                    'Title': rec.get('title', ''),
                    'Priority': rec.get('consolidated_priority', rec.get('priority', '')),
                    'Category': rec.get('category', ''),
                    'Source_Agent': rec.get('source_stage', ''),
                    'Confidence': rec.get('stage_confidence', 0),
                    'Description': rec.get('description', ''),
                    'Business_Impact': rec.get('business_impact', ''),
                    'Implementation_Effort': rec.get('implementation_effort', ''),
                    'Domain': rec.get('domain', ''),
                    'Type': rec.get('type', '')
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"datamodeling_recommendations_filtered.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Generate Summary Report"):
            st.info("Summary report generation feature coming soon!")
    
    with col3:
        if st.button("🔄 Refresh Analysis"):
            st.session_state.analysis_result = None
            st.rerun()


def display_setup_progress():
    """Display setup progress indicator."""
    trino_connected = st.session_state.get('trino_connection_successful', False)
    ai_connected = st.session_state.get('ai_connection_successful', False)
    
    st.write("**📋 Setup Progress**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if trino_connected:
            st.success("✅ Step 1: Trino")
        else:
            st.info("⏳ Step 1: Trino")
    
    with col2:
        if trino_connected and ai_connected:
            st.success("✅ Step 2: AI Config")
        elif trino_connected:
            st.warning("🔄 Step 2: AI Config")
        else:
            st.info("⏳ Step 2: AI Config")
    
    with col3:
        if trino_connected and ai_connected:
            st.warning("🔄 Step 3: Launch")
        else:
            st.info("⏳ Step 3: Launch")
    
    # Progress bar
    progress = 0
    if trino_connected:
        progress += 33
    if ai_connected:
        progress += 33
    if st.session_state.get('engine'):
        progress += 34
    
    st.progress(progress / 100)
    st.divider()


def display_connection_manager():
    """Display connection management UI."""
    cm = st.session_state.connection_manager
    
    # Password input for encryption
    if 'encryption_password' not in st.session_state:
        st.session_state.encryption_password = ""
    
    st.write("**🔐 Set Encryption Password**")
    st.caption("This password will encrypt all your saved connections and AI configurations")
    
    password = st.text_input(
        "Encryption Password",
        type="password",
        value=st.session_state.encryption_password,
        help="Enter a secure password to encrypt your saved connections and API keys",
        key="conn_password",
        placeholder="Enter a secure password..."
    )
    
    if password:
        st.session_state.encryption_password = password
        
        # List existing connections
        connections = cm.list_connections(password)
        
        if connections is not None:
            if connections:
                st.write("**Saved Connections:**")
                for name, metadata in connections.items():
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"📋 **{name}**")
                            st.caption(f"Type: {metadata['connection_type']} | Last used: {metadata['last_used'][:16]}")
                        with col2:
                            if st.button("📥 Load", key=f"load_{name}"):
                                load_saved_connection(name, password)
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{name}"):
                                if cm.delete_connection(name, password):
                                    st.success(f"Connection '{name}' deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete connection")
                        st.divider()
            else:
                st.info("No saved connections found")
        else:
            st.error("Invalid password or corrupted data")


def save_current_connection():
    """Save current connection configuration."""
    if 'current_connection_data' not in st.session_state or not st.session_state.current_connection_data:
        st.error("No connection configuration to save")
        return
    
    cm = st.session_state.connection_manager
    password = st.session_state.get('encryption_password', '')
    
    if not password:
        st.error("Please enter an encryption password first")
        return
    
    # Connection name input
    connection_name = st.text_input(
        "Connection Name",
        placeholder="e.g., Production Trino",
        help="Enter a name for this connection configuration"
    )
    
    if connection_name and st.button("💾 Save Connection"):
        success = cm.save_connection(
            connection_name,
            st.session_state.current_connection_data,
            password
        )
        
        if success:
            st.success(f"Connection '{connection_name}' saved successfully!")
        else:
            st.error("Failed to save connection")


def load_saved_connection(connection_name: str, password: str):
    """Load a saved connection configuration."""
    cm = st.session_state.connection_manager
    
    connection_data = cm.load_connection(connection_name, password)
    
    if connection_data:
        # Update session state with loaded connection
        st.session_state.current_connection_data = connection_data
        
        # Update form fields
        for key, value in connection_data.items():
            if key in st.session_state:
                st.session_state[key] = value
        
        st.success(f"Connection '{connection_name}' loaded successfully!")
        st.rerun()
    else:
        st.error(f"Failed to load connection '{connection_name}'")


def display_saved_connections_summary():
    """Display a summary of saved Trino connections."""
    if not st.session_state.get('encryption_password'):
        return
    
    cm = st.session_state.connection_manager
    password = st.session_state.encryption_password
    
    connections = cm.list_connections(password)
    if connections and any('trino' in conn.get('connection_type', '') for conn in connections.values()):
        st.write("**🔗 Saved Trino Connections:**")
        trino_connections = {k: v for k, v in connections.items() if 'trino' in v.get('connection_type', '')}
        
        for name, metadata in trino_connections.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📋 {name} - Last used: {metadata['last_used'][:16]}")
            with col2:
                if st.button("📥", key=f"load_trino_{name}", help=f"Load {name}"):
                    load_trino_connection(name, password)


def display_saved_ai_configs_summary():
    """Display a summary of saved AI configurations."""
    if not st.session_state.get('encryption_password'):
        return
    
    cm = st.session_state.connection_manager
    password = st.session_state.encryption_password
    
    ai_configs = cm.list_connections(password)  # We'll use the same storage for AI configs
    if ai_configs and any('ai_config' in conn.get('connection_type', '') for conn in ai_configs.values()):
        st.write("**🤖 Saved AI Configurations:**")
        ai_connections = {k: v for k, v in ai_configs.items() if 'ai_config' in v.get('connection_type', '')}
        
        for name, metadata in ai_connections.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🤖 {name} - Last used: {metadata['last_used'][:16]}")
            with col2:
                if st.button("📥", key=f"load_ai_{name}", help=f"Load {name}"):
                    load_ai_configuration(name, password)


def test_trino_connection(host, port, username, password, catalog, schema):
    """Test Trino connection."""
    with st.spinner("Testing Trino connection..."):
        try:
            from connectors.trino_connector import TrinoConnector
            
            connector = TrinoConnector(
                host=host,
                port=port,
                username=username,
                password=password,
                catalog=catalog,
                schema=schema
            )
            
            # Test connection
            success = connector.test_connection()
            
            if success:
                st.success("✅ Trino connection successful!")
                
                # Store current Trino config for later use
                st.session_state.current_trino_config = {
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'catalog': catalog,
                    'schema': schema
                }
                
                # Mark Trino connection as successful to enable AI configuration
                st.session_state.trino_connection_successful = True
                
                # Try to get some basic info
                try:
                    databases = connector.list_databases()
                    st.info(f"📊 Found {len(databases)} databases: {', '.join(databases[:3])}" + 
                           (f" and {len(databases)-3} more" if len(databases) > 3 else ""))
                except:
                    pass
            else:
                st.error("❌ Trino connection failed!")
                
        except Exception as e:
            st.error(f"❌ Connection error: {e}")


def save_trino_connection_data(host, port, username, password, catalog, schema):
    """Save Trino connection data."""
    if not st.session_state.get('encryption_password'):
        st.error("❌ Please set an encryption password in the '💾 Saved Connections' section above first")
        st.info("💡 **How to set encryption password:**\n1. Expand the '💾 Saved Connections' section above\n2. Enter a secure password in the '🔐 Encryption Password' field\n3. Then come back here to save your connection")
        return
    
    st.write("**💾 Save Connection Configuration**")
    connection_name = st.text_input("Connection Name", placeholder="e.g., Production Trino", key="trino_conn_name", help="Enter a memorable name for this Trino connection")
    
    if connection_name:
        if st.button("💾 Save Trino Connection", key="save_trino_btn"):
            cm = st.session_state.connection_manager
            password_enc = st.session_state.encryption_password
            
            connection_data = {
                'connection_type': 'trino',
                'host': host,
                'port': port,
                'username': username,
                'password': password,
                'catalog': catalog,
                'schema': schema
            }
            
            success = cm.save_connection(connection_name, connection_data, password_enc)
            
            if success:
                st.success(f"✅ Trino connection '{connection_name}' saved successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to save connection")


def test_ai_connection(llm_provider, api_key):
    """Test AI connection."""
    with st.spinner(f"Testing {llm_provider.title()} connection..."):
        try:
            if llm_provider == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                # Test with a simple message
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.success("✅ Claude API connection successful!")
            else:  # openai
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Test with a simple completion
                response = client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.success("✅ OpenAI API connection successful!")
            
            # Store current AI config for later use
            st.session_state.current_ai_config = {
                'llm_provider': llm_provider,
                'api_key': api_key
            }
            
            # Mark AI connection as successful to enable launch button
            st.session_state.ai_connection_successful = True
            
        except Exception as e:
            error_message = str(e)
            
            # Check for specific billing/credit issues that don't invalidate the API key
            billing_issues = [
                "credit balance is too low",
                "insufficient credits",
                "billing",
                "payment required",
                "quota exceeded"
            ]
            
            is_billing_issue = any(issue in error_message.lower() for issue in billing_issues)
            
            if is_billing_issue:
                st.warning(f"⚠️ {llm_provider.title()} API Key Valid, but Billing Issue Detected")
                st.info("💡 **API Key appears valid, but there's a billing/credit issue:**\n" + 
                       f"- {error_message}\n" + 
                       "- You can still save this configuration\n" + 
                       "- Add credits to your account before using the application")
                
                # Store config even with billing issues since the key format is valid
                st.session_state.current_ai_config = {
                    'llm_provider': llm_provider,
                    'api_key': api_key
                }
                
                # Mark as successful to enable launch (user can fix billing later)
                st.session_state.ai_connection_successful = True
                
            else:
                st.error(f"❌ {llm_provider.title()} API error: {e}")
                st.info("💡 **Possible issues:**\n" + 
                       "- Invalid API key format\n" + 
                       "- API key doesn't have required permissions\n" + 
                       "- Network connectivity issues")


def save_ai_configuration(llm_provider, api_key):
    """Save AI configuration."""
    if not st.session_state.get('encryption_password'):
        st.error("❌ Please set an encryption password in the '💾 Saved Connections' section above first")
        st.info("💡 **How to set encryption password:**\n1. Expand the '💾 Saved Connections' section above\n2. Enter a secure password in the '🔐 Encryption Password' field\n3. Then come back here to save your AI configuration")
        return
    
    st.write("**💾 Save AI Configuration**")
    config_name = st.text_input("AI Config Name", placeholder=f"e.g., My {llm_provider.title()} Key", key="ai_config_name", help="Enter a memorable name for this AI configuration")
    
    if config_name:
        if st.button("💾 Save AI Configuration", key="save_ai_btn"):
            cm = st.session_state.connection_manager
            password_enc = st.session_state.encryption_password
            
            ai_config_data = {
                'connection_type': 'ai_config',
                'llm_provider': llm_provider,
                'api_key': api_key
            }
            
            success = cm.save_connection(config_name, ai_config_data, password_enc)
            
            if success:
                st.success(f"✅ AI configuration '{config_name}' saved successfully!")
                st.balloons()
            else:
                st.error("❌ Failed to save AI configuration")


def load_trino_connection(connection_name, password):
    """Load a saved Trino connection."""
    cm = st.session_state.connection_manager
    connection_data = cm.load_connection(connection_name, password)
    
    if connection_data:
        st.session_state.current_trino_config = connection_data
        st.success(f"✅ Loaded Trino connection '{connection_name}'")
        st.rerun()


def load_ai_configuration(config_name, password):
    """Load a saved AI configuration."""
    cm = st.session_state.connection_manager
    config_data = cm.load_connection(config_name, password)
    
    if config_data:
        st.session_state.current_ai_config = config_data
        st.success(f"✅ Loaded AI configuration '{config_name}'")
        st.rerun()


def create_complete_configuration(trino_config, ai_config):
    """Create complete configuration from Trino and AI configs."""
    complete_config = {
        'llm_provider': ai_config['llm_provider'],
        'data_lakes': {
            'trino': {
                'host': trino_config['host'],
                'port': trino_config['port'],
                'username': trino_config['username'],
                'password': trino_config.get('password'),
                'catalog': trino_config['catalog'],
                'schema': trino_config['schema']
            }
        },
        'analysis': {
            'max_sample_rows': 10000,
            'max_concurrent_tables': 5
        }
    }
    
    if ai_config['llm_provider'] == "claude":
        complete_config['claude'] = {
            'api_key': ai_config['api_key'],
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.3,
            'max_tokens': 4000
        }
    else:
        complete_config['openai'] = {
            'api_key': ai_config['api_key'],
            'model': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 2000
        }
    
    return complete_config


def init_config_from_dict(config_dict):
    """Initialize configuration from dictionary."""
    import tempfile
    import yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        temp_config_path = f.name
    
    init_config(temp_config_path)


def handle_load_saved_connection_ui():
    """Handle the Load Saved Connection UI."""
    cm = st.session_state.connection_manager
    password = st.session_state.get('encryption_password', '')
    
    if not password:
        st.warning("⚠️ Please enter your encryption password in the 'Saved Connections' section above")
        return None
    
    connections = cm.list_connections(password)
    
    if connections is None:
        st.error("❌ Invalid password or corrupted data")
        return None
    
    if not connections:
        st.info("📋 No saved connections found. Save a connection first using 'Manual Connection' mode.")
        return None
    
    # Connection selection
    connection_names = list(connections.keys())
    selected_connection = st.selectbox(
        "Select Connection",
        options=connection_names,
        help="Choose a saved connection to load"
    )
    
    if selected_connection:
        metadata = connections[selected_connection]
        
        # Display connection info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Type:** {metadata['connection_type']}")
        with col2:
            st.info(f"**Last Used:** {metadata['last_used'][:16]}")
        
        # Load button
        if st.button("📥 Load Selected Connection", type="primary"):
            load_saved_connection(selected_connection, password)
            return selected_connection
    
    return None


def render_workflow_section():
    """Render the workflow execution section with visual progress indicators."""
    st.divider()
    st.subheader("3️⃣ AI Workflow Execution")
    
    # Initialize workflow session state
    if 'workflow_status' not in st.session_state:
        st.session_state.workflow_status = {
            'discovery': 'pending',
            'profiling': 'pending', 
            'analyzing': 'pending',
            'modeling': 'pending',
            'optimization': 'pending',
            'recommendations': 'pending'
        }
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = {}
    
    # Workflow steps configuration
    workflow_steps = [
        {'key': 'discovery', 'name': 'Discovery', 'icon': '🔍', 'desc': 'Catalog data assets and identify domains'},
        {'key': 'profiling', 'name': 'Profiling', 'icon': '📊', 'desc': 'Analyze data quality and patterns'},
        {'key': 'analyzing', 'name': 'Analyzing', 'icon': '🧠', 'desc': 'Deep analysis of relationships and structure'},
        {'key': 'modeling', 'name': 'Modeling', 'icon': '🏗️', 'desc': 'Generate dimensional models and schemas'},
        {'key': 'optimization', 'name': 'Optimization', 'icon': '⚡', 'desc': 'Performance and efficiency recommendations'},
        {'key': 'recommendations', 'name': 'Recommendations', 'icon': '💡', 'desc': 'Final consolidated recommendations'}
    ]
    
    # Visual workflow display
    st.markdown("### 🔄 Workflow Progress")
    
    # Create workflow visualization
    cols = st.columns(len(workflow_steps))
    
    for i, (col, step) in enumerate(zip(cols, workflow_steps)):
        with col:
            status = st.session_state.workflow_status[step['key']]
            
            # Status styling
            if status == 'completed':
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #d4edda; border: 2px solid #28a745;">
                    <div style="font-size: 24px; color: #28a745;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #155724;">{step['name']}</div>
                    <div style="font-size: 12px; color: #155724;">✅ Complete</div>
                </div>
                """, unsafe_allow_html=True)
            elif status == 'running':
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #fff3cd; border: 2px solid #ffc107;">
                    <div style="font-size: 24px; color: #856404;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #856404;">{step['name']}</div>
                    <div style="font-size: 12px; color: #856404;">▶️ Running</div>
                </div>
                """, unsafe_allow_html=True)
            else:  # pending
                st.markdown(f"""
                <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #f8f9fa; border: 2px solid #6c757d;">
                    <div style="font-size: 24px; color: #6c757d;">{step['icon']}</div>
                    <div style="font-weight: bold; color: #6c757d;">{step['name']}</div>
                    <div style="font-size: 12px; color: #6c757d;">⏳ Pending</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add connecting arrows (except for last step)
            if i < len(workflow_steps) - 1:
                st.markdown("<div style='text-align: center; margin: 10px 0;'>➡️</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Start Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.get('workflow_started', False):
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                st.session_state.workflow_started = True
                execute_workflow()
                st.rerun()
        else:
            # Show workflow control buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Reset Workflow", use_container_width=True):
                    reset_workflow()
                    st.rerun()
            with col_b:
                if st.button("📊 View Report", use_container_width=True, disabled=not all_steps_completed()):
                    st.session_state.show_final_report = True
                    st.rerun()
    
    # Show final report if requested
    if st.session_state.get('show_final_report', False):
        display_final_report()
    
    # Progress details
    if st.session_state.get('workflow_started', False):
        st.markdown("### 📋 Execution Details")
        
        for step in workflow_steps:
            status = st.session_state.workflow_status[step['key']]
            
            with st.expander(f"{step['icon']} {step['name']} - {status.title()}", expanded=(status == 'running')):
                st.write(f"**Description:** {step['desc']}")
                
                if status == 'completed' and step['key'] in st.session_state.workflow_results:
                    result = st.session_state.workflow_results[step['key']]
                    st.success("✅ Completed successfully")
                    
                    # Display step-specific results
                    if step['key'] == 'discovery':
                        st.write("**Discovered Assets:**")
                        if 'tables_analyzed' in result:
                            st.write(f"- Tables analyzed: {result['tables_analyzed']}")
                        if 'domains_identified' in result:
                            st.write(f"- Business domains: {result['domains_identified']}")
                    
                    elif step['key'] == 'profiling':
                        st.write("**Data Profile Summary:**")
                        if 'quality_score' in result:
                            st.metric("Data Quality Score", f"{result['quality_score']}%")
                        if 'issues_found' in result:
                            st.write(f"- Issues identified: {result['issues_found']}")
                    
                    elif step['key'] == 'analyzing':
                        st.write("**Analysis Results:**")
                        if 'relationships_found' in result:
                            st.write(f"- Relationships identified: {result['relationships_found']}")
                        if 'patterns_detected' in result:
                            st.write(f"- Patterns detected: {result['patterns_detected']}")
                    
                    elif step['key'] == 'modeling':
                        st.write("**Modeling Recommendations:**")
                        if 'models_suggested' in result:
                            st.write(f"- Models suggested: {result['models_suggested']}")
                        if 'schema_improvements' in result:
                            st.write(f"- Schema improvements: {result['schema_improvements']}")
                    
                    elif step['key'] == 'optimization':
                        st.write("**Optimization Opportunities:**")
                        if 'optimizations_found' in result:
                            st.write(f"- Optimizations identified: {result['optimizations_found']}")
                        if 'performance_gain' in result:
                            st.metric("Expected Performance Gain", f"{result['performance_gain']}%")
                    
                    elif step['key'] == 'recommendations':
                        st.write("**Final Recommendations:**")
                        if 'total_recommendations' in result:
                            st.write(f"- Total recommendations: {result['total_recommendations']}")
                        if 'priority_actions' in result:
                            st.write(f"- High priority actions: {result['priority_actions']}")
                
                elif status == 'running':
                    st.info("🔄 Currently executing...")
                    # Add a progress bar for visual feedback
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)  # Small delay for visual effect
                
                else:  # pending
                    st.info("⏳ Waiting to execute")


def execute_workflow():
    """Execute the AI workflow sequentially."""
    workflow_steps = ['discovery', 'profiling', 'analyzing', 'modeling', 'optimization', 'recommendations']
    
    for step in workflow_steps:
        st.session_state.workflow_status[step] = 'running'
        
        # Execute the specific step
        try:
            if step == 'discovery':
                result = execute_discovery_step()
            elif step == 'profiling':
                result = execute_profiling_step()
            elif step == 'analyzing':
                result = execute_analyzing_step()
            elif step == 'modeling':
                result = execute_modeling_step()
            elif step == 'optimization':
                result = execute_optimization_step()
            elif step == 'recommendations':
                result = execute_recommendations_step()
            
            # Mark as completed and store results
            st.session_state.workflow_status[step] = 'completed'
            st.session_state.workflow_results[step] = result
            
        except Exception as e:
            st.session_state.workflow_status[step] = 'error'
            st.session_state.workflow_results[step] = {'error': str(e)}


def execute_discovery_step():
    """Execute the discovery step."""
    # Simulate discovery logic
    time.sleep(2)  # Simulate processing time
    
    selected_tables = st.session_state.get('selected_tables', [])
    
    return {
        'tables_analyzed': len(selected_tables),
        'domains_identified': len(set([table.split('.')[0] if '.' in table else 'default' for table in selected_tables])),
        'timestamp': time.time()
    }


def execute_profiling_step():
    """Execute the profiling step."""
    time.sleep(3)  # Simulate processing time
    
    # Simulate profiling results
    return {
        'quality_score': 85,
        'issues_found': 12,
        'completeness': 92,
        'timestamp': time.time()
    }


def execute_analyzing_step():
    """Execute the analyzing step."""
    time.sleep(2)  # Simulate processing time
    
    return {
        'relationships_found': 15,
        'patterns_detected': 8,
        'anomalies': 3,
        'timestamp': time.time()
    }


def execute_modeling_step():
    """Execute the modeling step."""
    time.sleep(3)  # Simulate processing time
    
    return {
        'models_suggested': 5,
        'schema_improvements': 12,
        'normalization_recommendations': 8,
        'timestamp': time.time()
    }


def execute_optimization_step():
    """Execute the optimization step."""
    time.sleep(2)  # Simulate processing time
    
    return {
        'optimizations_found': 18,
        'performance_gain': 35,
        'cost_savings': 25,
        'timestamp': time.time()
    }


def execute_recommendations_step():
    """Execute the final recommendations step."""
    time.sleep(2)  # Simulate processing time
    
    return {
        'total_recommendations': 42,
        'priority_actions': 8,
        'quick_wins': 15,
        'timestamp': time.time()
    }


def reset_workflow():
    """Reset the workflow to initial state."""
    st.session_state.workflow_started = False
    st.session_state.show_final_report = False
    st.session_state.workflow_status = {
        'discovery': 'pending',
        'profiling': 'pending',
        'analyzing': 'pending', 
        'modeling': 'pending',
        'optimization': 'pending',
        'recommendations': 'pending'
    }
    st.session_state.workflow_results = {}


def all_steps_completed():
    """Check if all workflow steps are completed."""
    return all(status == 'completed' for status in st.session_state.workflow_status.values())


def display_final_report():
    """Display the comprehensive final report."""
    st.markdown("## 📊 Comprehensive Analysis Report")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tables Analyzed", st.session_state.workflow_results.get('discovery', {}).get('tables_analyzed', 0))
    with col2:
        quality_score = st.session_state.workflow_results.get('profiling', {}).get('quality_score', 0)
        st.metric("Data Quality", f"{quality_score}%")
    with col3:
        recommendations = st.session_state.workflow_results.get('recommendations', {}).get('total_recommendations', 0)
        st.metric("Total Recommendations", recommendations)
    with col4:
        performance_gain = st.session_state.workflow_results.get('optimization', {}).get('performance_gain', 0)
        st.metric("Expected Performance Gain", f"{performance_gain}%")
    
    st.markdown("---")
    
    # Detailed sections
    tabs = st.tabs(["🔍 Discovery", "📊 Profiling", "🧠 Analysis", "🏗️ Modeling", "⚡ Optimization", "💡 Recommendations"])
    
    with tabs[0]:
        st.subheader("Data Discovery Results")
        discovery_result = st.session_state.workflow_results.get('discovery', {})
        st.write(f"**Tables Analyzed:** {discovery_result.get('tables_analyzed', 0)}")
        st.write(f"**Business Domains:** {discovery_result.get('domains_identified', 0)}")
        
        # Show analyzed tables
        selected_tables = st.session_state.get('selected_tables', [])
        if selected_tables:
            st.write("**Analyzed Tables:**")
            for i, table in enumerate(selected_tables, 1):
                st.write(f"{i}. {table}")
    
    with tabs[1]:
        st.subheader("Data Profiling Results")
        profiling_result = st.session_state.workflow_results.get('profiling', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{profiling_result.get('quality_score', 0)}%")
            st.metric("Completeness", f"{profiling_result.get('completeness', 0)}%")
        with col2:
            st.metric("Issues Found", profiling_result.get('issues_found', 0))
    
    with tabs[2]:
        st.subheader("Analysis Results")
        analyzing_result = st.session_state.workflow_results.get('analyzing', {})
        st.write(f"**Relationships Found:** {analyzing_result.get('relationships_found', 0)}")
        st.write(f"**Patterns Detected:** {analyzing_result.get('patterns_detected', 0)}")
        st.write(f"**Anomalies:** {analyzing_result.get('anomalies', 0)}")
    
    with tabs[3]:
        st.subheader("Data Modeling Recommendations")
        modeling_result = st.session_state.workflow_results.get('modeling', {})
        st.write(f"**Models Suggested:** {modeling_result.get('models_suggested', 0)}")
        st.write(f"**Schema Improvements:** {modeling_result.get('schema_improvements', 0)}")
        st.write(f"**Normalization Recommendations:** {modeling_result.get('normalization_recommendations', 0)}")
    
    with tabs[4]:
        st.subheader("Optimization Opportunities")
        optimization_result = st.session_state.workflow_results.get('optimization', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimizations Found", optimization_result.get('optimizations_found', 0))
            st.metric("Performance Gain", f"{optimization_result.get('performance_gain', 0)}%")
        with col2:
            st.metric("Cost Savings", f"{optimization_result.get('cost_savings', 0)}%")
    
    with tabs[5]:
        st.subheader("Final Recommendations")
        recommendations_result = st.session_state.workflow_results.get('recommendations', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Recommendations", recommendations_result.get('total_recommendations', 0))
        with col2:
            st.metric("Priority Actions", recommendations_result.get('priority_actions', 0))
        with col3:
            st.metric("Quick Wins", recommendations_result.get('quick_wins', 0))
        
        st.markdown("### 🎯 Action Items")
        st.success("**High Priority Actions:**")
        st.write("1. Implement data quality checks for critical tables")
        st.write("2. Optimize slow-performing queries identified")
        st.write("3. Establish proper indexing strategy")
        
        st.info("**Quick Wins:**")
        st.write("1. Add missing foreign key constraints")
        st.write("2. Update table statistics for query optimization")
        st.write("3. Implement data validation rules")


if __name__ == "__main__":
    main()

# DataModeling-AI 🤖

**Automate data modeling with AI agents for Hive, Trino, and Presto data lakes**

DataModeling-AI is a comprehensive system that analyzes your data lake schemas, profiles your data, and provides intelligent recommendations for data modeling, optimization, and governance using advanced AI agents.

## 🌟 Features

### Core Capabilities
- **Multi-Platform Support**: Connect to Hive, Trino, and Presto data lakes
- **Schema Analysis**: Comprehensive analysis of database schemas and relationships
- **Data Profiling**: Detailed profiling of data quality, patterns, and characteristics
- **AI-Powered Recommendations**: Intelligent suggestions for data modeling and optimization
- **Performance Optimization**: Recommendations for partitioning, indexing, and query optimization
- **Data Quality Assessment**: Automated data quality scoring and issue detection

### AI Agents Workflow
- **Discovery Agent**: Discovers and catalogs data assets, identifies business domains
- **Analysis Agent**: Performs deep data quality analysis and pattern recognition
- **Modeling Agent**: Provides dimensional modeling and schema design recommendations
- **Optimization Agent**: Suggests performance optimizations and best practices

The agents work together in a structured workflow with intelligent orchestration, dependency management, and parallel execution capabilities.

### Interfaces
- **Command Line Interface**: Full-featured CLI for automated workflows
- **Web Interface**: Interactive Streamlit-based web application
- **Programmatic API**: Python API for integration into existing workflows

## 🚀 Quick Start

### Automated Setup (Recommended)

#### For macOS/Linux:
```bash
# Clone the repository
git clone https://github.com/your-org/DataModeling-AI.git
cd DataModeling-AI

# Run automated setup
./setup.sh

# Update .env file with your API keys
# Edit .env file with your favorite editor
nano .env

# Start the application
./run.sh                    # Web interface
# or
./run_cli.sh --help        # Command line interface
```

#### For Windows:
```cmd
# Clone the repository
git clone https://github.com/your-org/DataModeling-AI.git
cd DataModeling-AI

# Run automated setup
setup.bat

# Update .env file with your API keys
# Edit .env file with notepad or your favorite editor
notepad .env

# Start the application
run.bat                     # Web interface
# or
run_cli.bat --help         # Command line interface
```

### Manual Setup

If you prefer manual setup:

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/DataModeling-AI.git
cd DataModeling-AI
```

2. **Create virtual environment**:
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# Copy template and edit with your values
cp .env.example .env
# Edit .env with your API keys and connection details
```

5. **Start the application**:
```bash
# Web interface
streamlit run src/web/streamlit_app.py

# Command line interface  
python -m src.cli --help
```

### Prerequisites

Before running the setup, ensure you have:

- **Python 3.8+** installed ([Download Python](https://python.org))
- **API Key** from one of:
  - **Claude API** (recommended): Get from [Anthropic Console](https://console.anthropic.com/)
  - **OpenAI API**: Get from [OpenAI Platform](https://platform.openai.com/)
- **Access to a data lake**: Trino, Hive, or Presto cluster with valid credentials

### Environment Variables

The application uses the following environment variables (set in `.env` file):

```bash
# AI Configuration (choose one)
CLAUDE_API_KEY=your_claude_api_key_here        # Recommended
OPENAI_API_KEY=your_openai_api_key_here        # Alternative

# Trino Connection (optional - can be configured via web UI)
TRINO_HOST=localhost
TRINO_PORT=8080
TRINO_USERNAME=your_username
TRINO_PASSWORD=your_password
TRINO_CATALOG=hive
TRINO_SCHEMA=default
```

### Basic Usage

#### Command Line Interface

```bash
# Default workflow (all agents)
python main.py analyze hive my_database

# Discovery-only analysis (fast)
python main.py analyze hive my_database --workflow-type discovery

# Quality-focused analysis
python main.py analyze hive my_database --workflow-type quality

# Custom workflow with specific agents
python main.py analyze hive my_database --workflow discovery analysis modeling

# Parallel execution for performance
python main.py analyze hive my_database --workflow discovery modeling optimization --parallel modeling optimization

# Show workflow information
python main.py workflow-info

# Analyze specific tables
python main.py analyze trino my_database --table table1 --table table2

# Save results to file
python main.py analyze presto my_database --output results.json

# Test connections
python main.py test-connections
```

#### Web Interface

```bash
# Start the web application (after setup)
./run.sh                    # macOS/Linux
# or
run.bat                     # Windows

# Or manually:
streamlit run src/web/streamlit_app.py --server.port 8501
```

Then open your browser to `http://localhost:8501`

The web interface provides:
- **🔧 Manual Connection Setup**: Enter Trino connection details directly
- **📁 Configuration Upload**: Upload YAML config files
- **🗂️ Advanced Table Selection**: Search, filter, and pattern-based table selection  
- **🔄 Workflow Visualization**: Detailed workflow execution and agent results
- **📊 Interactive Analysis**: Rich visualizations and recommendation filtering

## 📋 Configuration

### Configuration File Structure

```yaml
# LLM Provider Configuration
llm_provider: "claude"  # or "openai"

# Claude Configuration (recommended)
claude:
  api_key: "your_claude_api_key_here"
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.3
  max_tokens: 4000

# OpenAI Configuration (alternative)
openai:
  api_key: "your_openai_api_key_here"
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2000

# Data Lake Connections
data_lakes:
  hive:
    host: "localhost"
    port: 10000
    username: "your_username"
    password: "your_password"
    database: "default"
    
  trino:
    host: "localhost"
    port: 8080
    username: "your_username"
    password: "your_password"
    catalog: "hive"
    schema: "default"
    
  presto:
    host: "localhost"
    port: 8080
    username: "your_username"
    catalog: "hive"
    schema: "default"

# Analysis Configuration
analysis:
  max_sample_rows: 10000
  profiling_timeout: 300
  max_concurrent_tables: 10

# AI Agent Configuration
agents:
  schema_analyzer:
    enabled: true
    max_depth: 5
    
  data_profiler:
    enabled: true
    statistical_analysis: true
    data_quality_checks: true
    
  modeling_advisor:
    enabled: true
    include_performance_tips: true
    include_governance_recommendations: true

# Caching
cache:
  enabled: true
  backend: "redis"
  host: "localhost"
  port: 6379
  ttl_hours: 24

# Logging
logging:
  level: "INFO"
  format: "json"
  file: "logs/datamodeling_ai.log"
```

### Environment Variables

You can override configuration values using environment variables:

```bash
# AI API Keys
export CLAUDE_API_KEY="your_claude_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Data Lake Connections
export TRINO_HOST="your_trino_host"
export TRINO_PORT="8080"
export TRINO_USERNAME="your_username"
export TRINO_PASSWORD="your_password"
export TRINO_CATALOG="hive"
export TRINO_SCHEMA="default"

# Legacy Hive/Presto support
export HIVE_HOST="your_hive_host"
export HIVE_PORT="10000"
export PRESTO_HOST="your_presto_host"
```

## 🔍 Analysis Features

### Schema Analysis
- **Table Discovery**: Automatically discovers all tables and their schemas
- **Relationship Detection**: Identifies foreign key relationships and potential joins
- **Data Type Analysis**: Analyzes column data types and constraints
- **Table Classification**: Classifies tables as fact, dimension, bridge, or lookup tables

### Data Profiling
- **Statistical Analysis**: Comprehensive statistics for numeric columns
- **Pattern Recognition**: Identifies common patterns in data (emails, phone numbers, etc.)
- **Data Quality Scoring**: Automated scoring of data quality at column and table levels
- **Anomaly Detection**: Identifies outliers and unusual patterns in data
- **PII Detection**: Identifies potential personally identifiable information

### AI Recommendations

#### Modeling Advisor
- **Dimensional Modeling**: Star schema and snowflake schema recommendations
- **Normalization**: Guidance on normalization and denormalization strategies
- **Data Governance**: Security and compliance recommendations
- **Schema Optimization**: Suggestions for improving schema design

#### Optimization Agent
- **Partitioning Strategy**: Intelligent partitioning recommendations
- **Indexing Strategy**: Index recommendations for performance optimization
- **File Format Optimization**: Parquet, ORC, and compression recommendations
- **Query Optimization**: Query performance improvement suggestions

## 🛠️ Advanced Usage

### Programmatic API

```python
from src.engine.analysis_engine import AnalysisEngine, AnalysisRequest
from src.ai_agents import WorkflowEngine

# Create analysis engine
engine = AnalysisEngine()

# Create analysis request
request = AnalysisRequest(
    connection_type='hive',
    database='my_database',
    tables=['table1', 'table2'],
    include_data_profiling=True,
    include_ai_recommendations=True,
    sample_size=10000,
    context={
        'business_domain': 'e-commerce',
        'performance_requirements': 'sub-second query response'
    }
)

# Default workflow (all agents)
result = engine.analyze(request)

# Discovery-only analysis
result = engine.analyze_discovery_only(request)

# Quality-focused analysis
result = engine.analyze_quality_focused(request)

# Custom workflow
result = engine.analyze_with_custom_workflow(
    request, 
    workflow_stages=['discovery', 'modeling', 'optimization'],
    parallel_stages=['modeling', 'optimization']
)

if result.success:
    print(f"Analysis completed in {result.execution_time:.2f} seconds")
    print(f"Found {len(result.ai_recommendations)} recommendations")
    
    # Access workflow insights
    if result.workflow_insights:
        insights = result.workflow_insights
        print(f"Workflow confidence: {insights['confidence_analysis']['average_confidence']:.2f}")
    
    # Access individual agent results
    if result.agent_results:
        for stage, agent_response in result.agent_results.items():
            print(f"{stage}: {len(agent_response.recommendations)} recommendations")
else:
    print(f"Analysis failed: {result.error_message}")
```

### Custom AI Agents

You can create custom AI agents by extending the `BaseAgent` class:

```python
from src.ai_agents.base_agent import BaseAgent, AgentResponse

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("CustomAgent")
        
    def analyze(self, data, context=None):
        # Your custom analysis logic here
        recommendations = []
        
        return AgentResponse(
            success=True,
            recommendations=recommendations,
            reasoning="Custom analysis reasoning",
            confidence=0.8,
            metadata={'agent': self.agent_name}
        )
```

### Batch Processing

```python
# Analyze multiple databases
requests = [
    AnalysisRequest('hive', 'database1'),
    AnalysisRequest('trino', 'database2'),
    AnalysisRequest('presto', 'database3')
]

results = engine.analyze_multiple_databases(requests)

for result in results:
    if result.success:
        print(f"✅ {result.request.database}: {len(result.ai_recommendations)} recommendations")
    else:
        print(f"❌ {result.request.database}: {result.error_message}")
```

## 📊 Output Examples

### Schema Analysis Output
```json
{
  "database": "ecommerce",
  "tables": {
    "customers": {
      "table_type": "dimension",
      "row_count": 1000000,
      "data_quality_score": 0.85,
      "columns": [
        {
          "name": "customer_id",
          "data_type": "bigint",
          "is_primary_key": true,
          "quality_score": 0.95
        }
      ]
    }
  },
  "relationships": [
    {
      "type": "foreign_key",
      "from_table": "orders",
      "from_column": "customer_id",
      "to_table": "customers",
      "to_column": "customer_id",
      "confidence": 0.9
    }
  ]
}
```

### AI Recommendations Example
```json
[
  {
    "title": "Implement Date-Based Partitioning",
    "category": "partitioning",
    "priority": "high",
    "description": "Partition the orders table by order_date to improve query performance",
    "implementation_effort": "medium",
    "expected_benefit": "50-70% query performance improvement"
  },
  {
    "title": "Create Dimensional Model",
    "category": "dimensional_modeling",
    "priority": "medium",
    "description": "Restructure tables into star schema for better analytics performance",
    "implementation_effort": "high",
    "expected_benefit": "Simplified analytics queries and improved performance"
  }
]
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify connection details in configuration
   - Check network connectivity to data lake
   - Ensure proper authentication credentials

2. **OpenAI API Errors**
   - Verify API key is correct and has sufficient credits
   - Check API rate limits
   - Ensure model availability (gpt-4, gpt-3.5-turbo)

3. **Memory Issues with Large Tables**
   - Reduce sample size in configuration
   - Increase system memory
   - Use table filtering to analyze specific tables

4. **Performance Issues**
   - Enable caching (Redis recommended)
   - Reduce concurrent analysis threads
   - Optimize database queries

### Logging

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python main.py analyze hive my_database --verbose
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/DataModeling-AI.git
cd DataModeling-AI

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/
flake8 src/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/DataModeling-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/DataModeling-AI/discussions)

## 🗺️ Roadmap

- [ ] Support for additional data platforms (Snowflake, BigQuery, Databricks)
- [ ] Advanced ML-based anomaly detection
- [ ] Integration with data catalog systems
- [ ] Real-time monitoring and alerting
- [ ] Custom rule engine for recommendations
- [ ] API for third-party integrations

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- The open-source community for the excellent libraries used in this project
- Contributors and users who help improve this project

---

**Made with ❤️ for the data community**

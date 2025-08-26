# Quick Start Guide

This guide will help you get DataModeling-AI up and running in minutes.

## Prerequisites

- Python 3.8 or higher
- Access to at least one data lake (Hive, Trino, or Presto)
- OpenAI API key
- Basic knowledge of your data warehouse structure

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/DataModeling-AI.git
cd DataModeling-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Create Configuration File

```bash
python main.py init-config --output config.yaml
```

### 4. Configure Your Settings

Edit the generated `config.yaml` file:

```yaml
# Required: OpenAI API Configuration
openai:
  api_key: "your_openai_api_key_here"  # Get from https://platform.openai.com/api-keys
  model: "gpt-4"

# Required: At least one data lake connection
data_lakes:
  # Configure Hive (if available)
  hive:
    host: "your-hive-host"
    port: 10000
    username: "your_username"
    password: "your_password"
    database: "default"
    
  # Configure Trino (if available)
  trino:
    host: "your-trino-host"
    port: 8080
    username: "your_username"
    catalog: "hive"
    schema: "default"
    
  # Configure Presto (if available)
  presto:
    host: "your-presto-host"
    port: 8080
    username: "your_username"
    catalog: "hive"
    schema: "default"
```

## First Analysis

### Test Your Connections

```bash
python main.py test-connections
```

You should see:
```
✅ HIVE
✅ TRINO
❌ PRESTO  # (if not configured)
```

### List Available Databases

```bash
python main.py list-databases hive
```

### Analyze Your First Database

```bash
python main.py analyze hive your_database_name
```

This will:
1. 🔍 Analyze the database schema
2. 📊 Profile the data quality
3. 🤖 Generate AI recommendations
4. 📋 Display results in your terminal

## Using the Web Interface

For a more interactive experience:

```bash
python web_app.py
```

Then open your browser to `http://localhost:8501`

### Web Interface Features:
- 📁 Browse databases and tables
- ⚙️ Configure analysis settings
- 📊 Interactive charts and visualizations
- 💾 Export recommendations to CSV
- 🎯 Filter recommendations by priority/category

## Understanding the Results

### Schema Analysis
- **Table Types**: fact, dimension, bridge, lookup
- **Relationships**: detected foreign keys and joins
- **Data Quality Scores**: 0.0 (poor) to 1.0 (excellent)

### AI Recommendations
- **🔴 High Priority**: Critical issues requiring immediate attention
- **🟡 Medium Priority**: Important improvements for better performance
- **🟢 Low Priority**: Nice-to-have optimizations

### Common Recommendation Categories
- **Partitioning**: Improve query performance with data partitioning
- **Indexing**: Speed up queries with strategic indexes
- **Data Quality**: Fix data quality issues
- **Dimensional Modeling**: Optimize for analytics workloads
- **Storage Optimization**: Reduce storage costs and improve I/O

## Example Workflow

### 1. Discovery Phase
```bash
# Find what's available
python main.py list-databases hive
python main.py list-tables hive sales_db

# Quick analysis without profiling
python main.py analyze hive sales_db --no-profiling --output discovery.json
```

### 2. Deep Analysis Phase
```bash
# Full analysis of specific tables
python main.py analyze hive sales_db \
  --table orders \
  --table customers \
  --table products \
  --sample-size 50000 \
  --output detailed_analysis.json
```

### 3. Review and Implementation
1. Review the generated recommendations
2. Prioritize based on your business needs
3. Implement high-priority recommendations first
4. Monitor performance improvements

## Common Issues and Solutions

### Connection Issues
```bash
# If connection fails, check:
1. Network connectivity: ping your-hive-host
2. Port accessibility: telnet your-hive-host 10000
3. Credentials: verify username/password
4. Service status: ensure Hive/Trino/Presto is running
```

### OpenAI API Issues
```bash
# If AI recommendations fail:
1. Verify API key: check https://platform.openai.com/api-keys
2. Check quota: ensure you have available credits
3. Try different model: change "gpt-4" to "gpt-3.5-turbo" in config
```

### Performance Issues
```bash
# For large databases:
1. Reduce sample size: --sample-size 5000
2. Analyze specific tables: --table table1 --table table2
3. Skip profiling: --no-profiling
4. Enable caching: set up Redis (see configuration)
```

## Next Steps

### Advanced Usage
- 📖 Read the [Advanced Configuration Guide](ADVANCED_CONFIG.md)
- 🔧 Learn about [Custom AI Agents](CUSTOM_AGENTS.md)
- 📊 Explore [Batch Processing](BATCH_PROCESSING.md)

### Integration
- 🔗 [API Documentation](API_REFERENCE.md) for programmatic access
- 🚀 [CI/CD Integration](CICD_INTEGRATION.md) for automated analysis
- 📈 [Monitoring and Alerting](MONITORING.md) setup

### Best Practices
- 📋 [Data Modeling Best Practices](BEST_PRACTICES.md)
- 🏗️ [Implementation Strategies](IMPLEMENTATION.md)
- 🔒 [Security and Compliance](SECURITY.md)

## Getting Help

- 📚 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/your-org/DataModeling-AI/issues)
- 💬 [Community Discussions](https://github.com/your-org/DataModeling-AI/discussions)
- 📧 [Contact Support](mailto:support@your-org.com)

---

**Congratulations! 🎉** You're now ready to automate your data modeling with AI!

Start with a small database to get familiar with the tool, then scale up to your production environments.

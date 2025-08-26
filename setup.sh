#!/bin/bash

# DataModeling-AI Setup Script
# This script sets up a virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up DataModeling-AI Application..."
echo "================================================"

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python $python_version found (required: $required_version+)"
else
    echo "❌ Python $required_version+ is required. Current version: $python_version"
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created successfully!"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# DataModeling-AI Environment Variables
# Copy this file and update with your actual values

# Claude API Key (recommended)
CLAUDE_API_KEY=your_claude_api_key_here

# OpenAI API Key (alternative)
OPENAI_API_KEY=your_openai_api_key_here

# Trino Connection (optional - can be configured via UI)
TRINO_HOST=localhost
TRINO_PORT=8080
TRINO_USERNAME=your_username
TRINO_PASSWORD=your_password
TRINO_CATALOG=hive
TRINO_SCHEMA=default
EOF
    echo "✅ .env template created! Please update it with your actual API keys."
fi

# Create run script
echo "🏃 Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash

# DataModeling-AI Run Script
echo "🚀 Starting DataModeling-AI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists and source it
if [ -f ".env" ]; then
    echo "📝 Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Start the Streamlit application
echo "🌐 Starting web interface on http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "================================================"

streamlit run src/web/streamlit_app.py --server.port 8501 --server.address localhost
EOF

chmod +x run.sh

# Create CLI run script
echo "⚙️  Creating CLI run script..."
cat > run_cli.sh << 'EOF'
#!/bin/bash

# DataModeling-AI CLI Run Script
echo "🔧 Starting DataModeling-AI CLI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists and source it
if [ -f ".env" ]; then
    echo "📝 Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Run CLI with provided arguments
python -m src.cli "$@"
EOF

chmod +x run_cli.sh

echo ""
echo "🎉 Setup completed successfully!"
echo "================================================"
echo ""
echo "📋 Next Steps:"
echo "1. Update the .env file with your API keys:"
echo "   - Get Claude API key from: https://console.anthropic.com/"
echo "   - Or get OpenAI API key from: https://platform.openai.com/"
echo ""
echo "2. Start the application:"
echo "   🌐 Web Interface: ./run.sh"
echo "   ⚙️  Command Line:  ./run_cli.sh --help"
echo ""
echo "3. Access the web interface at: http://localhost:8501"
echo ""
echo "📚 For more information, check the README.md file"
echo "================================================"

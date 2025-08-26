@echo off
REM DataModeling-AI Setup Script for Windows
REM This script sets up a virtual environment and installs all dependencies

echo 🚀 Setting up DataModeling-AI Application...
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org and try again.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% found

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist "venv" (
    echo ⚠️  Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created successfully!

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully!
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

REM Create .env template if it doesn't exist
if not exist ".env" (
    echo 📝 Creating .env template...
    (
        echo # DataModeling-AI Environment Variables
        echo # Copy this file and update with your actual values
        echo.
        echo # Claude API Key ^(recommended^)
        echo CLAUDE_API_KEY=your_claude_api_key_here
        echo.
        echo # OpenAI API Key ^(alternative^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # Trino Connection ^(optional - can be configured via UI^)
        echo TRINO_HOST=localhost
        echo TRINO_PORT=8080
        echo TRINO_USERNAME=your_username
        echo TRINO_PASSWORD=your_password
        echo TRINO_CATALOG=hive
        echo TRINO_SCHEMA=default
    ) > .env
    echo ✅ .env template created! Please update it with your actual API keys.
)

REM Create run script
echo 🏃 Creating run script...
(
    echo @echo off
    echo REM DataModeling-AI Run Script
    echo echo 🚀 Starting DataModeling-AI...
    echo.
    echo REM Check if virtual environment exists
    echo if not exist "venv" ^(
    echo     echo ❌ Virtual environment not found. Please run setup.bat first.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo REM Activate virtual environment
    echo call venv\Scripts\activate.bat
    echo.
    echo REM Load environment variables from .env if it exists
    echo if exist ".env" ^(
    echo     echo 📝 Loading environment variables from .env...
    echo     for /f "usebackq tokens=1,2 delims==" %%%%a in ^(".env"^) do ^(
    echo         if not "%%%%a"=="" if not "%%%%a:~0,1"=="#" set %%%%a=%%%%b
    echo     ^)
    echo ^)
    echo.
    echo REM Start the Streamlit application
    echo echo 🌐 Starting web interface on http://localhost:8501
    echo echo Press Ctrl+C to stop the application
    echo echo ================================================
    echo.
    echo streamlit run src/web/streamlit_app.py --server.port 8501 --server.address localhost
) > run.bat

REM Create CLI run script
echo ⚙️  Creating CLI run script...
(
    echo @echo off
    echo REM DataModeling-AI CLI Run Script
    echo echo 🔧 Starting DataModeling-AI CLI...
    echo.
    echo REM Check if virtual environment exists
    echo if not exist "venv" ^(
    echo     echo ❌ Virtual environment not found. Please run setup.bat first.
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo REM Activate virtual environment
    echo call venv\Scripts\activate.bat
    echo.
    echo REM Load environment variables from .env if it exists
    echo if exist ".env" ^(
    echo     echo 📝 Loading environment variables from .env...
    echo     for /f "usebackq tokens=1,2 delims==" %%%%a in ^(".env"^) do ^(
    echo         if not "%%%%a"=="" if not "%%%%a:~0,1"=="#" set %%%%a=%%%%b
    echo     ^)
    echo ^)
    echo.
    echo REM Run CLI with provided arguments
    echo python -m src.cli %%*
) > run_cli.bat

echo.
echo 🎉 Setup completed successfully!
echo ================================================
echo.
echo 📋 Next Steps:
echo 1. Update the .env file with your API keys:
echo    - Get Claude API key from: https://console.anthropic.com/
echo    - Or get OpenAI API key from: https://platform.openai.com/
echo.
echo 2. Start the application:
echo    🌐 Web Interface: run.bat
echo    ⚙️  Command Line:  run_cli.bat --help
echo.
echo 3. Access the web interface at: http://localhost:8501
echo.
echo 📚 For more information, check the README.md file
echo ================================================
pause

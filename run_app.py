#!/usr/bin/env python3
"""Simple script to run the DataModeling-AI web application."""

import sys
import os
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Run the Streamlit application."""
    print("🚀 Starting DataModeling-AI Web Application...")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    
    # Load .env file if it exists
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print("📝 Loading environment variables from .env...")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env[key] = value
    
    # Run Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'src/web/streamlit_app_simple.py',
        '--server.port', '8501',
        '--server.address', 'localhost'
    ]
    
    print("🌐 Starting web interface on http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")

if __name__ == "__main__":
    main()

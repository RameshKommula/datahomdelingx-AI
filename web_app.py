#!/usr/bin/env python3
"""Web application entry point."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import and run the Streamlit app
if __name__ == "__main__":
    import streamlit.cli
    sys.argv = ["streamlit", "run", "src/web/streamlit_app.py"]
    streamlit.cli.main()

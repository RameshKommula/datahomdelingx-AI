"""
Simplified connection management without encryption.
Stores connections and API keys in local JSON files.
"""

import json
import os
import streamlit as st
from pathlib import Path

# Local storage paths
CONNECTIONS_FILE = "saved_connections.json"
API_KEYS_FILE = "saved_api_keys.json"

def load_connections():
    """Load saved connections from local file."""
    if os.path.exists(CONNECTIONS_FILE):
        try:
            with open(CONNECTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_connections(connections):
    """Save connections to local file."""
    try:
        with open(CONNECTIONS_FILE, 'w') as f:
            json.dump(connections, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save connections: {e}")
        return False

def load_api_keys():
    """Load saved API keys from local file."""
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_api_keys(api_keys):
    """Save API keys to local file."""
    try:
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(api_keys, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save API keys: {e}")
        return False

def test_trino_connection(host, port, username, password, catalog, schema):
    """Test Trino connection."""
    try:
        import trino
        conn = trino.dbapi.connect(
            host=host,
            port=int(port),
            user=username,
            catalog=catalog,
            schema=schema,
            http_scheme='https' if port == 443 else 'http'
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return True, "Connection successful!"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def test_api_connection(provider, api_key):
    """Test API connection."""
    try:
        if provider.lower() == "openai":
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True, "OpenAI API connection successful!"
        elif provider.lower() == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True, "Claude API connection successful!"
    except Exception as e:
        return False, f"API connection failed: {str(e)}"

def render_connection_management():
    """Render the simplified connection management UI."""
    st.header("🔧 Connection Management")
    
    # Load existing connections and API keys
    connections = load_connections()
    api_keys = load_api_keys()
    
    # Database Connections Section
    st.subheader("🗄️ Database Connections")
    
    # Show existing connections
    if connections:
        st.write("**Saved Connections:**")
        for conn_name, conn_data in connections.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📊 **{conn_name}** ({conn_data['type']})")
                st.caption(f"Host: {conn_data['host']}:{conn_data['port']}")
            with col2:
                if st.button(f"Use", key=f"use_{conn_name}"):
                    st.session_state.selected_connection = conn_data
                    st.success(f"Using connection: {conn_name}")
            with col3:
                if st.button(f"Delete", key=f"del_{conn_name}"):
                    del connections[conn_name]
                    save_connections(connections)
                    st.rerun()
        st.divider()
    
    # New Connection Form
    with st.form("new_connection"):
        st.write("**Create New Connection:**")
        
        conn_name = st.text_input("Connection Name", placeholder="My Trino Connection")
        conn_type = st.selectbox("Connection Type", ["Trino", "Hive", "Presto"])
        
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", placeholder="localhost")
            username = st.text_input("Username", placeholder="admin")
            catalog = st.text_input("Catalog", placeholder="hive")
        with col2:
            port = st.text_input("Port", placeholder="8080")
            password = st.text_input("Password", type="password")
            schema = st.text_input("Schema", placeholder="default")
        
        col1, col2 = st.columns(2)
        with col1:
            test_conn = st.form_submit_button("🔍 Test Connection", type="secondary")
        with col2:
            save_conn = st.form_submit_button("💾 Save Connection", type="primary")
        
        if test_conn:
            if conn_type == "Trino" and all([host, port, username, catalog, schema]):
                success, message = test_trino_connection(host, port, username, password, catalog, schema)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please fill in all required fields")
        
        if save_conn:
            if conn_name and host and port and username:
                new_connection = {
                    'type': conn_type,
                    'host': host,
                    'port': port,
                    'username': username,
                    'password': password,
                    'catalog': catalog,
                    'schema': schema
                }
                connections[conn_name] = new_connection
                if save_connections(connections):
                    st.success(f"Connection '{conn_name}' saved successfully!")
                    st.rerun()
            else:
                st.error("Please provide connection name and required details")
    
    st.divider()
    
    # AI API Keys Section
    st.subheader("🤖 AI API Keys")
    
    # Show existing API keys
    if api_keys:
        st.write("**Saved API Keys:**")
        for key_name, key_data in api_keys.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"🔑 **{key_name}** ({key_data['provider']})")
                st.caption(f"Key: {key_data['api_key'][:10]}...")
            with col2:
                if st.button(f"Use", key=f"use_api_{key_name}"):
                    st.session_state.selected_api_key = key_data
                    st.success(f"Using API key: {key_name}")
            with col3:
                if st.button(f"Delete", key=f"del_api_{key_name}"):
                    del api_keys[key_name]
                    save_api_keys(api_keys)
                    st.rerun()
        st.divider()
    
    # New API Key Form
    with st.form("new_api_key"):
        st.write("**Add New API Key:**")
        
        key_name = st.text_input("Key Name", placeholder="My OpenAI Key")
        provider = st.selectbox("AI Provider", ["OpenAI", "Claude"])
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        
        col1, col2 = st.columns(2)
        with col1:
            test_api = st.form_submit_button("🔍 Test API", type="secondary")
        with col2:
            save_api = st.form_submit_button("💾 Save API Key", type="primary")
        
        if test_api:
            if api_key:
                success, message = test_api_connection(provider, api_key)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please provide API key")
        
        if save_api:
            if key_name and api_key:
                new_api_key = {
                    'provider': provider,
                    'api_key': api_key
                }
                api_keys[key_name] = new_api_key
                if save_api_keys(api_keys):
                    st.success(f"API key '{key_name}' saved successfully!")
                    st.rerun()
            else:
                st.error("Please provide key name and API key")

if __name__ == "__main__":
    st.set_page_config(page_title="Simple Connection Manager", layout="wide")
    render_connection_management()

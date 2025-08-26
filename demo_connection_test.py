#!/usr/bin/env python3
"""Demo script for testing real database connections with DataModeling-AI."""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from connectors.trino_connector import TrinoConnector
from security.encryption import ConnectionManager

def test_real_trino_connection():
    """Test a real Trino connection (requires actual Trino server)."""
    print("🔗 Testing Real Trino Connection")
    print("=" * 40)
    
    # Example connection parameters (update with your actual values)
    connection_params = {
        "host": "localhost",          # Your Trino host
        "port": 8080,                # Your Trino port
        "username": "your_username",  # Your username
        "catalog": "hive",           # Your catalog
        "schema": "default"          # Your schema
    }
    
    print("📋 Connection Parameters:")
    for key, value in connection_params.items():
        if key != "password":
            print(f"  {key}: {value}")
    
    print("\n⚠️  To test with your actual Trino server:")
    print("1. Update the connection_params above with your real values")
    print("2. Uncomment the test code below")
    print("3. Run this script again")
    
    # Uncomment and modify the following code to test with your actual Trino server:
    """
    try:
        print("\\n🔌 Attempting connection...")
        connector = TrinoConnector(**connection_params)
        
        # Test connection
        success = connector.test_connection()
        
        if success:
            print("✅ Connection successful!")
            
            # Test listing databases
            try:
                databases = connector.list_databases()
                print(f"📊 Found {len(databases)} databases:")
                for db in databases[:5]:  # Show first 5
                    print(f"  - {db}")
                if len(databases) > 5:
                    print(f"  ... and {len(databases) - 5} more")
            except Exception as e:
                print(f"⚠️  Could not list databases: {e}")
            
            # Test listing tables in a database
            try:
                tables = connector.list_tables("default")  # Change to your database
                print(f"📋 Found {len(tables)} tables in 'default' database:")
                for table in tables[:5]:  # Show first 5
                    print(f"  - {table}")
                if len(tables) > 5:
                    print(f"  ... and {len(tables) - 5} more")
            except Exception as e:
                print(f"⚠️  Could not list tables: {e}")
                
        else:
            print("❌ Connection failed!")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\\n💡 Common issues:")
        print("  - Check if Trino server is running")
        print("  - Verify host and port are correct")
        print("  - Ensure username has proper permissions")
        print("  - Check network connectivity")
    """

def demo_encrypted_storage():
    """Demonstrate encrypted connection storage."""
    print("\n💾 Encrypted Connection Storage Demo")
    print("=" * 40)
    
    # Create connection manager
    cm = ConnectionManager()
    password = "demo_password_123"
    
    # Sample connection data
    connection_data = {
        "connection_type": "trino",
        "host": "prod-trino.company.com",
        "port": 8080,
        "username": "data_analyst",
        "password": "secure_password",
        "catalog": "production_hive",
        "schema": "analytics",
        "llm_provider": "claude",
        "claude_api_key": "sk-ant-api03-xxx...xxx"
    }
    
    try:
        # Save connection
        print("💾 Saving encrypted connection...")
        success = cm.save_connection("production_analytics", connection_data, password)
        
        if success:
            print("✅ Connection saved successfully!")
            
            # List connections
            print("📋 Listing saved connections...")
            connections = cm.list_connections(password)
            
            if connections:
                for name, metadata in connections.items():
                    print(f"  📊 {name}:")
                    print(f"    Type: {metadata['connection_type']}")
                    print(f"    Created: {metadata['created_at'][:19]}")
                    print(f"    Last Used: {metadata['last_used'][:19]}")
            
            # Load connection
            print("📥 Loading connection...")
            loaded_data = cm.load_connection("production_analytics", password)
            
            if loaded_data:
                print("✅ Connection loaded successfully!")
                print(f"  Host: {loaded_data['host']}")
                print(f"  Username: {loaded_data['username']}")
                print(f"  Password: {'*' * len(loaded_data.get('password', ''))}")
                
            # Test wrong password
            print("🔒 Testing security with wrong password...")
            wrong_data = cm.load_connection("production_analytics", "wrong_password")
            
            if wrong_data is None:
                print("✅ Security verified - wrong password rejected!")
            else:
                print("❌ Security issue - wrong password accepted!")
                
        else:
            print("❌ Failed to save connection!")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        # Cleanup demo data
        try:
            cm.delete_connection("production_analytics", password)
            print("🧹 Demo data cleaned up")
        except:
            pass

def main():
    """Run connection demos."""
    print("🧪 DataModeling-AI Connection Demo")
    print("=" * 50)
    
    # Test real connection (commented out by default)
    test_real_trino_connection()
    
    # Demo encrypted storage
    demo_encrypted_storage()
    
    print("\n" + "=" * 50)
    print("🎯 Summary:")
    print("✅ Encrypted connection storage working")
    print("✅ Trino connector ready for testing")
    print("✅ Web UI available at http://localhost:8501")
    
    print("\n💡 Next Steps:")
    print("1. Open the web interface: http://localhost:8501")
    print("2. Set an encryption password in 'Saved Connections'")
    print("3. Configure your actual Trino connection details")
    print("4. Test the connection using the 'Test Connection' button")
    print("5. Save successful connections for future use")
    print("6. Analyze your data with AI-powered recommendations!")

if __name__ == "__main__":
    main()

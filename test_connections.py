#!/usr/bin/env python3
"""Test script for connection management and encryption functionality."""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from security.encryption import ConnectionManager, SecureStorage
import json

def test_encryption():
    """Test basic encryption functionality."""
    print("🔐 Testing Encryption Functionality...")
    
    # Test data
    test_data = {
        "test_connection": {
            "host": "localhost",
            "port": 8080,
            "username": "test_user",
            "password": "secret_password",
            "api_key": "sk-test-key-12345"
        }
    }
    
    # Test password
    password = "test_encryption_password"
    
    # Create secure storage instance
    storage = SecureStorage("test_connections.enc")
    
    try:
        # Test encryption
        print("  ✅ Encrypting test data...")
        success = storage.encrypt_data(test_data, password)
        if not success:
            print("  ❌ Encryption failed!")
            return False
        
        # Test decryption
        print("  ✅ Decrypting test data...")
        decrypted_data = storage.decrypt_data(password)
        if decrypted_data is None:
            print("  ❌ Decryption failed!")
            return False
        
        # Verify data integrity
        if decrypted_data == test_data:
            print("  ✅ Data integrity verified!")
        else:
            print("  ❌ Data integrity check failed!")
            return False
        
        # Test wrong password
        print("  ✅ Testing wrong password...")
        wrong_decrypted = storage.decrypt_data("wrong_password")
        if wrong_decrypted is None:
            print("  ✅ Wrong password correctly rejected!")
        else:
            print("  ❌ Wrong password was accepted!")
            return False
        
        # Cleanup
        storage.delete_storage()
        print("  ✅ Encryption tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Encryption test failed: {e}")
        return False

def test_connection_manager():
    """Test connection manager functionality."""
    print("\n📋 Testing Connection Manager...")
    
    # Create connection manager
    cm = ConnectionManager("test_connections")
    password = "test_manager_password"
    
    try:
        # Test saving connections
        print("  ✅ Testing save connection...")
        connection_data = {
            "connection_type": "trino",
            "host": "test-trino.example.com",
            "port": 8080,
            "username": "test_user",
            "password": "test_password",
            "catalog": "hive",
            "schema": "default",
            "llm_provider": "claude",
            "claude_api_key": "sk-test-claude-key"
        }
        
        success = cm.save_connection("test_production", connection_data, password)
        if not success:
            print("  ❌ Save connection failed!")
            return False
        
        # Test listing connections
        print("  ✅ Testing list connections...")
        connections = cm.list_connections(password)
        if connections is None or "test_production" not in connections:
            print("  ❌ List connections failed!")
            return False
        
        print(f"  ✅ Found connection: {connections['test_production']['connection_type']}")
        
        # Test loading connection
        print("  ✅ Testing load connection...")
        loaded_data = cm.load_connection("test_production", password)
        if loaded_data is None:
            print("  ❌ Load connection failed!")
            return False
        
        # Verify loaded data
        if loaded_data["host"] == connection_data["host"]:
            print("  ✅ Connection data loaded correctly!")
        else:
            print("  ❌ Connection data mismatch!")
            return False
        
        # Test saving another connection
        print("  ✅ Testing multiple connections...")
        dev_connection = {
            "connection_type": "trino",
            "host": "dev-trino.example.com",
            "port": 8080,
            "username": "dev_user",
            "catalog": "test_catalog"
        }
        
        cm.save_connection("test_development", dev_connection, password)
        connections = cm.list_connections(password)
        
        if len(connections) == 2:
            print("  ✅ Multiple connections saved successfully!")
        else:
            print(f"  ❌ Expected 2 connections, found {len(connections)}")
            return False
        
        # Test deleting connection
        print("  ✅ Testing delete connection...")
        success = cm.delete_connection("test_development", password)
        if success:
            connections = cm.list_connections(password)
            if len(connections) == 1:
                print("  ✅ Connection deleted successfully!")
            else:
                print("  ❌ Connection not deleted properly!")
                return False
        else:
            print("  ❌ Delete connection failed!")
            return False
        
        # Cleanup
        import shutil
        if os.path.exists("test_connections"):
            shutil.rmtree("test_connections")
        
        print("  ✅ Connection manager tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Connection manager test failed: {e}")
        return False

def test_trino_connector():
    """Test Trino connector (without actual connection)."""
    print("\n🔗 Testing Trino Connector Import...")
    
    try:
        from connectors.trino_connector import TrinoConnector
        print("  ✅ TrinoConnector imported successfully!")
        
        # Test connector creation
        connector = TrinoConnector(
            host="localhost",
            port=8080,
            username="test_user",
            catalog="hive",
            schema="default"
        )
        print("  ✅ TrinoConnector instance created!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Trino connector test failed: {e}")
        return False

def main():
    """Run all connection tests."""
    print("🧪 DataModeling-AI Connection Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_encryption():
        tests_passed += 1
    
    if test_connection_manager():
        tests_passed += 1
    
    if test_trino_connector():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Connection management is working correctly.")
        print("\n📋 Features Ready:")
        print("  ✅ Secure encryption of connection data")
        print("  ✅ Save/load connection configurations")
        print("  ✅ Multiple connection management")
        print("  ✅ Password-protected storage")
        print("  ✅ Trino connector ready")
        
        print("\n🌐 Your app is running at: http://localhost:8501")
        print("💡 Try the new features:")
        print("  1. Set an encryption password in 'Saved Connections'")
        print("  2. Configure a manual connection")
        print("  3. Save the connection for later use")
        print("  4. Load saved connections quickly")
        
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

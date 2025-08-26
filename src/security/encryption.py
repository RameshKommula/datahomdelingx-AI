"""Secure encryption utilities for sensitive data like connection credentials."""

import os
import json
import base64
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger
import getpass


class SecureStorage:
    """Handles secure encryption and decryption of sensitive data."""
    
    def __init__(self, storage_file: str = "connections.enc"):
        """
        Initialize secure storage.
        
        Args:
            storage_file: Path to encrypted storage file
        """
        self.storage_file = storage_file
        self._key = None
        self._cipher_suite = None
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def _get_cipher_suite(self, password: str, salt: Optional[bytes] = None) -> tuple[Fernet, bytes]:
        """Get cipher suite for encryption/decryption."""
        if salt is None:
            salt = os.urandom(16)
        
        key = self._derive_key(password, salt)
        cipher_suite = Fernet(key)
        return cipher_suite, salt
    
    def encrypt_data(self, data: Dict[str, Any], password: str) -> bool:
        """
        Encrypt and save data to storage file.
        
        Args:
            data: Dictionary containing data to encrypt
            password: Password for encryption
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, indent=2)
            
            # Get cipher suite with new salt
            cipher_suite, salt = self._get_cipher_suite(password)
            
            # Encrypt the data
            encrypted_data = cipher_suite.encrypt(json_data.encode())
            
            # Create storage structure
            storage_data = {
                "salt": base64.b64encode(salt).decode(),
                "encrypted_data": base64.b64encode(encrypted_data).decode(),
                "version": "1.0"
            }
            
            # Save to file
            os.makedirs(os.path.dirname(self.storage_file) if os.path.dirname(self.storage_file) else ".", exist_ok=True)
            with open(self.storage_file, 'w') as f:
                json.dump(storage_data, f, indent=2)
            
            logger.info(f"Data encrypted and saved to {self.storage_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            return False
    
    def decrypt_data(self, password: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt and load data from storage file.
        
        Args:
            password: Password for decryption
            
        Returns:
            Decrypted data dictionary or None if failed
        """
        try:
            # Check if storage file exists
            if not os.path.exists(self.storage_file):
                logger.warning(f"Storage file {self.storage_file} does not exist")
                return None
            
            # Load storage data
            with open(self.storage_file, 'r') as f:
                storage_data = json.load(f)
            
            # Extract salt and encrypted data
            salt = base64.b64decode(storage_data["salt"])
            encrypted_data = base64.b64decode(storage_data["encrypted_data"])
            
            # Get cipher suite with stored salt
            cipher_suite, _ = self._get_cipher_suite(password, salt)
            
            # Decrypt the data
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            
            # Parse JSON
            data = json.loads(decrypted_data.decode())
            
            logger.info(f"Data decrypted successfully from {self.storage_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None
    
    def storage_exists(self) -> bool:
        """Check if encrypted storage file exists."""
        return os.path.exists(self.storage_file)
    
    def delete_storage(self) -> bool:
        """Delete the encrypted storage file."""
        try:
            if os.path.exists(self.storage_file):
                os.remove(self.storage_file)
                logger.info(f"Storage file {self.storage_file} deleted")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete storage file: {e}")
            return False


class ConnectionManager:
    """Manages encrypted storage of database connection configurations."""
    
    def __init__(self, storage_dir: str = "~/.datamodeling_ai"):
        """
        Initialize connection manager.
        
        Args:
            storage_dir: Directory to store encrypted connection files
        """
        self.storage_dir = os.path.expanduser(storage_dir)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.connections_file = os.path.join(self.storage_dir, "connections.enc")
        self.secure_storage = SecureStorage(self.connections_file)
    
    def save_connection(self, connection_name: str, connection_data: Dict[str, Any], 
                       password: str) -> bool:
        """
        Save a connection configuration securely.
        
        Args:
            connection_name: Name for the connection
            connection_data: Connection configuration data
            password: Password for encryption
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing connections
            existing_connections = self.secure_storage.decrypt_data(password) or {}
            
            # Add new connection
            existing_connections[connection_name] = {
                "connection_data": connection_data,
                "created_at": str(pd.Timestamp.now()),
                "last_used": str(pd.Timestamp.now())
            }
            
            # Save back
            success = self.secure_storage.encrypt_data(existing_connections, password)
            
            if success:
                logger.info(f"Connection '{connection_name}' saved successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save connection '{connection_name}': {e}")
            return False
    
    def load_connection(self, connection_name: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific connection configuration.
        
        Args:
            connection_name: Name of the connection to load
            password: Password for decryption
            
        Returns:
            Connection data or None if not found/failed
        """
        try:
            connections = self.secure_storage.decrypt_data(password)
            
            if connections and connection_name in connections:
                # Update last used timestamp
                connections[connection_name]["last_used"] = str(pd.Timestamp.now())
                self.secure_storage.encrypt_data(connections, password)
                
                return connections[connection_name]["connection_data"]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load connection '{connection_name}': {e}")
            return None
    
    def list_connections(self, password: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        List all saved connection configurations.
        
        Args:
            password: Password for decryption
            
        Returns:
            Dictionary of connection metadata or None if failed
        """
        try:
            connections = self.secure_storage.decrypt_data(password)
            
            if not connections:
                return {}
            
            # Return metadata only (without sensitive connection data)
            metadata = {}
            for name, data in connections.items():
                metadata[name] = {
                    "created_at": data.get("created_at"),
                    "last_used": data.get("last_used"),
                    "connection_type": data.get("connection_data", {}).get("connection_type", "unknown")
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to list connections: {e}")
            return None
    
    def delete_connection(self, connection_name: str, password: str) -> bool:
        """
        Delete a specific connection configuration.
        
        Args:
            connection_name: Name of the connection to delete
            password: Password for decryption
            
        Returns:
            True if successful, False otherwise
        """
        try:
            connections = self.secure_storage.decrypt_data(password)
            
            if connections and connection_name in connections:
                del connections[connection_name]
                success = self.secure_storage.encrypt_data(connections, password)
                
                if success:
                    logger.info(f"Connection '{connection_name}' deleted successfully")
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete connection '{connection_name}': {e}")
            return False
    
    def update_connection(self, connection_name: str, connection_data: Dict[str, Any],
                         password: str) -> bool:
        """
        Update an existing connection configuration.
        
        Args:
            connection_name: Name of the connection to update
            connection_data: New connection configuration data
            password: Password for encryption
            
        Returns:
            True if successful, False otherwise
        """
        try:
            connections = self.secure_storage.decrypt_data(password)
            
            if connections and connection_name in connections:
                # Preserve creation timestamp
                created_at = connections[connection_name].get("created_at")
                
                connections[connection_name] = {
                    "connection_data": connection_data,
                    "created_at": created_at,
                    "last_used": str(pd.Timestamp.now())
                }
                
                success = self.secure_storage.encrypt_data(connections, password)
                
                if success:
                    logger.info(f"Connection '{connection_name}' updated successfully")
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update connection '{connection_name}': {e}")
            return False
    
    def storage_exists(self) -> bool:
        """Check if any connection storage exists."""
        return self.secure_storage.storage_exists()
    
    def change_password(self, old_password: str, new_password: str) -> bool:
        """
        Change the encryption password for stored connections.
        
        Args:
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Decrypt with old password
            connections = self.secure_storage.decrypt_data(old_password)
            
            if connections is None:
                return False
            
            # Re-encrypt with new password
            success = self.secure_storage.encrypt_data(connections, new_password)
            
            if success:
                logger.info("Password changed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to change password: {e}")
            return False


# Add pandas import
import pandas as pd

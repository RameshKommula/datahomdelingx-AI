"""Relationship detector for identifying relationships between tables."""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from loguru import logger


class RelationshipDetector:
    """Detects relationships between database tables."""
    
    def __init__(self):
        """Initialize the relationship detector."""
        self.relationships = []
    
    def detect_relationships(self, tables_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect relationships between tables based on column names and data patterns.
        
        Args:
            tables_data: Dictionary containing table metadata and sample data
            
        Returns:
            List of detected relationships
        """
        logger.info(f"Detecting relationships across {len(tables_data)} tables")
        
        relationships = []
        table_names = list(tables_data.keys())
        
        # Compare each pair of tables
        for i, table1 in enumerate(table_names):
            for j, table2 in enumerate(table_names):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                table1_data = tables_data[table1]
                table2_data = tables_data[table2]
                
                # Detect potential foreign key relationships
                fk_relationships = self._detect_foreign_keys(
                    table1, table1_data, table2, table2_data
                )
                relationships.extend(fk_relationships)
        
        logger.info(f"Detected {len(relationships)} potential relationships")
        return relationships
    
    def _detect_foreign_keys(self, table1: str, table1_data: Dict[str, Any], 
                           table2: str, table2_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect foreign key relationships between two tables."""
        relationships = []
        
        # Get column information
        table1_columns = table1_data.get('schema', [])
        table2_columns = table2_data.get('schema', [])
        
        # Look for potential FK relationships based on naming patterns
        for col1 in table1_columns:
            col1_name = col1.get('column_name', '').lower()
            col1_type = col1.get('data_type', '')
            
            for col2 in table2_columns:
                col2_name = col2.get('column_name', '').lower()
                col2_type = col2.get('data_type', '')
                
                # Check if column names suggest a relationship
                if self._is_potential_relationship(col1_name, col2_name, table1, table2):
                    # Check if data types are compatible
                    if self._are_types_compatible(col1_type, col2_type):
                        confidence = self._calculate_relationship_confidence(
                            col1_name, col2_name, table1, table2
                        )
                        
                        relationship = {
                            'type': 'foreign_key',
                            'from_table': table1,
                            'from_column': col1.get('column_name'),
                            'to_table': table2,
                            'to_column': col2.get('column_name'),
                            'confidence': confidence,
                            'reasoning': f"Column names suggest relationship: {col1_name} -> {col2_name}"
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _is_potential_relationship(self, col1_name: str, col2_name: str, 
                                 table1: str, table2: str) -> bool:
        """Check if column names suggest a potential relationship."""
        # Direct match
        if col1_name == col2_name:
            return True
        
        # FK naming patterns
        table2_singular = table2.rstrip('s').lower()
        table1_singular = table1.rstrip('s').lower()
        
        # Check if col1 references table2 (e.g., user_id -> users.id)
        if col1_name == f"{table2_singular}_id" and col2_name == "id":
            return True
        
        # Check if col2 references table1 (e.g., id -> orders.user_id)
        if col2_name == f"{table1_singular}_id" and col1_name == "id":
            return True
        
        # Common FK patterns
        fk_patterns = [
            f"{table2_singular}id",
            f"{table2_singular}_id",
            f"{table2}_id",
        ]
        
        if col1_name in fk_patterns and col2_name == "id":
            return True
        
        if col2_name in [f"{table1_singular}_id", f"{table1}_id"] and col1_name == "id":
            return True
        
        return False
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two data types are compatible for relationships."""
        # Normalize type names
        type1 = type1.lower().strip()
        type2 = type2.lower().strip()
        
        # Exact match
        if type1 == type2:
            return True
        
        # Integer types
        integer_types = ['int', 'integer', 'bigint', 'smallint', 'tinyint']
        if any(t in type1 for t in integer_types) and any(t in type2 for t in integer_types):
            return True
        
        # String types
        string_types = ['varchar', 'char', 'text', 'string']
        if any(t in type1 for t in string_types) and any(t in type2 for t in string_types):
            return True
        
        return False
    
    def _calculate_relationship_confidence(self, col1_name: str, col2_name: str,
                                         table1: str, table2: str) -> float:
        """Calculate confidence score for a potential relationship."""
        confidence = 0.5  # Base confidence
        
        # Exact column name match
        if col1_name == col2_name:
            confidence += 0.3
        
        # Standard FK patterns
        table2_singular = table2.rstrip('s').lower()
        if col1_name == f"{table2_singular}_id" and col2_name == "id":
            confidence += 0.4
        
        # ID column involvement
        if "id" in col1_name.lower() or "id" in col2_name.lower():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def analyze_table_relationships(self, table_name: str, 
                                  relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships for a specific table."""
        incoming = []
        outgoing = []
        
        for rel in relationships:
            if rel['to_table'] == table_name:
                incoming.append(rel)
            elif rel['from_table'] == table_name:
                outgoing.append(rel)
        
        return {
            'table_name': table_name,
            'incoming_relationships': incoming,
            'outgoing_relationships': outgoing,
            'total_relationships': len(incoming) + len(outgoing),
            'is_hub': len(incoming) + len(outgoing) > 3,  # Tables with many relationships
            'is_lookup': len(incoming) > len(outgoing) and len(incoming) > 1
        }

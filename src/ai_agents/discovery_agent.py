"""Discovery Agent - Discovers and catalogs data assets and their basic characteristics."""

from typing import Dict, List, Any, Optional
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class DiscoveryAgent(BaseAgent):
    """AI agent responsible for discovering and cataloging data assets."""
    
    def __init__(self):
        super().__init__("DiscoveryAgent")
        
        self.system_prompt = """
You are a Data Discovery Agent, an expert in data asset discovery and cataloging. 
Your primary role is to explore and understand data landscapes, identifying:

1. Data Assets: Tables, views, and other data objects
2. Data Lineage: Understanding data flow and dependencies
3. Business Context: Inferring business purpose and usage patterns
4. Data Domains: Categorizing data by business domain
5. Access Patterns: Understanding how data is typically accessed

Your analysis should focus on:
- Cataloging all discoverable data assets
- Identifying data domains and business contexts
- Understanding data relationships and dependencies
- Assessing data accessibility and usage patterns
- Providing initial classification of data importance

Provide clear, structured findings that will help other agents understand 
the data landscape for further analysis.
"""
    
    def analyze(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Discover and catalog data assets."""
        try:
            logger.info("DiscoveryAgent starting data asset discovery")
            
            # Prepare discovery prompt
            user_prompt = self._create_discovery_prompt(data, context)
            
            # Get discovery insights from LLM
            llm_response = self._make_llm_request(self.system_prompt, user_prompt)
            
            # Parse discovery findings
            discoveries = self._parse_discovery_findings(llm_response, data)
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_discovery_confidence(data, discoveries)
            
            return AgentResponse(
                success=True,
                recommendations=discoveries,
                reasoning=llm_response,
                confidence=confidence,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'data_discovery',
                    'assets_discovered': len(data.get('tables', {})),
                    'domains_identified': len(set(d.get('domain', 'unknown') for d in discoveries)),
                    'discovery_scope': 'full_catalog'
                }
            )
            
        except Exception as e:
            logger.error(f"DiscoveryAgent analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _create_discovery_prompt(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Create discovery analysis prompt."""
        prompt = "Please analyze the following data landscape for discovery and cataloging:\n\n"
        
        # Add database overview
        if 'database' in data:
            prompt += f"DATABASE: {data['database']}\n"
        
        # Add table information for discovery
        if 'tables' in data:
            prompt += "DISCOVERED DATA ASSETS:\n"
            tables = data['tables']
            
            # Handle both list and dict formats
            if isinstance(tables, list):
                for table_name in tables:
                    prompt += f"\nTable: {table_name}\n"
                    prompt += f"- Type: table\n"
                    prompt += f"- Status: discovered\n"
            elif isinstance(tables, dict):
                for table_name, table_info in tables.items():
                    prompt += f"\nTable: {table_name}\n"
                    prompt += f"- Rows: {table_info.get('row_count', 'unknown')}\n"
                    prompt += f"- Columns: {len(table_info.get('columns', []))}\n"
                    prompt += f"- Type: {table_info.get('table_type', 'unknown')}\n"
                
                # Add column information for context understanding
                columns = table_info.get('columns', [])[:10]  # First 10 columns
                if columns:
                    prompt += "- Key Columns: "
                    col_names = [col.get('name', '') for col in columns]
                    prompt += ", ".join(col_names) + "\n"
        
        # Add relationship information
        if 'relationships' in data:
            relationships = data['relationships']
            if relationships:
                prompt += f"\nDETECTED RELATIONSHIPS: {len(relationships)} relationships found\n"
                for rel in relationships[:5]:  # Show first 5 relationships
                    prompt += f"- {rel.get('from_table', '')}.{rel.get('from_column', '')} -> "
                    prompt += f"{rel.get('to_table', '')}.{rel.get('to_column', '')}\n"
        
        # Add business context if provided
        if context:
            prompt += "\nBUSINESS CONTEXT:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += """
Please provide a comprehensive discovery analysis including:

1. **Data Asset Catalog**:
   - Classify each table by business domain (e.g., customer, product, sales, finance)
   - Identify core business entities vs supporting/reference data
   - Assess data asset importance (critical, important, supporting)

2. **Business Context Understanding**:
   - Infer business processes represented in the data
   - Identify potential data marts or subject areas
   - Understand data granularity and aggregation levels

3. **Data Relationship Mapping**:
   - Identify hub tables (highly connected entities)
   - Map data flow patterns and dependencies
   - Detect potential master data entities

4. **Access and Usage Patterns**:
   - Identify frequently accessed vs rarely used tables
   - Suggest typical query patterns based on structure
   - Assess data freshness and update patterns

5. **Data Domain Classification**:
   - Group tables into logical business domains
   - Identify cross-domain dependencies
   - Suggest data governance boundaries

For each finding, provide:
- Clear business justification
- Confidence level (high/medium/low)
- Recommended next steps for detailed analysis

Format your response with clear sections and actionable insights.
"""
        
        return prompt
    
    def _parse_discovery_findings(self, llm_response: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM response into structured discovery findings."""
        findings = []
        
        # Parse the response into sections
        sections = self._split_into_sections(llm_response)
        
        for section_title, section_content in sections.items():
            category = self._categorize_discovery_finding(section_title)
            section_findings = self._parse_section_findings(section_content, category)
            findings.extend(section_findings)
        
        # Enhance findings with metadata from original data
        self._enhance_findings_with_metadata(findings, data)
        
        return findings
    
    def _categorize_discovery_finding(self, section_title: str) -> str:
        """Categorize discovery finding based on section title."""
        title_lower = section_title.lower()
        
        if 'catalog' in title_lower or 'asset' in title_lower:
            return 'asset_catalog'
        elif 'business' in title_lower or 'context' in title_lower:
            return 'business_context'
        elif 'relationship' in title_lower or 'mapping' in title_lower:
            return 'relationship_mapping'
        elif 'access' in title_lower or 'usage' in title_lower:
            return 'usage_patterns'
        elif 'domain' in title_lower or 'classification' in title_lower:
            return 'domain_classification'
        else:
            return 'general_discovery'
    
    def _parse_section_findings(self, section_content: str, category: str) -> List[Dict[str, Any]]:
        """Parse findings from a section."""
        findings = []
        
        # Split by bullet points or numbered items
        items = []
        current_item = []
        
        for line in section_content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this starts a new item
            if (line.startswith('-') or line.startswith('•') or 
                line.startswith('*') or line[0].isdigit()):
                
                if current_item:
                    items.append('\n'.join(current_item))
                current_item = [line]
            else:
                current_item.append(line)
        
        # Add the last item
        if current_item:
            items.append('\n'.join(current_item))
        
        # Process each item
        for item in items:
            if len(item.strip()) > 10:  # Skip very short items
                finding = self._parse_discovery_item(item, category)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _parse_discovery_item(self, item: str, category: str) -> Optional[Dict[str, Any]]:
        """Parse a single discovery item."""
        lines = [line.strip() for line in item.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Extract title from first line
        title = lines[0]
        title = title.lstrip('-•*0123456789. ')
        
        # Extract description from remaining lines
        description = '\n'.join(lines[1:]) if len(lines) > 1 else title
        
        # Extract metadata
        confidence = self._extract_confidence(description)
        domain = self._extract_domain(title + ' ' + description)
        importance = self._extract_importance(description)
        
        return {
            'title': title,
            'description': description,
            'category': category,
            'confidence': confidence,
            'domain': domain,
            'importance': importance,
            'type': 'discovery_finding',
            'agent_source': self.agent_name
        }
    
    def _extract_confidence(self, text: str) -> str:
        """Extract confidence level from text."""
        text_lower = text.lower()
        
        if 'high confidence' in text_lower or 'certain' in text_lower or 'clear' in text_lower:
            return 'high'
        elif 'low confidence' in text_lower or 'uncertain' in text_lower or 'unclear' in text_lower:
            return 'low'
        else:
            return 'medium'
    
    def _extract_domain(self, text: str) -> str:
        """Extract business domain from text."""
        text_lower = text.lower()
        
        # Common business domains
        domains = {
            'customer': ['customer', 'client', 'user', 'account'],
            'product': ['product', 'item', 'catalog', 'inventory'],
            'sales': ['sales', 'order', 'transaction', 'revenue'],
            'finance': ['finance', 'payment', 'billing', 'invoice'],
            'marketing': ['marketing', 'campaign', 'promotion', 'lead'],
            'operations': ['operations', 'logistics', 'supply', 'fulfillment'],
            'hr': ['employee', 'staff', 'payroll', 'hr'],
            'reference': ['reference', 'lookup', 'code', 'type']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _extract_importance(self, text: str) -> str:
        """Extract business importance from text."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['critical', 'core', 'essential', 'key']):
            return 'critical'
        elif any(term in text_lower for term in ['important', 'significant', 'major']):
            return 'important'
        else:
            return 'supporting'
    
    def _enhance_findings_with_metadata(self, findings: List[Dict[str, Any]], data: Dict[str, Any]):
        """Enhance findings with metadata from original data."""
        tables = data.get('tables', {})
        
        for finding in findings:
            # Try to associate finding with specific tables
            title_lower = finding['title'].lower()
            associated_tables = []
            
            for table_name in tables.keys():
                if table_name.lower() in title_lower:
                    associated_tables.append(table_name)
            
            if associated_tables:
                finding['associated_tables'] = associated_tables
                
                # Add table-specific metadata
                table_info = tables.get(associated_tables[0], {})
                finding['table_metadata'] = {
                    'row_count': table_info.get('row_count'),
                    'column_count': len(table_info.get('columns', [])),
                    'table_type': table_info.get('table_type')
                }
    
    def _calculate_discovery_confidence(self, data: Dict[str, Any], discoveries: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for discovery analysis."""
        base_confidence = 0.8  # High base confidence for discovery
        
        # Adjust based on data completeness
        tables = data.get('tables', {})
        if not tables:
            return 0.3
        
        # Increase confidence with more data
        table_count = len(tables)
        if table_count > 10:
            base_confidence += 0.1
        elif table_count < 3:
            base_confidence -= 0.2
        
        # Adjust based on relationship data availability
        relationships = data.get('relationships', [])
        if relationships:
            base_confidence += 0.1
        
        # Adjust based on number of discoveries
        if len(discoveries) > 5:
            base_confidence += 0.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def discover_data_domains(self, tables: Dict[str, Any]) -> AgentResponse:
        """Specialized method to discover and classify data domains."""
        try:
            prompt = f"""
Analyze the following tables and classify them into business domains:

TABLES TO CLASSIFY:
{self._format_tables_for_domain_analysis(tables)}

Please identify distinct business domains and classify each table. Common domains include:
- Customer/User Management
- Product/Inventory Management  
- Sales/Order Management
- Financial/Billing
- Marketing/Campaign
- Operations/Logistics
- Reference/Lookup Data
- System/Audit Data

For each domain, provide:
1. Domain name and description
2. Tables belonging to this domain
3. Key business processes supported
4. Data relationships within the domain
5. Cross-domain dependencies
"""
            
            llm_response = self._make_llm_request(self.system_prompt, prompt)
            domains = self._parse_domain_classifications(llm_response, tables)
            
            return AgentResponse(
                success=True,
                recommendations=domains,
                reasoning=llm_response,
                confidence=0.85,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'domain_discovery',
                    'domains_identified': len(domains),
                    'tables_classified': len(tables)
                }
            )
            
        except Exception as e:
            logger.error(f"Domain discovery failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _format_tables_for_domain_analysis(self, tables) -> str:
        """Format tables for domain analysis."""
        formatted = ""
        
        # Handle both list and dict formats
        if isinstance(tables, list):
            for table_name in tables:
                formatted += f"\n{table_name}:\n"
                formatted += f"  Type: table\n"
                formatted += f"  Status: discovered\n"
        elif isinstance(tables, dict):
            for table_name, table_info in tables.items():
                formatted += f"\n{table_name}:\n"
                formatted += f"  Rows: {table_info.get('row_count', 'unknown')}\n"
                formatted += f"  Type: {table_info.get('table_type', 'unknown')}\n"
            
            # Add key columns that indicate business purpose
            columns = table_info.get('columns', [])
            key_columns = []
            for col in columns[:15]:  # First 15 columns
                col_name = col.get('name', '').lower()
                # Include columns that suggest business context
                if any(indicator in col_name for indicator in 
                      ['id', 'name', 'type', 'status', 'date', 'amount', 'price', 'code']):
                    key_columns.append(col.get('name', ''))
            
            if key_columns:
                formatted += f"  Key Columns: {', '.join(key_columns[:10])}\n"
        
        return formatted
    
    def _parse_domain_classifications(self, llm_response: str, tables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse domain classifications from LLM response."""
        domains = []
        
        # Simple parsing - look for domain sections
        sections = self._split_into_sections(llm_response)
        
        for section_title, section_content in sections.items():
            if any(term in section_title.lower() for term in ['domain', 'classification', 'category']):
                domain_info = {
                    'title': f"Data Domain: {section_title}",
                    'description': section_content,
                    'category': 'domain_classification',
                    'domain': self._extract_domain(section_title),
                    'type': 'domain_discovery',
                    'tables': self._extract_table_references(section_content, tables),
                    'confidence': 'high'
                }
                domains.append(domain_info)
        
        return domains
    
    def _extract_table_references(self, content: str, tables: Dict[str, Any]) -> List[str]:
        """Extract table references from content."""
        referenced_tables = []
        content_lower = content.lower()
        
        for table_name in tables.keys():
            if table_name.lower() in content_lower:
                referenced_tables.append(table_name)
        
        return referenced_tables

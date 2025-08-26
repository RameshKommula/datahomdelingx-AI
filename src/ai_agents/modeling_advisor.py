"""AI agent for data modeling recommendations."""

from typing import Dict, List, Any, Optional
import json
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class ModelingAdvisor(BaseAgent):
    """AI agent that provides data modeling recommendations."""
    
    def __init__(self):
        super().__init__("ModelingAdvisor")
        
        self.system_prompt = """
You are an expert data modeling advisor with deep knowledge of data warehousing, 
dimensional modeling, and modern data lake architectures. Your role is to analyze 
database schemas and provide specific, actionable recommendations for improving 
data models.

Focus on:
1. Dimensional modeling best practices (star schema, snowflake schema)
2. Data normalization and denormalization strategies
3. Partitioning and clustering recommendations
4. Indexing strategies
5. Data quality improvements
6. Performance optimization
7. Data governance and security considerations

Provide specific, actionable recommendations with clear implementation guidance.
Format your response with clear sections and bullet points.
"""
    
    def analyze(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Analyze schema and provide modeling recommendations."""
        try:
            logger.info("ModelingAdvisor analyzing data model")
            
            # Prepare the analysis prompt
            user_prompt = self._create_analysis_prompt(data, context)
            
            # Get recommendations from LLM
            llm_response = self._make_llm_request(self.system_prompt, user_prompt)
            
            # Parse recommendations
            recommendations = self._parse_modeling_recommendations(llm_response)
            
            # Calculate confidence
            confidence = self._calculate_confidence(data, recommendations)
            
            return AgentResponse(
                success=True,
                recommendations=recommendations,
                reasoning=llm_response,
                confidence=confidence,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'data_modeling',
                    'tables_analyzed': len(data.get('tables', {})),
                    'recommendations_count': len(recommendations)
                }
            )
            
        except Exception as e:
            logger.error(f"ModelingAdvisor analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _create_analysis_prompt(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Create the analysis prompt for the LLM."""
        prompt = "Please analyze the following database schema and provide data modeling recommendations:\n\n"
        
        # Add schema information
        if 'tables' in data:
            prompt += "TABLES:\n"
            for table_name, table_info in data['tables'].items():
                prompt += self._format_table_info(table_info)
                prompt += "\n"
        
        # Add schema analysis if available
        if 'insights' in data:
            prompt += "SCHEMA ANALYSIS:\n"
            prompt += self._format_schema_analysis(data)
            prompt += "\n"
        
        # Add context if provided
        if context:
            prompt += "CONTEXT:\n"
            if 'business_domain' in context:
                prompt += f"Business Domain: {context['business_domain']}\n"
            if 'data_volume' in context:
                prompt += f"Data Volume: {context['data_volume']}\n"
            if 'query_patterns' in context:
                prompt += f"Query Patterns: {context['query_patterns']}\n"
            if 'performance_requirements' in context:
                prompt += f"Performance Requirements: {context['performance_requirements']}\n"
            prompt += "\n"
        
        prompt += """
Please provide specific recommendations in the following areas:

1. **Dimensional Modeling**: Suggest fact and dimension table designs
2. **Schema Design**: Recommend star schema, snowflake, or other patterns
3. **Data Normalization**: Identify over-normalization or under-normalization issues
4. **Partitioning Strategy**: Recommend partitioning columns and strategies
5. **Indexing Strategy**: Suggest indexes for performance optimization
6. **Data Quality**: Identify and address data quality issues
7. **Performance Optimization**: Recommend performance improvements
8. **Data Governance**: Suggest governance and security measures

For each recommendation, please include:
- Clear description of the recommendation
- Business justification
- Implementation priority (high/medium/low)
- Estimated implementation effort (low/medium/high)
- Expected benefits

Format your response with clear headings and bullet points.
"""
        
        return prompt
    
    def _parse_modeling_recommendations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured modeling recommendations."""
        recommendations = []
        
        # Enhanced parsing for modeling-specific recommendations
        sections = self._split_into_sections(llm_response)
        
        for section_title, section_content in sections.items():
            category = self._categorize_recommendation(section_title)
            section_recommendations = self._parse_section_recommendations(section_content, category)
            
            for rec in section_recommendations:
                rec['category'] = category
                recommendations.append(rec)
        
        return recommendations
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into sections based on headers."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a header line
            if (line.startswith('##') or line.startswith('**') or 
                (line.endswith(':') and len(line.split()) <= 4)):
                
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.replace('##', '').replace('**', '').replace(':', '').strip()
                current_content = []
            
            elif current_section:
                current_content.append(line)
        
        # Save the last section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _categorize_recommendation(self, section_title: str) -> str:
        """Categorize recommendation based on section title."""
        title_lower = section_title.lower()
        
        if 'dimensional' in title_lower or 'fact' in title_lower or 'dimension' in title_lower:
            return 'dimensional_modeling'
        elif 'schema' in title_lower or 'design' in title_lower:
            return 'schema_design'
        elif 'partition' in title_lower:
            return 'partitioning'
        elif 'index' in title_lower:
            return 'indexing'
        elif 'performance' in title_lower or 'optimization' in title_lower:
            return 'performance'
        elif 'quality' in title_lower:
            return 'data_quality'
        elif 'governance' in title_lower or 'security' in title_lower:
            return 'governance'
        elif 'normalization' in title_lower:
            return 'normalization'
        else:
            return 'general'
    
    def _parse_section_recommendations(self, section_content: str, category: str) -> List[Dict[str, Any]]:
        """Parse recommendations from a section."""
        recommendations = []
        
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
                rec = self._parse_recommendation_item(item, category)
                if rec:
                    recommendations.append(rec)
        
        return recommendations
    
    def _parse_recommendation_item(self, item: str, category: str) -> Optional[Dict[str, Any]]:
        """Parse a single recommendation item."""
        lines = [line.strip() for line in item.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Extract title from first line
        title = lines[0]
        title = title.lstrip('-•*0123456789. ')
        
        # Extract description from remaining lines
        description = '\n'.join(lines[1:]) if len(lines) > 1 else title
        
        # Extract metadata if present
        priority = self._extract_priority(description)
        effort = self._extract_effort(description)
        
        return {
            'title': title,
            'description': description,
            'category': category,
            'priority': priority,
            'implementation_effort': effort,
            'type': 'modeling_recommendation'
        }
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority from text."""
        text_lower = text.lower()
        
        if 'high priority' in text_lower or 'critical' in text_lower or 'urgent' in text_lower:
            return 'high'
        elif 'low priority' in text_lower or 'nice to have' in text_lower:
            return 'low'
        else:
            return 'medium'
    
    def _extract_effort(self, text: str) -> str:
        """Extract implementation effort from text."""
        text_lower = text.lower()
        
        if 'high effort' in text_lower or 'complex' in text_lower or 'significant' in text_lower:
            return 'high'
        elif 'low effort' in text_lower or 'simple' in text_lower or 'quick' in text_lower:
            return 'low'
        else:
            return 'medium'
    
    def analyze_dimensional_model(self, tables: Dict[str, Any]) -> AgentResponse:
        """Specifically analyze dimensional modeling aspects."""
        try:
            # Identify potential fact and dimension tables
            fact_tables = []
            dimension_tables = []
            
            for table_name, table_info in tables.items():
                table_type = table_info.get('table_type', 'unknown')
                if table_type == 'fact':
                    fact_tables.append(table_name)
                elif table_type == 'dimension':
                    dimension_tables.append(table_name)
            
            prompt = f"""
Analyze the dimensional modeling aspects of this schema:

FACT TABLES: {', '.join(fact_tables) if fact_tables else 'None identified'}
DIMENSION TABLES: {', '.join(dimension_tables) if dimension_tables else 'None identified'}

TABLE DETAILS:
{self._format_tables_for_dimensional_analysis(tables)}

Please provide specific recommendations for:
1. Fact table design and optimization
2. Dimension table design (SCD handling, hierarchies)
3. Star vs snowflake schema recommendations
4. Bridge tables for many-to-many relationships
5. Conformed dimensions opportunities
"""
            
            llm_response = self._make_llm_request(self.system_prompt, prompt)
            recommendations = self._parse_modeling_recommendations(llm_response)
            
            return AgentResponse(
                success=True,
                recommendations=recommendations,
                reasoning=llm_response,
                confidence=self._calculate_confidence({'tables': tables}, recommendations),
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'dimensional_modeling',
                    'fact_tables': len(fact_tables),
                    'dimension_tables': len(dimension_tables)
                }
            )
            
        except Exception as e:
            logger.error(f"Dimensional model analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _format_tables_for_dimensional_analysis(self, tables: Dict[str, Any]) -> str:
        """Format tables specifically for dimensional analysis."""
        formatted = ""
        
        for table_name, table_info in tables.items():
            formatted += f"\n{table_name} ({table_info.get('table_type', 'unknown')}):\n"
            
            # Group columns by type
            measures = []
            dimensions = []
            keys = []
            
            for col in table_info.get('columns', []):
                col_name = col.get('name', '')
                col_type = col.get('data_type', '').lower()
                
                if col.get('is_primary_key') or col.get('is_foreign_key'):
                    keys.append(col_name)
                elif col_type in ['int', 'bigint', 'float', 'double', 'decimal', 'numeric']:
                    measures.append(col_name)
                else:
                    dimensions.append(col_name)
            
            if keys:
                formatted += f"  Keys: {', '.join(keys)}\n"
            if measures:
                formatted += f"  Measures: {', '.join(measures)}\n"
            if dimensions:
                formatted += f"  Dimensions: {', '.join(dimensions)}\n"
        
        return formatted

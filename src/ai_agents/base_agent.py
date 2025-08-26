"""Base class for AI agents."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai
import anthropic
from loguru import logger

from config import get_config


@dataclass
class AgentResponse:
    """Response from an AI agent."""
    success: bool
    recommendations: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all AI agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.config = get_config()
        
        # Initialize the appropriate LLM client
        if self.config.llm_provider == "claude":
            if not self.config.claude or not self.config.claude.api_key:
                raise ValueError("Claude configuration is required when llm_provider is 'claude'")
            self.claude_client = anthropic.Anthropic(api_key=self.config.claude.api_key)
            self.openai_client = None
        else:  # openai
            if not self.config.openai or not self.config.openai.api_key:
                raise ValueError("OpenAI configuration is required when llm_provider is 'openai'")
            self.openai_client = openai.OpenAI(api_key=self.config.openai.api_key)
            self.claude_client = None
        
    @abstractmethod
    def analyze(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Analyze data and provide recommendations."""
        pass
    
    def _make_llm_request(self, system_prompt: str, user_prompt: str, 
                         temperature: Optional[float] = None) -> str:
        """Make a request to the language model."""
        try:
            if self.config.llm_provider == "claude":
                response = self.claude_client.messages.create(
                    model=self.config.claude.model,
                    max_tokens=self.config.claude.max_tokens,
                    temperature=temperature or self.config.claude.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text.strip()
            else:  # openai
                response = self.openai_client.chat.completions.create(
                    model=self.config.openai.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature or self.config.openai.temperature,
                    max_tokens=self.config.openai.max_tokens
                )
                return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM request failed for {self.agent_name}: {e}")
            raise
    
    def _format_table_info(self, table_info: Dict[str, Any]) -> str:
        """Format table information for LLM consumption."""
        formatted = f"Table: {table_info.get('database', 'unknown')}.{table_info.get('name', 'unknown')}\n"
        formatted += f"Rows: {table_info.get('row_count', 'unknown')}\n"
        formatted += f"Size: {table_info.get('size_bytes', 'unknown')} bytes\n"
        formatted += f"Table Type: {table_info.get('table_type', 'unknown')}\n\n"
        
        formatted += "Columns:\n"
        for col in table_info.get('columns', []):
            formatted += f"- {col.get('name', 'unknown')} ({col.get('data_type', 'unknown')})"
            if col.get('nullable'):
                formatted += " [nullable]"
            if col.get('is_primary_key'):
                formatted += " [PK]"
            if col.get('is_foreign_key'):
                formatted += " [FK]"
            formatted += f" - Quality: {col.get('data_quality_score', 'unknown')}\n"
            
            if col.get('quality_issues'):
                formatted += f"  Issues: {', '.join(col['quality_issues'])}\n"
        
        return formatted
    
    def _format_schema_analysis(self, schema_analysis: Dict[str, Any]) -> str:
        """Format schema analysis results for LLM consumption."""
        formatted = f"Database: {schema_analysis.get('database', 'unknown')}\n"
        formatted += f"Total Tables: {schema_analysis.get('summary', {}).get('total_tables', 0)}\n"
        formatted += f"Total Columns: {schema_analysis.get('summary', {}).get('total_columns', 0)}\n"
        formatted += f"Total Rows: {schema_analysis.get('summary', {}).get('total_rows', 0)}\n\n"
        
        # Table types distribution
        insights = schema_analysis.get('insights', {})
        table_types = insights.get('table_types', {})
        if table_types:
            formatted += "Table Types:\n"
            for table_type, count in table_types.items():
                formatted += f"- {table_type}: {count}\n"
            formatted += "\n"
        
        # Relationships
        relationships = schema_analysis.get('relationships', [])
        if relationships:
            formatted += f"Relationships: {len(relationships)} detected\n"
            for rel in relationships[:5]:  # Show first 5
                formatted += f"- {rel.get('from_table', 'unknown')}.{rel.get('from_column', 'unknown')} -> "
                formatted += f"{rel.get('to_table', 'unknown')}.{rel.get('to_column', 'unknown')} "
                formatted += f"(confidence: {rel.get('confidence', 0):.2f})\n"
            formatted += "\n"
        
        # Data quality
        data_quality = insights.get('data_quality', {})
        if data_quality:
            formatted += f"Average Data Quality Score: {data_quality.get('average_score', 0):.2f}\n"
            
            tables_needing_attention = data_quality.get('tables_needing_attention', [])
            if tables_needing_attention:
                formatted += f"Tables Needing Attention: {', '.join(tables_needing_attention[:5])}\n"
            formatted += "\n"
        
        return formatted
    
    def _parse_llm_recommendations(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured recommendations."""
        recommendations = []
        
        # Simple parsing - in production, you might want more sophisticated parsing
        lines = llm_response.split('\n')
        current_recommendation = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_recommendation:
                    recommendations.append(current_recommendation)
                    current_recommendation = {}
                continue
            
            # Look for recommendation patterns
            if line.startswith('**') or line.startswith('##'):
                # New recommendation title
                if current_recommendation:
                    recommendations.append(current_recommendation)
                
                title = line.replace('**', '').replace('##', '').strip()
                current_recommendation = {
                    'title': title,
                    'description': '',
                    'priority': 'medium',
                    'category': 'general',
                    'implementation_effort': 'medium'
                }
            
            elif line.startswith('-') or line.startswith('•'):
                # Bullet point - add to description
                if current_recommendation:
                    current_recommendation['description'] += line + '\n'
            
            elif ':' in line and current_recommendation:
                # Key-value pair
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in ['priority', 'category', 'implementation_effort']:
                    current_recommendation[key] = value.lower()
                else:
                    current_recommendation['description'] += line + '\n'
            
            else:
                # Regular text - add to description
                if current_recommendation:
                    current_recommendation['description'] += line + '\n'
        
        # Add the last recommendation
        if current_recommendation:
            recommendations.append(current_recommendation)
        
        # Clean up descriptions
        for rec in recommendations:
            if 'description' in rec:
                rec['description'] = rec['description'].strip()
        
        return recommendations
    
    def _calculate_confidence(self, data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for recommendations."""
        # Base confidence factors
        confidence = 0.7  # Base confidence
        
        # Increase confidence based on data quality
        if 'data_quality_score' in data:
            quality_score = data['data_quality_score']
            if quality_score > 0.8:
                confidence += 0.1
            elif quality_score < 0.5:
                confidence -= 0.1
        
        # Increase confidence based on data completeness
        if 'completeness_score' in data:
            completeness = data['completeness_score']
            if completeness > 0.9:
                confidence += 0.1
            elif completeness < 0.7:
                confidence -= 0.1
        
        # Adjust based on number of recommendations
        if len(recommendations) > 10:
            confidence -= 0.1  # Too many recommendations might indicate uncertainty
        elif len(recommendations) < 3:
            confidence += 0.05  # Focused recommendations
        
        return max(0.0, min(1.0, confidence))

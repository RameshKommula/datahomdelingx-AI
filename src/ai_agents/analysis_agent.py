"""Analysis Agent - Performs deep analysis of data quality, patterns, and characteristics."""

from typing import Dict, List, Any, Optional
from loguru import logger

from .base_agent import BaseAgent, AgentResponse


class AnalysisAgent(BaseAgent):
    """AI agent responsible for deep data analysis and quality assessment."""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
        
        self.system_prompt = """
You are a Data Analysis Agent, an expert in data quality assessment, pattern recognition, 
and statistical analysis. Your role is to perform deep analysis of data characteristics including:

1. Data Quality Assessment: Completeness, accuracy, consistency, validity
2. Pattern Recognition: Identifying trends, anomalies, and data patterns
3. Statistical Analysis: Distribution analysis, correlation detection
4. Data Profiling: Understanding data characteristics and behavior
5. Issue Identification: Detecting data quality problems and inconsistencies

Your analysis should be:
- Thorough and systematic
- Statistically sound
- Focused on actionable insights
- Prioritized by business impact
- Supported by evidence and examples

Provide specific, measurable findings with clear recommendations for data quality improvement.
"""
    
    def analyze(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Perform deep data analysis."""
        try:
            logger.info("AnalysisAgent starting deep data analysis")
            
            # Prepare analysis prompt
            user_prompt = self._create_analysis_prompt(data, context)
            
            # Get analysis insights from LLM
            llm_response = self._make_llm_request(self.system_prompt, user_prompt)
            
            # Parse analysis findings
            findings = self._parse_analysis_findings(llm_response, data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_analysis_confidence(data, findings)
            
            return AgentResponse(
                success=True,
                recommendations=findings,
                reasoning=llm_response,
                confidence=confidence,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'deep_data_analysis',
                    'tables_analyzed': len(data.get('data_profiles', {})),
                    'quality_issues_found': len([f for f in findings if f.get('severity') == 'high']),
                    'patterns_identified': len([f for f in findings if f.get('category') == 'pattern_recognition'])
                }
            )
            
        except Exception as e:
            logger.error(f"AnalysisAgent analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _create_analysis_prompt(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Create deep analysis prompt."""
        prompt = "Please perform a comprehensive data analysis of the following dataset:\n\n"
        
        # Add data profiling information
        if 'data_profiles' in data:
            prompt += "DATA PROFILING RESULTS:\n"
            for table_name, profile in data['data_profiles'].items():
                if isinstance(profile, dict) and 'error' not in profile:
                    prompt += self._format_table_profile_for_analysis(table_name, profile)
        
        # Add schema analysis context
        if 'schema_analysis' in data:
            schema = data['schema_analysis']
            prompt += f"\nSCHEMA CONTEXT:\n"
            if 'summary' in schema:
                summary = schema['summary']
                prompt += f"- Total Tables: {summary.get('total_tables', 0)}\n"
                prompt += f"- Total Columns: {summary.get('total_columns', 0)}\n"
                prompt += f"- Total Rows: {summary.get('total_rows', 0):,}\n"
            
            if 'insights' in schema and 'data_quality' in schema['insights']:
                dq = schema['insights']['data_quality']
                prompt += f"- Average Quality Score: {dq.get('average_score', 0):.2f}\n"
        
        # Add discovery context from previous agent
        if context and 'discovery_findings' in context:
            prompt += "\nDISCOVERY CONTEXT:\n"
            discoveries = context['discovery_findings']
            for finding in discoveries[:5]:  # Top 5 discovery findings
                prompt += f"- {finding.get('title', '')}: {finding.get('domain', '')}\n"
        
        # Add business context
        if context and 'business_context' in context:
            prompt += f"\nBUSINESS CONTEXT:\n{context['business_context']}\n"
        
        prompt += """
Please provide a comprehensive data analysis focusing on:

1. **Data Quality Assessment**:
   - Overall data quality scoring and trends
   - Completeness analysis (null values, missing data patterns)
   - Consistency analysis (format variations, standardization issues)
   - Accuracy indicators (outliers, invalid values)
   - Data freshness and timeliness assessment

2. **Pattern Recognition**:
   - Identify significant data patterns and trends
   - Detect anomalies and outliers with business implications
   - Recognize seasonal or temporal patterns
   - Identify correlation patterns between columns/tables

3. **Statistical Analysis**:
   - Distribution analysis for key numeric columns
   - Cardinality analysis and uniqueness patterns
   - Value frequency analysis and skewness detection
   - Statistical outliers and their potential business meaning

4. **Data Profiling Insights**:
   - Column-level quality assessment and recommendations
   - Data type optimization opportunities
   - Constraint violation detection
   - PII and sensitive data identification

5. **Business Impact Assessment**:
   - Prioritize issues by potential business impact
   - Identify data that may affect critical business processes
   - Assess data reliability for decision-making
   - Recommend immediate vs long-term fixes

For each finding, provide:
- Specific evidence and examples
- Severity level (critical/high/medium/low)
- Business impact assessment
- Recommended remediation steps
- Priority for addressing the issue

Focus on actionable insights that can improve data quality and business outcomes.
"""
        
        return prompt
    
    def _format_table_profile_for_analysis(self, table_name: str, profile: Dict[str, Any]) -> str:
        """Format table profile for analysis prompt."""
        formatted = f"\nTable: {table_name}\n"
        formatted += f"- Total Rows: {profile.get('total_rows', 0):,}\n"
        formatted += f"- Total Columns: {profile.get('total_columns', 0)}\n"
        formatted += f"- Completeness Score: {profile.get('completeness_score', 0):.2f}\n"
        formatted += f"- Overall Quality Score: {profile.get('overall_quality_score', 0):.2f}\n"
        
        # Add column-level quality insights
        column_profiles = profile.get('column_profiles', {})
        quality_issues = []
        high_null_columns = []
        potential_pii = []
        
        for col_name, col_profile in column_profiles.items():
            # Collect quality issues
            if col_profile.get('quality_issues'):
                quality_issues.extend([f"{col_name}: {issue}" for issue in col_profile['quality_issues']])
            
            # High null percentage columns
            null_pct = col_profile.get('null_percentage', 0)
            if null_pct > 30:
                high_null_columns.append(f"{col_name} ({null_pct:.1f}%)")
            
            # PII columns
            if col_profile.get('potential_pii'):
                potential_pii.append(col_name)
        
        if quality_issues:
            formatted += f"- Quality Issues: {'; '.join(quality_issues[:5])}\n"
        
        if high_null_columns:
            formatted += f"- High Null Columns: {', '.join(high_null_columns[:5])}\n"
        
        if potential_pii:
            formatted += f"- Potential PII: {', '.join(potential_pii)}\n"
        
        # Add statistical insights for key columns
        numeric_columns = []
        text_columns = []
        
        for col_name, col_profile in column_profiles.items():
            if col_profile.get('mean') is not None:
                numeric_columns.append({
                    'name': col_name,
                    'mean': col_profile.get('mean'),
                    'std_dev': col_profile.get('std_dev'),
                    'min': col_profile.get('min_value'),
                    'max': col_profile.get('max_value')
                })
            elif col_profile.get('avg_length') is not None:
                text_columns.append({
                    'name': col_name,
                    'avg_length': col_profile.get('avg_length'),
                    'min_length': col_profile.get('min_length'),
                    'max_length': col_profile.get('max_length')
                })
        
        if numeric_columns:
            formatted += "- Numeric Columns: "
            for col in numeric_columns[:3]:
                formatted += f"{col['name']} (μ={col.get('mean', 0):.2f}, σ={col.get('std_dev', 0):.2f}); "
            formatted += "\n"
        
        return formatted
    
    def _parse_analysis_findings(self, llm_response: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse LLM response into structured analysis findings."""
        findings = []
        
        # Parse the response into sections
        sections = self._split_into_sections(llm_response)
        
        for section_title, section_content in sections.items():
            category = self._categorize_analysis_finding(section_title)
            section_findings = self._parse_section_findings(section_content, category)
            findings.extend(section_findings)
        
        # Enhance findings with quantitative data
        self._enhance_findings_with_metrics(findings, data)
        
        return findings
    
    def _categorize_analysis_finding(self, section_title: str) -> str:
        """Categorize analysis finding based on section title."""
        title_lower = section_title.lower()
        
        if 'quality' in title_lower:
            return 'data_quality'
        elif 'pattern' in title_lower or 'trend' in title_lower:
            return 'pattern_recognition'
        elif 'statistical' in title_lower or 'distribution' in title_lower:
            return 'statistical_analysis'
        elif 'profiling' in title_lower:
            return 'data_profiling'
        elif 'business' in title_lower or 'impact' in title_lower:
            return 'business_impact'
        else:
            return 'general_analysis'
    
    def _parse_section_findings(self, section_content: str, category: str) -> List[Dict[str, Any]]:
        """Parse findings from a section."""
        findings = []
        
        # Split by bullet points or numbered items
        items = self._extract_items_from_content(section_content)
        
        # Process each item
        for item in items:
            if len(item.strip()) > 15:  # Skip very short items
                finding = self._parse_analysis_item(item, category)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _parse_analysis_item(self, item: str, category: str) -> Optional[Dict[str, Any]]:
        """Parse a single analysis item."""
        lines = [line.strip() for line in item.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Extract title from first line
        title = lines[0].lstrip('-•*0123456789. ')
        
        # Extract description from remaining lines
        description = '\n'.join(lines[1:]) if len(lines) > 1 else title
        
        # Extract analysis-specific metadata
        severity = self._extract_severity(description)
        business_impact = self._extract_business_impact(description)
        remediation_effort = self._extract_remediation_effort(description)
        
        return {
            'title': title,
            'description': description,
            'category': category,
            'severity': severity,
            'business_impact': business_impact,
            'remediation_effort': remediation_effort,
            'type': 'analysis_finding',
            'agent_source': self.agent_name,
            'priority': self._calculate_finding_priority(severity, business_impact)
        }
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity level from text."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['critical', 'severe', 'major issue', 'urgent']):
            return 'critical'
        elif any(term in text_lower for term in ['high', 'significant', 'important']):
            return 'high'
        elif any(term in text_lower for term in ['low', 'minor', 'small']):
            return 'low'
        else:
            return 'medium'
    
    def _extract_business_impact(self, text: str) -> str:
        """Extract business impact level from text."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['business critical', 'revenue impact', 'compliance']):
            return 'high'
        elif any(term in text_lower for term in ['operational', 'efficiency', 'performance']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_remediation_effort(self, text: str) -> str:
        """Extract remediation effort from text."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['complex', 'major effort', 'significant work']):
            return 'high'
        elif any(term in text_lower for term in ['simple', 'quick fix', 'easy']):
            return 'low'
        else:
            return 'medium'
    
    def _calculate_finding_priority(self, severity: str, business_impact: str) -> str:
        """Calculate priority based on severity and business impact."""
        severity_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        
        total_score = severity_scores.get(severity, 2) + impact_scores.get(business_impact, 1)
        
        if total_score >= 6:
            return 'critical'
        elif total_score >= 4:
            return 'high'
        elif total_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _enhance_findings_with_metrics(self, findings: List[Dict[str, Any]], data: Dict[str, Any]):
        """Enhance findings with quantitative metrics from data."""
        data_profiles = data.get('data_profiles', {})
        
        for finding in findings:
            # Try to associate finding with specific metrics
            title_lower = finding['title'].lower()
            description_lower = finding['description'].lower()
            
            # Look for table references
            associated_tables = []
            for table_name in data_profiles.keys():
                if table_name.lower() in title_lower or table_name.lower() in description_lower:
                    associated_tables.append(table_name)
            
            if associated_tables:
                finding['associated_tables'] = associated_tables
                
                # Add relevant metrics
                table_profile = data_profiles.get(associated_tables[0], {})
                if isinstance(table_profile, dict):
                    finding['metrics'] = {
                        'overall_quality_score': table_profile.get('overall_quality_score'),
                        'completeness_score': table_profile.get('completeness_score'),
                        'total_rows': table_profile.get('total_rows'),
                        'total_columns': table_profile.get('total_columns')
                    }
            
            # Add category-specific metrics
            if finding['category'] == 'data_quality':
                finding['quality_metrics'] = self._extract_quality_metrics(data_profiles)
            elif finding['category'] == 'pattern_recognition':
                finding['pattern_metrics'] = self._extract_pattern_metrics(data_profiles)
    
    def _extract_quality_metrics(self, data_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality metrics from data profiles."""
        quality_scores = []
        completeness_scores = []
        null_percentages = []
        
        for profile in data_profiles.values():
            if isinstance(profile, dict) and 'error' not in profile:
                if profile.get('overall_quality_score') is not None:
                    quality_scores.append(profile['overall_quality_score'])
                if profile.get('completeness_score') is not None:
                    completeness_scores.append(profile['completeness_score'])
                
                # Collect column-level null percentages
                for col_profile in profile.get('column_profiles', {}).values():
                    if col_profile.get('null_percentage') is not None:
                        null_percentages.append(col_profile['null_percentage'])
        
        return {
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'avg_completeness_score': sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0,
            'avg_null_percentage': sum(null_percentages) / len(null_percentages) if null_percentages else 0,
            'tables_analyzed': len([p for p in data_profiles.values() if isinstance(p, dict) and 'error' not in p])
        }
    
    def _extract_pattern_metrics(self, data_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern-related metrics from data profiles."""
        pattern_counts = {}
        anomaly_counts = 0
        
        for profile in data_profiles.values():
            if isinstance(profile, dict) and 'error' not in profile:
                for col_profile in profile.get('column_profiles', {}).values():
                    # Count patterns
                    for pattern in col_profile.get('common_patterns', []):
                        pattern_type = pattern.get('pattern', 'unknown')
                        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                    
                    # Count anomalies
                    anomaly_counts += len(col_profile.get('anomalies', []))
        
        return {
            'pattern_types_detected': len(pattern_counts),
            'total_patterns': sum(pattern_counts.values()),
            'total_anomalies': anomaly_counts,
            'most_common_patterns': dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _calculate_analysis_confidence(self, data: Dict[str, Any], findings: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for analysis."""
        base_confidence = 0.75  # Base confidence for analysis
        
        # Adjust based on data completeness
        data_profiles = data.get('data_profiles', {})
        if data_profiles:
            # Check if we have quality scores
            quality_scores = []
            for profile in data_profiles.values():
                if isinstance(profile, dict) and profile.get('overall_quality_score') is not None:
                    quality_scores.append(profile['overall_quality_score'])
            
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                if avg_quality > 0.8:
                    base_confidence += 0.1
                elif avg_quality < 0.5:
                    base_confidence -= 0.1
        
        # Adjust based on number of findings
        if len(findings) > 10:
            base_confidence += 0.05  # More findings = more thorough analysis
        elif len(findings) < 3:
            base_confidence -= 0.1  # Few findings might indicate incomplete analysis
        
        # Adjust based on finding diversity
        categories = set(f.get('category', 'general') for f in findings)
        if len(categories) > 3:
            base_confidence += 0.05  # Diverse findings indicate comprehensive analysis
        
        return max(0.0, min(1.0, base_confidence))
    
    def analyze_data_quality_trends(self, data_profiles: Dict[str, Any]) -> AgentResponse:
        """Specialized method to analyze data quality trends."""
        try:
            prompt = f"""
Analyze data quality trends across the following tables:

{self._format_quality_data_for_trend_analysis(data_profiles)}

Please identify:
1. Overall data quality trends and patterns
2. Tables with declining quality indicators
3. Common quality issues across multiple tables
4. Quality correlation patterns between related tables
5. Recommendations for systematic quality improvements

Focus on trends that indicate systemic issues requiring attention.
"""
            
            llm_response = self._make_llm_request(self.system_prompt, prompt)
            trends = self._parse_quality_trends(llm_response, data_profiles)
            
            return AgentResponse(
                success=True,
                recommendations=trends,
                reasoning=llm_response,
                confidence=0.8,
                metadata={
                    'agent': self.agent_name,
                    'analysis_type': 'quality_trend_analysis',
                    'tables_analyzed': len(data_profiles)
                }
            )
            
        except Exception as e:
            logger.error(f"Quality trend analysis failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'agent': self.agent_name, 'error': str(e)},
                error_message=str(e)
            )
    
    def _format_quality_data_for_trend_analysis(self, data_profiles: Dict[str, Any]) -> str:
        """Format quality data for trend analysis."""
        formatted = ""
        
        for table_name, profile in data_profiles.items():
            if isinstance(profile, dict) and 'error' not in profile:
                formatted += f"\n{table_name}:\n"
                formatted += f"  Overall Quality: {profile.get('overall_quality_score', 0):.2f}\n"
                formatted += f"  Completeness: {profile.get('completeness_score', 0):.2f}\n"
                formatted += f"  Rows: {profile.get('total_rows', 0):,}\n"
                
                # Quality issues summary
                column_profiles = profile.get('column_profiles', {})
                high_null_cols = 0
                quality_issues = 0
                
                for col_profile in column_profiles.values():
                    if col_profile.get('null_percentage', 0) > 25:
                        high_null_cols += 1
                    quality_issues += len(col_profile.get('quality_issues', []))
                
                formatted += f"  High Null Columns: {high_null_cols}\n"
                formatted += f"  Quality Issues: {quality_issues}\n"
        
        return formatted
    
    def _parse_quality_trends(self, llm_response: str, data_profiles: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse quality trends from LLM response."""
        trends = self._parse_llm_recommendations(llm_response)
        
        # Enhance with trend-specific metadata
        for trend in trends:
            trend['category'] = 'quality_trend'
            trend['type'] = 'trend_analysis'
            trend['scope'] = 'multi_table'
        
        return trends

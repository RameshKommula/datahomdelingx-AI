"""Data profiling engine for comprehensive data analysis."""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime, timedelta
from loguru import logger

from connectors.base import BaseConnector


@dataclass
class DataProfile:
    """Comprehensive data profile for a column or table."""
    
    # Basic information
    column_name: str
    data_type: str
    total_count: int
    
    # Null analysis
    null_count: int = 0
    null_percentage: float = 0.0
    
    # Uniqueness
    unique_count: int = 0
    unique_percentage: float = 0.0
    duplicate_count: int = 0
    
    # Statistical measures (for numeric data)
    min_value: Any = None
    max_value: Any = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Distribution analysis
    quartiles: Optional[Dict[str, float]] = None
    percentiles: Optional[Dict[str, float]] = None
    
    # String analysis (for text data)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Pattern analysis
    common_patterns: List[Dict[str, Any]] = field(default_factory=list)
    format_consistency: Optional[float] = None
    
    # Value analysis
    most_frequent_values: List[Dict[str, Any]] = field(default_factory=list)
    least_frequent_values: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data quality indicators
    quality_score: float = 0.0
    quality_issues: List[str] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Business insights
    potential_pii: bool = False
    potential_key: bool = False
    data_classification: Optional[str] = None
    
    # Temporal analysis (for date/time data)
    date_range: Optional[Dict[str, Any]] = None
    temporal_patterns: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TableProfile:
    """Comprehensive profile for an entire table."""
    
    database: str
    table_name: str
    total_rows: int
    total_columns: int
    
    # Column profiles
    column_profiles: Dict[str, DataProfile] = field(default_factory=dict)
    
    # Table-level statistics
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    overall_quality_score: float = 0.0
    
    # Relationships and dependencies
    potential_keys: List[str] = field(default_factory=list)
    potential_foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    
    # Business insights
    table_classification: Optional[str] = None
    recommended_partitioning: List[str] = field(default_factory=list)
    recommended_indexing: List[str] = field(default_factory=list)
    
    # Data freshness
    freshness_analysis: Optional[Dict[str, Any]] = None
    
    # Anomalies and issues
    table_level_issues: List[str] = field(default_factory=list)


class DataProfiler:
    """Comprehensive data profiling engine."""
    
    def __init__(self, connector: BaseConnector, sample_size: int = 10000):
        self.connector = connector
        self.sample_size = sample_size
        
        # Pattern definitions for common data types
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        }
    
    def profile_table(self, database: str, table_name: str, 
                     columns: Optional[List[str]] = None) -> TableProfile:
        """Profile an entire table."""
        logger.info(f"Profiling table: {database}.{table_name}")
        
        # Get table schema and sample data
        schema = self.connector.get_table_schema(database, table_name)
        sample_data = self.connector.get_sample_data(database, table_name, self.sample_size)
        table_stats = self.connector.get_table_stats(database, table_name)
        
        # Create table profile
        table_profile = TableProfile(
            database=database,
            table_name=table_name,
            total_rows=table_stats.get('row_count', len(sample_data)),
            total_columns=len(schema)
        )
        
        # Profile each column
        columns_to_profile = columns or [col['column_name'] for col in schema]
        
        for col_info in schema:
            col_name = col_info['column_name']
            if col_name not in columns_to_profile:
                continue
                
            if col_name in sample_data.columns:
                col_profile = self.profile_column(
                    col_name, 
                    col_info['data_type'], 
                    sample_data[col_name]
                )
                table_profile.column_profiles[col_name] = col_profile
        
        # Analyze table-level patterns
        self.analyze_table_relationships(table_profile)
        self.analyze_table_quality(table_profile)
        self.generate_table_recommendations(table_profile)
        
        return table_profile
    
    def profile_column(self, column_name: str, data_type: str, 
                      data: pd.Series) -> DataProfile:
        """Profile a single column."""
        logger.debug(f"Profiling column: {column_name}")
        
        profile = DataProfile(
            column_name=column_name,
            data_type=data_type,
            total_count=len(data)
        )
        
        # Basic null analysis
        profile.null_count = data.isnull().sum()
        profile.null_percentage = (profile.null_count / len(data)) * 100
        
        # Uniqueness analysis
        non_null_data = data.dropna()
        profile.unique_count = non_null_data.nunique()
        profile.unique_percentage = (profile.unique_count / len(non_null_data)) * 100 if len(non_null_data) > 0 else 0
        profile.duplicate_count = len(non_null_data) - profile.unique_count
        
        if len(non_null_data) > 0:
            # Statistical analysis for numeric data
            if pd.api.types.is_numeric_dtype(non_null_data):
                self.analyze_numeric_column(profile, non_null_data)
            
            # String analysis for text data
            elif pd.api.types.is_string_dtype(non_null_data):
                self.analyze_text_column(profile, non_null_data)
            
            # Temporal analysis for datetime data
            elif pd.api.types.is_datetime64_any_dtype(non_null_data):
                self.analyze_datetime_column(profile, non_null_data)
            
            # Value frequency analysis
            self.analyze_value_frequencies(profile, non_null_data)
            
            # Pattern analysis
            self.analyze_patterns(profile, non_null_data)
            
            # Anomaly detection
            self.detect_anomalies(profile, non_null_data)
            
            # Business insights
            self.generate_business_insights(profile, non_null_data)
        
        # Calculate quality score
        profile.quality_score = self.calculate_column_quality_score(profile)
        
        return profile
    
    def analyze_numeric_column(self, profile: DataProfile, data: pd.Series):
        """Analyze numeric column statistics."""
        try:
            profile.min_value = float(data.min())
            profile.max_value = float(data.max())
            profile.mean = float(data.mean())
            profile.median = float(data.median())
            profile.std_dev = float(data.std())
            profile.variance = float(data.var())
            
            # Advanced statistics
            profile.skewness = float(stats.skew(data))
            profile.kurtosis = float(stats.kurtosis(data))
            
            # Quartiles and percentiles
            profile.quartiles = {
                'Q1': float(data.quantile(0.25)),
                'Q2': float(data.quantile(0.5)),
                'Q3': float(data.quantile(0.75))
            }
            
            profile.percentiles = {
                'P1': float(data.quantile(0.01)),
                'P5': float(data.quantile(0.05)),
                'P95': float(data.quantile(0.95)),
                'P99': float(data.quantile(0.99))
            }
            
        except Exception as e:
            logger.warning(f"Error in numeric analysis for {profile.column_name}: {e}")
    
    def analyze_text_column(self, profile: DataProfile, data: pd.Series):
        """Analyze text column characteristics."""
        try:
            # String length analysis
            lengths = data.astype(str).str.len()
            profile.min_length = int(lengths.min())
            profile.max_length = int(lengths.max())
            profile.avg_length = float(lengths.mean())
            
            # Format consistency
            unique_lengths = lengths.nunique()
            total_values = len(lengths)
            profile.format_consistency = 1.0 - (unique_lengths / total_values) if total_values > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in text analysis for {profile.column_name}: {e}")
    
    def analyze_datetime_column(self, profile: DataProfile, data: pd.Series):
        """Analyze datetime column patterns."""
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data):
                data = pd.to_datetime(data, errors='coerce')
            
            data = data.dropna()
            if len(data) == 0:
                return
            
            profile.min_value = data.min()
            profile.max_value = data.max()
            
            # Date range analysis
            date_range = data.max() - data.min()
            profile.date_range = {
                'start_date': data.min(),
                'end_date': data.max(),
                'range_days': date_range.days,
                'range_description': self.describe_date_range(date_range)
            }
            
            # Temporal patterns
            profile.temporal_patterns = self.analyze_temporal_patterns(data)
            
        except Exception as e:
            logger.warning(f"Error in datetime analysis for {profile.column_name}: {e}")
    
    def analyze_value_frequencies(self, profile: DataProfile, data: pd.Series):
        """Analyze value frequency distribution."""
        try:
            value_counts = data.value_counts()
            total_count = len(data)
            
            # Most frequent values
            profile.most_frequent_values = [
                {
                    'value': str(value),
                    'count': int(count),
                    'percentage': (count / total_count) * 100
                }
                for value, count in value_counts.head(10).items()
            ]
            
            # Least frequent values (excluding single occurrences if many)
            if len(value_counts) > 20:
                least_frequent = value_counts.tail(10)
            else:
                least_frequent = value_counts.tail(min(5, len(value_counts)))
            
            profile.least_frequent_values = [
                {
                    'value': str(value),
                    'count': int(count),
                    'percentage': (count / total_count) * 100
                }
                for value, count in least_frequent.items()
            ]
            
        except Exception as e:
            logger.warning(f"Error in frequency analysis for {profile.column_name}: {e}")
    
    def analyze_patterns(self, profile: DataProfile, data: pd.Series):
        """Analyze common patterns in the data."""
        try:
            data_str = data.astype(str)
            
            # Check against predefined patterns
            for pattern_name, pattern_regex in self.patterns.items():
                matches = data_str.str.match(pattern_regex, na=False).sum()
                if matches > 0:
                    match_percentage = (matches / len(data_str)) * 100
                    profile.common_patterns.append({
                        'pattern': pattern_name,
                        'matches': matches,
                        'percentage': match_percentage,
                        'regex': pattern_regex
                    })
            
            # Analyze custom patterns (e.g., consistent formatting)
            self.analyze_custom_patterns(profile, data_str)
            
        except Exception as e:
            logger.warning(f"Error in pattern analysis for {profile.column_name}: {e}")
    
    def analyze_custom_patterns(self, profile: DataProfile, data: pd.Series):
        """Analyze custom patterns specific to the data."""
        try:
            # Analyze numeric patterns
            if data.str.match(r'^\d+$', na=False).sum() > len(data) * 0.8:
                profile.common_patterns.append({
                    'pattern': 'numeric_string',
                    'matches': data.str.match(r'^\d+$', na=False).sum(),
                    'percentage': (data.str.match(r'^\d+$', na=False).sum() / len(data)) * 100,
                    'description': 'Numeric values stored as strings'
                })
            
            # Analyze date patterns
            date_patterns = [
                (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
                (r'^\d{2}/\d{2}/\d{4}$', 'MM/DD/YYYY'),
                (r'^\d{2}-\d{2}-\d{4}$', 'MM-DD-YYYY')
            ]
            
            for pattern, description in date_patterns:
                matches = data.str.match(pattern, na=False).sum()
                if matches > len(data) * 0.5:
                    profile.common_patterns.append({
                        'pattern': 'date_format',
                        'matches': matches,
                        'percentage': (matches / len(data)) * 100,
                        'description': f'Date format: {description}'
                    })
                    break
            
        except Exception as e:
            logger.warning(f"Error in custom pattern analysis: {e}")
    
    def detect_anomalies(self, profile: DataProfile, data: pd.Series):
        """Detect anomalies in the data."""
        try:
            anomalies = []
            
            # For numeric data, use statistical methods
            if pd.api.types.is_numeric_dtype(data):
                # Z-score based outliers
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > 3]
                if len(outliers) > 0:
                    anomalies.append({
                        'type': 'statistical_outliers',
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(data)) * 100,
                        'description': 'Values with Z-score > 3',
                        'examples': outliers.head(5).tolist()
                    })
                
                # IQR based outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
                if len(outliers_iqr) > 0:
                    anomalies.append({
                        'type': 'iqr_outliers',
                        'count': len(outliers_iqr),
                        'percentage': (len(outliers_iqr) / len(data)) * 100,
                        'description': 'Values outside 1.5*IQR from quartiles',
                        'examples': outliers_iqr.head(5).tolist()
                    })
            
            # For text data, look for unusual patterns
            elif pd.api.types.is_string_dtype(data):
                data_str = data.astype(str)
                
                # Unusual length values
                lengths = data_str.str.len()
                length_outliers = data_str[np.abs(stats.zscore(lengths)) > 2]
                if len(length_outliers) > 0:
                    anomalies.append({
                        'type': 'length_outliers',
                        'count': len(length_outliers),
                        'percentage': (len(length_outliers) / len(data)) * 100,
                        'description': 'Values with unusual string length',
                        'examples': length_outliers.head(5).tolist()
                    })
            
            profile.anomalies = anomalies
            
        except Exception as e:
            logger.warning(f"Error in anomaly detection for {profile.column_name}: {e}")
    
    def generate_business_insights(self, profile: DataProfile, data: pd.Series):
        """Generate business insights about the column."""
        try:
            column_name_lower = profile.column_name.lower()
            
            # PII detection
            pii_indicators = ['email', 'phone', 'ssn', 'social', 'address', 'name']
            if any(indicator in column_name_lower for indicator in pii_indicators):
                profile.potential_pii = True
            
            # Check patterns for PII
            for pattern_info in profile.common_patterns:
                if pattern_info['pattern'] in ['email', 'phone', 'ssn', 'credit_card']:
                    profile.potential_pii = True
                    break
            
            # Key column detection
            key_indicators = ['id', '_id', 'key', 'code', 'number']
            if any(indicator in column_name_lower for indicator in key_indicators):
                profile.potential_key = True
            
            # High uniqueness also suggests key column
            if profile.unique_percentage > 95:
                profile.potential_key = True
            
            # Data classification
            if profile.potential_pii:
                profile.data_classification = 'sensitive'
            elif profile.potential_key:
                profile.data_classification = 'identifier'
            elif 'amount' in column_name_lower or 'price' in column_name_lower or 'cost' in column_name_lower:
                profile.data_classification = 'financial'
            elif 'date' in column_name_lower or 'time' in column_name_lower:
                profile.data_classification = 'temporal'
            else:
                profile.data_classification = 'general'
            
        except Exception as e:
            logger.warning(f"Error in business insights generation: {e}")
    
    def calculate_column_quality_score(self, profile: DataProfile) -> float:
        """Calculate overall quality score for a column."""
        score = 1.0
        
        # Penalize high null percentage
        if profile.null_percentage > 50:
            score -= 0.3
        elif profile.null_percentage > 20:
            score -= 0.1
        
        # Penalize low uniqueness in key columns
        if profile.potential_key and profile.unique_percentage < 95:
            score -= 0.2
        
        # Penalize format inconsistency
        if profile.format_consistency is not None and profile.format_consistency < 0.8:
            score -= 0.1
        
        # Penalize high anomaly percentage
        total_anomalies = sum(anomaly['count'] for anomaly in profile.anomalies)
        if total_anomalies > 0:
            anomaly_percentage = (total_anomalies / profile.total_count) * 100
            if anomaly_percentage > 10:
                score -= 0.2
            elif anomaly_percentage > 5:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def analyze_table_relationships(self, table_profile: TableProfile):
        """Analyze potential relationships within the table."""
        # Identify potential primary keys
        for col_name, col_profile in table_profile.column_profiles.items():
            if col_profile.potential_key and col_profile.unique_percentage > 95:
                table_profile.potential_keys.append(col_name)
        
        # Identify potential foreign keys
        for col_name, col_profile in table_profile.column_profiles.items():
            if col_name.endswith('_id') and not col_profile.potential_key:
                referenced_table = col_name[:-3]  # Remove '_id' suffix
                table_profile.potential_foreign_keys.append({
                    'column': col_name,
                    'referenced_table': referenced_table,
                    'confidence': 0.7
                })
    
    def analyze_table_quality(self, table_profile: TableProfile):
        """Analyze overall table quality."""
        if not table_profile.column_profiles:
            return
        
        # Calculate completeness score
        completeness_scores = []
        for col_profile in table_profile.column_profiles.values():
            completeness = 1.0 - (col_profile.null_percentage / 100)
            completeness_scores.append(completeness)
        
        table_profile.completeness_score = sum(completeness_scores) / len(completeness_scores)
        
        # Calculate consistency score
        consistency_scores = []
        for col_profile in table_profile.column_profiles.values():
            if col_profile.format_consistency is not None:
                consistency_scores.append(col_profile.format_consistency)
        
        if consistency_scores:
            table_profile.consistency_score = sum(consistency_scores) / len(consistency_scores)
        else:
            table_profile.consistency_score = 0.8  # Default if no consistency data
        
        # Calculate overall quality score
        quality_scores = [col_profile.quality_score for col_profile in table_profile.column_profiles.values()]
        table_profile.overall_quality_score = sum(quality_scores) / len(quality_scores)
    
    def generate_table_recommendations(self, table_profile: TableProfile):
        """Generate recommendations for the table."""
        # Partitioning recommendations
        date_columns = [
            col_name for col_name, col_profile in table_profile.column_profiles.items()
            if col_profile.data_classification == 'temporal'
        ]
        if date_columns and table_profile.total_rows > 1000000:
            table_profile.recommended_partitioning.extend(date_columns)
        
        # Indexing recommendations
        key_columns = table_profile.potential_keys + [
            fk['column'] for fk in table_profile.potential_foreign_keys
        ]
        table_profile.recommended_indexing.extend(key_columns)
        
        # Quality issues
        low_quality_columns = [
            col_name for col_name, col_profile in table_profile.column_profiles.items()
            if col_profile.quality_score < 0.7
        ]
        if low_quality_columns:
            table_profile.table_level_issues.append(
                f"Low quality columns requiring attention: {', '.join(low_quality_columns)}"
            )
    
    def describe_date_range(self, date_range: timedelta) -> str:
        """Describe a date range in human-readable terms."""
        days = date_range.days
        
        if days < 7:
            return f"{days} days"
        elif days < 30:
            return f"{days // 7} weeks"
        elif days < 365:
            return f"{days // 30} months"
        else:
            return f"{days // 365} years"
    
    def analyze_temporal_patterns(self, data: pd.Series) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in datetime data."""
        patterns = []
        
        try:
            # Day of week pattern
            dow_counts = data.dt.dayofweek.value_counts().sort_index()
            if dow_counts.std() / dow_counts.mean() > 0.5:  # Significant variation
                patterns.append({
                    'type': 'day_of_week',
                    'description': 'Significant variation by day of week',
                    'data': dow_counts.to_dict()
                })
            
            # Hour of day pattern (if time info available)
            if data.dt.hour.nunique() > 1:
                hour_counts = data.dt.hour.value_counts().sort_index()
                if hour_counts.std() / hour_counts.mean() > 0.5:
                    patterns.append({
                        'type': 'hour_of_day',
                        'description': 'Significant variation by hour of day',
                        'data': hour_counts.to_dict()
                    })
            
            # Monthly pattern
            month_counts = data.dt.month.value_counts().sort_index()
            if month_counts.std() / month_counts.mean() > 0.3:
                patterns.append({
                    'type': 'monthly',
                    'description': 'Seasonal patterns detected',
                    'data': month_counts.to_dict()
                })
            
        except Exception as e:
            logger.warning(f"Error in temporal pattern analysis: {e}")
        
        return patterns

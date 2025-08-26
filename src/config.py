"""Configuration management for DataModeling-AI."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger


class OpenAIConfig(BaseModel):
    """OpenAI configuration."""
    api_key: str
    model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 2000


class ClaudeConfig(BaseModel):
    """Claude configuration."""
    api_key: str
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.3
    max_tokens: int = 4000


class DataLakeConfig(BaseModel):
    """Data lake connection configuration."""
    host: str
    port: int
    username: str
    password: Optional[str] = None
    database: Optional[str] = None
    catalog: Optional[str] = None
    schema: Optional[str] = None


class AnalysisConfig(BaseModel):
    """Analysis configuration."""
    max_sample_rows: int = 10000
    profiling_timeout: int = 300
    max_concurrent_tables: int = 10


class AgentConfig(BaseModel):
    """AI agent configuration."""
    enabled: bool = True
    max_depth: Optional[int] = None
    confidence_threshold: Optional[float] = None
    statistical_analysis: Optional[bool] = None
    data_quality_checks: Optional[bool] = None
    include_performance_tips: Optional[bool] = None
    include_governance_recommendations: Optional[bool] = None


class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = True
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    ttl_hours: int = 24


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/datamodeling_ai.log"


class Config(BaseModel):
    """Main configuration class."""
    openai: Optional[OpenAIConfig] = None
    claude: Optional[ClaudeConfig] = None
    llm_provider: str = "claude"  # "openai" or "claude"
    data_lakes: Dict[str, DataLakeConfig]
    analysis: AnalysisConfig = AnalysisConfig()
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()


class ConfigManager:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config: Optional[Config] = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "config.yaml",
            "config.yml",
            os.path.expanduser("~/.datamodeling-ai/config.yaml"),
            "/etc/datamodeling-ai/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no config file found, use example config
        example_path = Path(__file__).parent.parent / "config.example.yaml"
        if example_path.exists():
            logger.warning(f"No config file found, using example config: {example_path}")
            return str(example_path)
        
        raise FileNotFoundError("No configuration file found")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Override with environment variables
            self._override_with_env(config_data)
            
            self.config = Config(**config_data)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _override_with_env(self, config_data: Dict[str, Any]):
        """Override configuration with environment variables."""
        env_mappings = {
            'OPENAI_API_KEY': ['openai', 'api_key'],
            'CLAUDE_API_KEY': ['claude', 'api_key'],
            'LLM_PROVIDER': ['llm_provider'],
            'HIVE_HOST': ['data_lakes', 'hive', 'host'],
            'HIVE_PORT': ['data_lakes', 'hive', 'port'],
            'HIVE_USERNAME': ['data_lakes', 'hive', 'username'],
            'HIVE_PASSWORD': ['data_lakes', 'hive', 'password'],
            'TRINO_HOST': ['data_lakes', 'trino', 'host'],
            'TRINO_PORT': ['data_lakes', 'trino', 'port'],
            'TRINO_USERNAME': ['data_lakes', 'trino', 'username'],
            'TRINO_PASSWORD': ['data_lakes', 'trino', 'password'],
            'PRESTO_HOST': ['data_lakes', 'presto', 'host'],
            'PRESTO_PORT': ['data_lakes', 'presto', 'port'],
            'PRESTO_USERNAME': ['data_lakes', 'presto', 'username'],
            'REDIS_HOST': ['cache', 'host'],
            'REDIS_PORT': ['cache', 'port'],
            'LOG_LEVEL': ['logging', 'level'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(config_data, config_path, value)
    
    def _set_nested_value(self, data: Dict[str, Any], path: list, value: str):
        """Set nested dictionary value."""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        final_key = path[-1]
        if final_key in ['port']:
            value = int(value)
        elif final_key in ['temperature']:
            value = float(value)
        elif final_key in ['enabled']:
            value = value.lower() in ('true', '1', 'yes')
        
        current[final_key] = value
    
    def get_config(self) -> Config:
        """Get the current configuration."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        return self.config
    
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_manager
    if not _config_manager:
        _config_manager = ConfigManager()
    return _config_manager.get_config()


def init_config(config_path: Optional[str] = None):
    """Initialize configuration with a specific path."""
    global _config_manager
    _config_manager = ConfigManager(config_path)

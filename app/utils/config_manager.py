from typing import Dict, Any, Optional
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Default configurations as per spec
DEFAULT_CONFIG = {
    "model_settings": {
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 1024
    },
    "retrieval_settings": {
        "num_chunks": 5,
        "similarity_threshold": 0.7,
        "hybrid_search_weight": 0.5,
        "reranking_enabled": True
    },
    "prompts": {
        "system_prompt": "You are a helpful AI assistant that answers questions based on the provided context. Always be clear, accurate, and cite your sources.",
        "retrieval_prompt": "Based on the following context, please provide a relevant and accurate answer to the question. Context: {context}\n\nQuestion: {question}",
        "reranking_prompt": "Rate the relevance of this passage to the query on a scale of 0-10. Consider semantic similarity and information value.\n\nQuery: {query}\nPassage: {passage}"
    },
    "advanced_settings": {
        "max_retries": 3,
        "cache_ttl": 3600,
        "debug_mode": False
    }
}

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigManager:
    """Manages per-schema configurations in the database."""
    
    def __init__(self, db_params: Dict[str, str]):
        """Initialize the configuration manager.
        
        Args:
            db_params: Database connection parameters
        """
        self.db_params = db_params
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.db_params)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration data.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            # Check required sections
            required_sections = ["model_settings", "retrieval_settings", "prompts", "advanced_settings"]
            for section in required_sections:
                if section not in config:
                    raise ConfigValidationError(f"Missing required section: {section}")
            
            # Validate model settings
            model_settings = config["model_settings"]
            if not 0 <= model_settings.get("temperature", 0) <= 1:
                raise ConfigValidationError("Temperature must be between 0 and 1")
            if not 1 <= model_settings.get("top_k", 0) <= 100:
                raise ConfigValidationError("Top K must be between 1 and 100")
            if not 0 <= model_settings.get("top_p", 0) <= 1:
                raise ConfigValidationError("Top P must be between 0 and 1")
            if not 100 <= model_settings.get("max_output_tokens", 0) <= 2048:
                raise ConfigValidationError("Max output tokens must be between 100 and 2048")
            
            # Validate retrieval settings
            retrieval_settings = config["retrieval_settings"]
            if not 1 <= retrieval_settings.get("num_chunks", 0) <= 20:
                raise ConfigValidationError("Number of chunks must be between 1 and 20")
            if not 0 <= retrieval_settings.get("similarity_threshold", 0) <= 1:
                raise ConfigValidationError("Similarity threshold must be between 0 and 1")
            if not 0 <= retrieval_settings.get("hybrid_search_weight", 0) <= 1:
                raise ConfigValidationError("Hybrid search weight must be between 0 and 1")
            
            # Validate prompts
            prompts = config["prompts"]
            for key in ["system_prompt", "retrieval_prompt", "reranking_prompt"]:
                if not prompts.get(key):
                    raise ConfigValidationError(f"Missing required prompt: {key}")
            
            # Validate advanced settings
            advanced_settings = config["advanced_settings"]
            if not 1 <= advanced_settings.get("max_retries", 0) <= 5:
                raise ConfigValidationError("Max retries must be between 1 and 5")
            if not 60 <= advanced_settings.get("cache_ttl", 0) <= 86400:
                raise ConfigValidationError("Cache TTL must be between 60 and 86400 seconds")
                
        except KeyError as e:
            raise ConfigValidationError(f"Missing required configuration key: {e}")
    
    def _schema_exists(self, schema_name: str) -> bool:
        """Check if a schema exists in the database.
        
        Args:
            schema_name: Name of the schema to check
            
        Returns:
            bool: True if schema exists, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name = %s
                    """, (schema_name,))
                    return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking schema existence: {e}")
            return False
    
    def get_config(self, schema_name: str) -> Dict[str, Any]:
        """Get configuration for a specific schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Configuration dictionary (defaults if not found)
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT config_data 
                        FROM configurations 
                        WHERE schema_name = %s
                    """, (schema_name,))
                    
                    result = cur.fetchone()
                    return result[0] if result else DEFAULT_CONFIG.copy()
                    
        except Exception as e:
            logger.error(f"Error getting configuration for schema {schema_name}: {e}")
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, schema_name: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration for a specific schema.
        
        Args:
            schema_name: Name of the schema
            config_data: Configuration data to save
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            # Validate schema name format
            if schema_name != "default" and not schema_name.startswith("team_"):
                raise ConfigValidationError("Schema name must start with 'team_' or be 'default'")
            
            # Check schema exists (except for default)
            if schema_name != "default" and not self._schema_exists(schema_name):
                raise ConfigValidationError(f"Schema '{schema_name}' does not exist")
            
            # Validate configuration
            self._validate_config(config_data)
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO configurations (schema_name, config_data, updated_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (schema_name) 
                        DO UPDATE SET 
                            config_data = EXCLUDED.config_data,
                            updated_at = EXCLUDED.updated_at
                    """, (schema_name, Json(config_data), datetime.now()))
                    return True
                    
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving configuration for schema {schema_name}: {e}")
            return False
    
    def reset_config(self, schema_name: str) -> bool:
        """Reset configuration to defaults for a specific schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # For new schemas, just return True as the get_config will return defaults
            if not self._config_exists(schema_name):
                return True
                
            # For existing configs, delete and let get_config handle defaults
            return self.delete_config(schema_name)
            
        except Exception as e:
            logger.error(f"Error resetting configuration for schema {schema_name}: {e}")
            return False
    
    def delete_config(self, schema_name: str) -> bool:
        """Delete configuration for a specific schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM configurations 
                        WHERE schema_name = %s
                    """, (schema_name,))
                    return cur.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Error deleting configuration for schema {schema_name}: {e}")
            return False
    
    def list_schemas(self) -> list[str]:
        """Get list of all valid schema names, including those without configurations.
        
        Returns:
            List of schema names including 'default' and all team schemas
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get all schemas that start with 'team_'
                    cur.execute("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name LIKE 'team_%'
                        ORDER BY schema_name
                    """)
                    team_schemas = [row[0] for row in cur.fetchall()]
                    
                    # Always include 'default' schema first
                    return ['default'] + team_schemas
                    
        except Exception as e:
            logger.error(f"Error listing schemas: {e}")
            return ['default']  # Always return at least default schema
    
    def _config_exists(self, schema_name: str) -> bool:
        """Check if configuration exists for a schema.
        
        Args:
            schema_name: Name of the schema to check
            
        Returns:
            bool: True if configuration exists, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 
                        FROM configurations 
                        WHERE schema_name = %s
                    """, (schema_name,))
                    return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking configuration existence: {e}")
            return False
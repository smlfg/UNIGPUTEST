#!/usr/bin/env python3
"""
Configuration Loader
Loads YAML configs with validation and environment variable support
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Load and validate training configurations"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_path: Path to YAML config file
        """
        if config_path is None:
            # Default to configs/training_config.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "training_config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            dict: Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Replace environment variables
        self.config = self._replace_env_vars(self.config)

        return self.config

    def _replace_env_vars(self, config: Any) -> Any:
        """
        Recursively replace ${ENV_VAR} with environment variables

        Args:
            config: Configuration dictionary or value

        Returns:
            Configuration with environment variables replaced
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace ${VAR} or $VAR with environment variable
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.getenv(var_name, config)
            elif config.startswith("$"):
                var_name = config[1:]
                return os.getenv(var_name, config)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)

        Args:
            key: Configuration key (e.g., "model.name")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        required_keys = [
            "model.name",
            "training.output_dir",
            "dataset.name",
        ]

        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Missing required config key: {key}")

        return True


if __name__ == "__main__":
    # Test config loading
    loader = ConfigLoader()
    config = loader.load()
    loader.validate()

    print("✅ Configuration loaded successfully!")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Batch size: {config['training']['per_device_train_batch_size']}")

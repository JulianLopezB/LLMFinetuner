import yaml
import os
from typing import Any, Dict

class Config:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load the configuration file from the specified path using YAML.

        Args:
            config_path (str): The path to the configuration YAML file.

        Returns:
            Dict[str, Any]: A dictionary containing configuration parameters.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"The configuration file was not found at {config_path}.")
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the configuration dictionary.

        Args:
            key (str): The key in the configuration dictionary.
            default (Any): The default value to return if the key is not found.

        Returns:
            Any: The value from the configuration dictionary or the default value.
        """
        # Access nested configurations easily
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default) if isinstance(value, dict) else default
        return value

# Example usage:
# config = Config('path/to/your/finetuning_config.yaml')
# model_name = config.get('model.name')
# dataset_path = config.get('dataset.path')

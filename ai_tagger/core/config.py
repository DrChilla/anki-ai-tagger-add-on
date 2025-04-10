import os
import json
from typing import Dict, Any, Optional

from aqt import mw
from ..utils.constants import DEFAULT_CONFIG
from ..utils.helpers import log_to_file


class ConfigManager:
    """
    Manages configuration settings for the AI Tagger add-on.
    
    This class handles loading, saving, and providing access to configuration settings.
    It ensures configuration validity and provides default values when needed.
    Implements the singleton pattern to ensure only one instance exists.
    """
    
    _instance = None
    
    def __init__(self):
        """Initialize the ConfigManager and load the addon config."""
        self.config = {}
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
        self.load_config()
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """
        Get the singleton instance of ConfigManager.
        
        Returns:
            ConfigManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance
    
    def load_config(self) -> None:
        """
        Load configuration from Anki addon manager or file.
        Falls back to default configuration if needed.
        """
        try:
            # First try to get config from Anki
            if mw and mw.addonManager:
                addon_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                self.config = mw.addonManager.getConfig(addon_dir) or {}
            
            # If Anki config is not available, try to load from file
            if not self.config and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Ensure all required config sections exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = value
                elif isinstance(value, dict):
                    # Ensure nested config items exist
                    for subkey, subvalue in value.items():
                        if key in self.config and isinstance(self.config[key], dict) and subkey not in self.config[key]:
                            self.config[key][subkey] = subvalue
            
            log_to_file("Configuration loaded successfully")
        except Exception as e:
            log_to_file(f"Error loading configuration: {str(e)}")
            self.config = DEFAULT_CONFIG.copy()
    
    def save_config(self) -> None:
        """Save the current configuration to both file and Anki addon manager."""
        try:
            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            # Update Anki config if available
            if mw and mw.addonManager:
                addon_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                mw.addonManager.writeConfig(addon_dir, self.config)
                
            log_to_file("Configuration saved successfully")
        except Exception as e:
            log_to_file(f"Error saving configuration: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.
        
        Returns:
            Dict[str, Any]: The configuration dictionary
        """
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a simple config value by key.
        
        Args:
            key (str): The configuration key
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: The configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a simple configuration value by key.
        
        Args:
            key (str): The configuration key
            value (Any): The value to set
        """
        self.config[key] = value
        self.save_config()


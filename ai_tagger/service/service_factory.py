from typing import Dict, List, Any, Optional, Union, Type

# Comment out other services to focus only on Gemini
# from .claude import ClaudeService
# from .openai import OpenAIService
from .gemini import GeminiService
from ..core.config import ConfigManager
from ..utils.constants import ServiceType
from ..utils.helpers import log_to_file
from .openAICompatible import OpenAICompatibleService


class ServiceFactory:
    """
    Factory class for creating AI service instances based on configuration.
    """
    
    # Registry of available service classes - only Gemini enabled
    _service_registry = {
        # ServiceType.ANTHROPIC_CLAUDE.value: ClaudeService,
        # ServiceType.OPENAI.value: OpenAIService,
        ServiceType.GEMINI.value: GeminiService,
    }
    @classmethod
    def create_service(cls, service_type: Union[str, ServiceType], config: Dict[str, Any]) -> Optional[OpenAICompatibleService]:
        """
        Create and return an instance of the requested AI service.
        
        Args:
            service_type: The type of service to create (from ServiceType constants)
            config: Dictionary containing configuration for the service, including API key
            
        Returns:
            An instance of the requested service or None if the service type is invalid
        """
        try:
            # Convert ServiceType enum to string value if needed
            service_key = service_type.value if isinstance(service_type, ServiceType) else service_type
            # Always use Gemini regardless of requested service_type
            service_key = ServiceType.GEMINI.value
            log_to_file(f"Using Gemini service (2.0 Flash) regardless of requested service: {service_type}")
            
            # if service_key not in cls._service_registry:
            #     log_to_file(f"Unsupported service type: {service_key}")
            #     return None
            service_class = cls._service_registry[service_key]
            # Extract api_key from config, defaulting to empty string if not present
            api_key = config.get('api_key', '')
            
            # Create a copy of the config to avoid modifying the original
            # Create a copy of the config to avoid modifying the original
            service_config = config.copy()
            
            # Handle Gemini service differently since it expects a single config parameter
            if service_key == ServiceType.GEMINI.value:
                # Put api_key back into config
                service_config['api_key'] = api_key
                return service_class(config=service_config)
            else:
                # For other services that expect api_key as separate parameter
                # Remove api_key if it exists, as we're explicitly passing it
                if 'api_key' in service_config:
                    service_config.pop('api_key')
                return service_class(api_key=api_key, **service_config)
        except Exception as e:
            log_to_file(f"Error creating service instance: {str(e)}")
            return None
            
    @staticmethod
    def create_service_from_config(config_manager: ConfigManager, service_type: Optional[ServiceType] = None) -> Optional[OpenAICompatibleService]:
        """
        Create and return an instance of the specified AI service using a ConfigManager.
        
        Args:
            config_manager: The configuration manager instance
            service_type: The type of service to create (optional, will use config if not specified)
            
        Returns:
            An instance of the specified AI service, or None if the service couldn't be created
        """
        if service_type is None:
            # Use the service type from the configuration
            service_config = config_manager.get_config().get('service', {})
            service_type_str = service_config.get('type', '')
            try:
                service_type = ServiceType(service_type_str)
            except ValueError:
                log_to_file(f"Invalid service type: {service_type_str}")
                return None
        
        # Get API key from config
        service_config = config_manager.get_config().get('service', {})
        api_key = service_config.get('api_key', '')
        
        # Prepare additional kwargs from config
        kwargs = {}
        for key, value in service_config.items():
            if key not in ['type', 'api_key']:
                kwargs[key] = value
                
        return ServiceFactory.create_service(service_type, {'api_key': api_key, **kwargs})
    
    @classmethod
    def register_service(cls, service_type: Union[str, ServiceType], service_class: Type[OpenAICompatibleService]) -> None:
        """
        Register a new service class with the factory.
        
        Args:
            service_type: The identifier for the service type (can be string or ServiceType enum)
            service_class: The service class to register
        """
        # Ensure we store the string value as the key
        key = service_type.value if isinstance(service_type, ServiceType) else service_type
        cls._service_registry[key] = service_class
    @classmethod
    def get_available_services(cls) -> Dict[str, Type[OpenAICompatibleService]]:
        """
        Get a dictionary of all registered services.
        
        Returns:
            Dictionary mapping service types to service classes
        """
        return cls._service_registry.copy()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[OpenAICompatibleService]:
        """
        Create a service instance from a configuration dictionary.
        
        Args:
            config: Dictionary containing service configuration
                   Must include 'service_type' and 'api_key' keys
                   
        Returns:
            Service instance or None if creation failed
        """
        if not config or 'service_type' not in config or 'api_key' not in config:
            log_to_file("Invalid service configuration: missing required fields")
            return None
            
        service_type = config.pop('service_type')
        
        return cls.create_service(service_type, config)

    @classmethod
    def get_service_for_config(cls, config: Dict[str, Any]) -> Optional[OpenAICompatibleService]:
        """
        Create a service instance from a full configuration dictionary.
        
        This method takes a complete configuration dictionary (as would be used by ConfigManager)
        and extracts the service type and configuration to create an appropriate service instance.
        
        Args:
            config: Complete configuration dictionary containing service settings
                   Expected to have a 'service' key with nested configuration
                   
        Returns:
            Service instance or None if creation failed
        """
        if not config or 'service' not in config:
            log_to_file("Invalid configuration: missing 'service' section")
            return None
            
        service_config = config['service']
        
        if 'type' not in service_config:
            log_to_file("Invalid service configuration: missing 'type' field")
            return None
            
        service_type_str = service_config['type']
        
        try:
            service_type = ServiceType(service_type_str)
        except ValueError:
            log_to_file(f"Invalid service type: {service_type_str}")
            return None
            
        # Create a copy of the service config without the type field
        config_copy = {k: v for k, v in service_config.items() if k != 'type'}
        
        return cls.create_service(service_type, config_copy)

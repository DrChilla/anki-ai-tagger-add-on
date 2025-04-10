from abc import ABC, abstractmethod
import json
import logging
from typing import Dict, List, Any, Optional, Union

from ..utils.helpers import log_to_file


class OpenAICompatibleService(ABC):
    """
    Abstract base class defining interface for OpenAI-compatible API services.
    Implementations should provide concrete implementations for specific services
    like OpenAI GPT, Anthropic Claude, etc.
    """
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """
        Initialize the OpenAI compatible service.
        
        Args:
            api_key: API key for the service
            api_base: Base URL for API endpoints
            model: Default model to use for the service
        """
        self.api_key = api_key
        self.api_base = api_base
        self._model = model
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion using the AI service.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters specific to the service
            
        Returns:
            Dictionary containing the completion response
        """
        pass
    
    @abstractmethod
    def embedding(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for the given text using the AI service.
        
        Args:
            text: Text or list of texts to generate embeddings for
            **kwargs: Additional parameters specific to the service
            
        Returns:
            Dictionary containing the embedding response
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from the service.
        
        Returns:
            List of model identifiers available for this service
        """
        pass
    
    @property
    def model(self) -> str:
        """
        Get the currently selected model.
        
        Returns:
            Model identifier string
        """
        return self._model
    
    @model.setter
    def model(self, model: str) -> None:
        """
        Set the model to use.
        
        Args:
            model: Model identifier to use
        """
        self._model = model
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is properly set and working.
        
        Returns:
            True if API key is valid, False otherwise
        """
        if not self.api_key:
            return False
        
        try:
            # Make a minimal API call to check if the key works
            result = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            log_to_file(f"API key validation failed: {str(e)}")
            return False
    
    def format_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format a prompt and optional system prompt into the messages format expected by the chat API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to guide the model
            
        Returns:
            List of message dictionaries in the format expected by the chat completion API
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def log_request(self, endpoint: str, params: Dict[str, Any]) -> None:
        """
        Log an API request for debugging purposes.
        
        Args:
            endpoint: The API endpoint being called
            params: The parameters being sent to the API
        """
        # Don't log the actual API key
        safe_params = params.copy()
        if 'api_key' in safe_params:
            safe_params['api_key'] = '***REDACTED***'
        
        log_to_file(
            f"API Request to {endpoint}: {json.dumps(safe_params, indent=2)}"
        )
    
    def log_response(self, endpoint: str, response: Dict[str, Any]) -> None:
        """
        Log an API response for debugging purposes.
        
        Args:
            endpoint: The API endpoint that was called
            response: The response received from the API
        """
        log_to_file(
            f"API Response from {endpoint}: {json.dumps(response, indent=2)}"
        )
    
    def is_compatible(self) -> bool:
        """
        Check if this service is properly configured and ready to use.
        
        Returns:
            True if the service is ready, False otherwise
        """
        return self.api_key is not None and self.model is not None


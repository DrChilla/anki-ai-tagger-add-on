import json
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI

from ..utils.constants import LOG_PREFIX, OPENAI_MODELS
from ..utils.helpers import log_to_file
from .openAICompatible import OpenAICompatibleService


class OpenAIService(OpenAICompatibleService):
    """
    Service class for handling OpenAI API interactions.
    Implements the OpenAICompatibleService interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI service with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                   including api_key, model, temperature, etc.
        """
        # Initialize base class
        super().__init__(
            api_key=config.get('api_key', ''),
            api_base=config.get('api_base', None),
            model=config.get('model', 'gpt-3.5-turbo')
        )
        
        # Store additional configuration
        self.temperature = float(config.get('temperature', 0.7))
        self.max_tokens = int(config.get('max_tokens', 500))
        
        # Validate the model
        if self._model not in OPENAI_MODELS:
            log_to_file(f"{LOG_PREFIX} Warning: Model {self._model} not in known OpenAI models. Using anyway.")
        
        # Validate API key is set
        if not self.api_key:
            log_to_file(f"{LOG_PREFIX} Error: OpenAI API key not set")
            raise ValueError("OpenAI API key not set")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    
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

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Example: [{"role": "system", "content": "You are a helpful assistant"},
                              {"role": "user", "content": "Generate tags for this content..."}]
            **kwargs: Additional parameters to pass to the API
        
        Returns:
            Dict containing the response from the API
            Example: {"content": "Generated tags: #history #world-war-2 #military"}
        
        Raises:
            Exception: If the API request fails
        """
        try:
            log_to_file(f"{LOG_PREFIX} Sending request to OpenAI API: {json.dumps({'model': self._model, 'messages': messages}, indent=2)}")
            
            # Override defaults with any kwargs provided
            temp = kwargs.get('temperature', self.temperature)
            max_tok = kwargs.get('max_tokens', self.max_tokens)
            
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            
            log_to_file(f"{LOG_PREFIX} OpenAI API response received")
            
            # Extract the content from the response
            # Handle both dictionary-style and object-style responses
            try:
                # Try accessing as an object (dot notation)
                content = response.choices[0].message.content
            except AttributeError:
                # Try accessing as a dictionary
                if isinstance(response, dict) and 'choices' in response:
                    content = response['choices'][0]['message']['content']
                else:
                    error_msg = f"{LOG_PREFIX} Unexpected response format: {response}"
                    log_to_file(error_msg)
                    return {"content": "Error: Unexpected response format", "error": True}
            
            return {"content": content, "error": False}
            
        except Exception as e:
            error_msg = f"{LOG_PREFIX} Error in OpenAI completion: {str(e)}"
            log_to_file(error_msg)
            return {"content": f"Error: {str(e)}", "error": True}
    
    def embedding(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for the given text using the OpenAI API.
        
        Args:
            text: Text or list of texts to generate embeddings for
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the embedding response
            
        Raises:
            Exception: If the API request fails
        """
        try:
            log_to_file(f"{LOG_PREFIX} Generating embeddings for text")
            
            # Convert single string to list if needed
            input_texts = [text] if isinstance(text, str) else text
            
            # Get model from kwargs or use ada embedding model by default
            embedding_model = kwargs.get('model', 'text-embedding-ada-002')
            
            response = self.client.embeddings.create(
                model=embedding_model,
                input=input_texts
            )
            
            log_to_file(f"{LOG_PREFIX} Embedding response received")
            
            # Format response in a similar way to chat completions
            try:
                # Try accessing as an object (dot notation)
                data = response.data
                model = response.model
            except AttributeError:
                # Try accessing as a dictionary
                if isinstance(response, dict):
                    data = response.get('data', [])
                    model = response.get('model', self._model)
                else:
                    error_msg = f"{LOG_PREFIX} Unexpected embedding response format: {response}"
                    log_to_file(error_msg)
                    return {"error": True, "message": "Unexpected response format"}
            
            return {
                "data": data,
                "model": model,
                "error": False
            }
            
        except Exception as e:
            error_msg = f"{LOG_PREFIX} Error generating embeddings: {str(e)}"
            log_to_file(error_msg)
            return {"error": True, "message": str(e)}
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from OpenAI.
        
        Returns:
            List of model identifiers available for OpenAI
        """
        try:
            log_to_file(f"{LOG_PREFIX} Getting available models from OpenAI")
            response = self.client.models.list()
            
            # Extract model IDs from response
            # Extract model IDs from response
            try:
                # Try accessing as an object (dot notation)
                models = [model.id for model in response.data]
            except AttributeError:
                # Try accessing as a dictionary
                if isinstance(response, dict) and 'data' in response:
                    models = [model.get('id') if isinstance(model, dict) else model.id
                             for model in response['data']]
                else:
                    log_to_file(f"{LOG_PREFIX} Unexpected models response format, using defaults")
                    return list(OPENAI_MODELS.keys())
            # Filter to include only the models we're interested in
            filtered_models = [model for model in models if any(
                model.startswith(prefix) for prefix in ['gpt-', 'text-embedding-', 'text-davinci-']
            )]
            
            log_to_file(f"{LOG_PREFIX} Retrieved {len(filtered_models)} available models")
            return filtered_models
            
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} Error retrieving models: {str(e)}")
            # Return our predefined list in case of error
            return OPENAI_MODELS
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is properly set and working.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Simple test message
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test connection. Reply with 'Connection successful'."}
            ]
            
            response = self.chat_completion(test_messages, max_tokens=10)
            
            if response.get("error", False):
                log_to_file(f"{LOG_PREFIX} API key validation failed: {response.get('content')}")
                return False
                
            log_to_file(f"{LOG_PREFIX} API key validation successful")
            return True
            
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} API key validation error: {str(e)}")
            return False
    
    # Alias for backward compatibility
    validate_connection = validate_api_key
    
    def is_compatible(self) -> bool:
        """
        Check if this service is properly configured and ready to use.
        
        Returns:
            True if the service is ready, False otherwise
        """
        return self.api_key is not None and self.model is not None

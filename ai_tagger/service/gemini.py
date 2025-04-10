import json
import requests
import logging
from typing import Dict, List, Any, Optional, Union

from ..utils.constants import LOG_PREFIX, GEMINI_MODELS
from ..utils.helpers import log_to_file
from .openAICompatible import OpenAICompatibleService


class GeminiService(OpenAICompatibleService):
    """
    Service class for handling Google's Gemini API interactions.
    Implements the OpenAICompatibleService interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gemini service with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                   including api_key, model, temperature, etc.
        """
        # Initialize base class
        super().__init__(
            api_key=config.get('api_key', ''),
            api_base=config.get('api_base', 'https://generativelanguage.googleapis.com/v1'),
            model=config.get('model', 'gemini-2.0-flash')
        )
        
        # Store additional configuration
        self.temperature = float(config.get('temperature', 0.7))
        self.max_tokens = int(config.get('max_tokens', 500))
        self.timeout = int(config.get('timeout', 30))
        
        # Validate the model
        if self._model not in GEMINI_MODELS:
            log_to_file(f"{LOG_PREFIX} Warning: Model {self._model} not in known Gemini models. Using anyway.")
        
        # Validate API key is set
        if not self.api_key:
            log_to_file(f"{LOG_PREFIX} Error: Gemini API key not set")
            raise ValueError("Gemini API key not set")
        
        self.logger = logging.getLogger(__name__)
    
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
        Generate a chat completion using Gemini API.
        
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
            # Convert OpenAI-style messages to Gemini format
            gemini_content = self._convert_to_gemini_format(messages)
            
            # Build the request URL
            url = f"{self.api_base}/models/{self._model}:generateContent"
            
            # Add API key as query parameter
            url = f"{url}?key={self.api_key}"
            
            # Set up the request headers
            headers = {
                "Content-Type": "application/json"
            }
            
            # Override defaults with any kwargs provided
            temp = kwargs.get('temperature', self.temperature)
            max_tok = kwargs.get('max_tokens', self.max_tokens)
            
            # Build the request payload
            payload = {
                "contents": gemini_content,
                "generationConfig": {
                    "temperature": temp,
                    "maxOutputTokens": max_tok,
                    "topP": kwargs.get('top_p', 0.95),
                    "topK": kwargs.get('top_k', 40)
                }
            }
            
            # Log the request (without API key)
            log_to_file(f"{LOG_PREFIX} Sending request to Gemini API with model {self._model}")
            
            # Make the API request
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', self.timeout)
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Log receipt of response 
            log_to_file(f"{LOG_PREFIX} Gemini API response received")
            
            # Convert to OpenAI-compatible format
            openai_compatible_response = self._convert_to_openai_format(response_data)
            
            return openai_compatible_response
            
        except Exception as e:
            error_msg = f"{LOG_PREFIX} Error in Gemini completion: {str(e)}"
            log_to_file(error_msg)
            return {"content": f"Error: {str(e)}", "error": True}
    
    def embedding(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for the given text using the Gemini API.
        Note: Gemini may not support embeddings natively, so this is a placeholder.
        
        Args:
            text: Text or list of texts to generate embeddings for
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the embedding response
            
        Raises:
            Exception: If the API request fails
        """
        error_msg = f"{LOG_PREFIX} Embedding functionality not currently supported by Gemini API implementation"
        log_to_file(error_msg)
        return {"error": True, "message": "Embedding functionality not supported"}
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Gemini.
        Currently returns a predefined list.
        
        Returns:
            List of model identifiers available for Gemini
        """
        return list(GEMINI_MODELS.keys())
    
    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            
        Returns:
            List of message dictionaries in Gemini format
        """
        gemini_content = []
        current_role = "user"
        current_parts = []
        system_instructions = []
        
        # Extract system messages first
        for message in messages:
            if message["role"] == "system":
                system_instructions.append(message["content"])
        
        # Process all messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Skip system messages (we added them to system_instructions)
            if role == "system":
                continue
                
            # If role changed, add the previous message
            if role != current_role and current_parts:
                gemini_content.append({
                    "role": "user" if current_role == "user" else "model",
                    "parts": current_parts.copy()
                })
                current_parts = []
                
            # Update current role
            current_role = role
            
            # For the first user message, prepend system instructions if any
            if role == "user" and not gemini_content and system_instructions:
                combined_content = "\n\n".join([
                    "System Instructions: " + instr for instr in system_instructions
                ]) + "\n\n" + content
                current_parts.append({"text": combined_content})
            else:
                current_parts.append({"text": content})
        
        # Add the last message if there are parts
        if current_parts:
            gemini_content.append({
                "role": "user" if current_role == "user" else "model",
                "parts": current_parts.copy()
            })
        
        return gemini_content
    
    def _convert_to_openai_format(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini API response to OpenAI-compatible format.
        
        Args:
            gemini_response: Response from Gemini API
            
        Returns:
            Response in OpenAI-compatible format
        """
        try:
            # Extract content from Gemini response (handling both object and dict formats)
            content = ""
            
            try:
                # Try object-style access first
                if hasattr(gemini_response, 'candidates') and len(gemini_response.candidates) > 0:
                    candidate = gemini_response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
                        content = candidate.content.parts[0].text
                elif hasattr(gemini_response, 'promptFeedback') and hasattr(gemini_response.promptFeedback, 'blockReason'):
                    content = f"Content blocked: {gemini_response.promptFeedback.blockReason}"
            except (AttributeError, IndexError, TypeError):
                # Fall back to dictionary access
                if isinstance(gemini_response, dict):
                    if 'candidates' in gemini_response and len(gemini_response['candidates']) > 0:
                        candidate = gemini_response['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                            content = candidate['content']['parts'][0]['text']
                        else:
                            log_to_file(f"{LOG_PREFIX} Unusual candidate structure: {json.dumps(candidate)[:200]}")
                            content = f"Response could not be parsed correctly: {json.dumps(candidate)[:50]}..."
                    elif 'promptFeedback' in gemini_response and 'blockReason' in gemini_response['promptFeedback']:
                        content = f"Content blocked: {gemini_response['promptFeedback']['blockReason']}"
                    else:
                        log_to_file(f"{LOG_PREFIX} Unexpected response structure: {json.dumps(gemini_response)[:200]}")
                        content = f"Response format not recognized"
                else:
                    content = f"Unexpected response type: {type(gemini_response)}"
            
            # Format in OpenAI-compatible way
            return {
                "id": gemini_response.get('name', 'gemini_response') if isinstance(gemini_response, dict) else 'gemini_response',
                "object": "chat.completion",
                "created": 0,  # Gemini doesn't provide a timestamp
                "model": self._model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Gemini doesn't always provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        except Exception as e:
            error_msg = f"{LOG_PREFIX} Error formatting Gemini response: {str(e)}"
            log_to_file(error_msg)
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        }
                    }
                ]
            }
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is properly set and working.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Simple test message
            test_messages = [
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


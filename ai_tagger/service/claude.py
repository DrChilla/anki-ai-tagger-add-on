import json
import logging
from typing import Dict, List, Any, Optional, Union

import anthropic
from anthropic import Anthropic

from .openAICompatible import OpenAICompatibleService
from ..utils.helpers import log_to_file
from ..utils.constants import ANTHROPIC_MODELS
import json
import logging
from typing import Dict, List, Any, Optional, Union

import anthropic
from anthropic import Anthropic

from .openAICompatible import OpenAICompatibleService
from ..utils.helpers import log_to_file


class ClaudeService(OpenAICompatibleService):
    """
    Implementation of the OpenAI-compatible interface for Anthropic's Claude model.
    """

    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """
        Initialize the Claude service.
        
        Args:
            api_key: Anthropic API key
            api_base: Base URL for API endpoints (usually not needed for Anthropic)
            model: Claude model to use (e.g., 'claude-3-opus-20240229')
        """
        super().__init__(api_key, api_base, model)
        
        # Default model if none specified
        if not self.model:
            self.model = "claude-3-opus-20240229"
        
        # Initialize the Anthropic client if we have an API key
        self.client = None
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a chat completion using Claude.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys (OpenAI format)
            **kwargs: Additional parameters for the completion request
            
        Returns:
            Dictionary containing the completion response in OpenAI-compatible format
        """
        if not self.client:
            raise ValueError("Anthropic client not initialized. Please provide a valid API key.")

        # Convert from OpenAI message format to Anthropic message format
        anthropic_messages = self._convert_to_anthropic_messages(messages)
        
        # Set default parameters
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0.7)
        
        # Log the request (without API key)
        self.log_request("chat.completions", {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        })
        
        try:
            # Make the API call to Claude
            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert the response to OpenAI-compatible format
            openai_compatible_response = self._convert_to_openai_format(response)
            
            # Log the response
            self.log_response("chat.completions", openai_compatible_response)
            
            return openai_compatible_response
            
        except Exception as e:
            error_message = f"Claude API request failed: {str(e)}"
            log_to_file(error_message)
            self.logger.error(error_message)
            
            # Return error in a format similar to OpenAI
            return {
                "error": {
                    "message": error_message,
                    "type": "api_error"
                }
            }
    
    def embedding(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        """
        Generate embeddings for the given text using Claude.
        
        Note: If Claude doesn't support native embeddings, this might use a compatible
        approach or return a meaningful error.
        
        Args:
            text: Text or list of texts to generate embeddings for
            **kwargs: Additional parameters for the embedding request
            
        Returns:
            Dictionary containing the embedding response in OpenAI-compatible format
        """
        # As of my knowledge cutoff, Claude may not support native embeddings
        # This is a placeholder implementation
        
        error_message = "Embedding functionality not currently supported by Claude."
        log_to_file(error_message)
        self.logger.error(error_message)
        
        return {
            "error": {
                "message": error_message,
                "type": "unsupported_operation"
            }
        }
    
    def _convert_to_anthropic_messages(self, openai_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI-style messages to Anthropic-compatible format.
        
        Args:
            openai_messages: Messages in OpenAI format (role, content)
            
        Returns:
            Messages in Anthropic format
        """
        anthropic_messages = []
        
        for msg in openai_messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map OpenAI roles to Anthropic roles
            if role == "system":
                # System messages are handled differently in Claude
                anthropic_messages.append({
                    "role": "system",
                    "content": content
                })
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif role == "function":
                # Claude might handle function calls differently
                # For now, we'll skip them or convert to user messages
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Function result: {content}"
                })
        
        return anthropic_messages
    
    def _convert_to_openai_format(self, claude_response) -> Dict[str, Any]:
        """
        Convert Claude API response to OpenAI-compatible format.
        
        Args:
            claude_response: Response from Claude API
            
        Returns:
            Response in OpenAI-compatible format
        """
        # Extract content from Claude response
        try:
            # Try to access as an object (dot notation)
            content = claude_response.content[0].text
        except (AttributeError, TypeError):
            # Try to access as a dictionary
            try:
                if isinstance(claude_response, dict) and 'content' in claude_response:
                    if isinstance(claude_response['content'], list) and len(claude_response['content']) > 0:
                        content_item = claude_response['content'][0]
                        if isinstance(content_item, dict) and 'text' in content_item:
                            content = content_item['text']
                        else:
                            content = str(content_item)
                    else:
                        content = str(claude_response['content'])
                else:
                    # Fallback for unexpected response format
                    content = "Error: Unable to extract content from Claude response"
                    log_to_file(f"Unexpected Claude response format: {claude_response}")
            except Exception as e:
                content = f"Error: {str(e)}"
                log_to_file(f"Error extracting content from Claude response: {str(e)}")
        
        # Format in OpenAI-compatible way
        return {
            "id": getattr(claude_response, "id", "claude_response"),
            "object": "chat.completion",
            "created": getattr(claude_response, "created_at", 0),
            "model": self.model,
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
                "prompt_tokens": getattr(claude_response, "usage", {}).get("input_tokens", 0),
                "completion_tokens": getattr(claude_response, "usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    getattr(claude_response, "usage", {}).get("input_tokens", 0) +
                    getattr(claude_response, "usage", {}).get("output_tokens", 0)
                )
            }
        }


"""
Tagger module for the AI Tagger add-on.
This module contains the Tagger class that handles the tag generation process.
Simplified version specifically optimized for Gemini API.
"""
import json
import time
import traceback
from typing import Dict, List, Any, Tuple, Optional, Union
import pprint

from ..service.service_factory import ServiceFactory
from ..core.config import ConfigManager
from ..utils.constants import LOG_PREFIX, ServiceType
from ..utils.helpers import log_to_file


class Tagger:
    """
    Simplified Tagger class for generating and applying tags to Anki notes.
    
    This specialized version is optimized to work exclusively with Gemini 2.0 Flash API.
    It has enhanced error handling and debugging to troubleshoot response format issues.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the tagger with configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Load config if not provided
        self.config_manager = ConfigManager()
        self.config = config or self.config_manager.get_config()
        
        # Always use Gemini, hardcoded
        self.service_type = ServiceType.GEMINI
        log_to_file(f"{LOG_PREFIX} Using Gemini 2.0 Flash service exclusively")

        # Create Gemini service with focused config
        gemini_config = self.config.get('gemini', {})
        
        # Add debug logging for the Gemini configuration
        log_to_file(f"{LOG_PREFIX} Gemini API Config: {json.dumps(gemini_config, indent=2)}")
        
        self.service = ServiceFactory.create_service(self.service_type.value, gemini_config)
        
        if not self.service:
            error_msg = f"Failed to create Gemini service. Check your API key and settings."
            log_to_file(f"{LOG_PREFIX} {error_msg}")
            raise ValueError(error_msg)
        
        # Load tag generation settings
        self.tag_generation_config = self.config.get("tag_generation", {})
        
        # Specialized prompt template for document tagging
        self.prompt_template = (
            "You are analyzing a document and finding relevant Anki flashcards. "
            "Based on the document content below, generate {max_tags_per_card} concise tags that "
            "would help categorize and find cards related to this content. "
            "Return only the tags in a comma-separated list.\n\n"
            "Document Content:\n{content}"
        )
        
        self.max_tags_per_card = self.config.get("max_tags_per_card", 5)
        
    def generate_tags(self, content: str, existing_tags: List[str] = None, fields: Dict[str, str] = None) -> Tuple[List[str], Optional[str]]:
        """
        Generate tags for the given content using Gemini API.
        
        Args:
            content: Document content to generate tags for
            existing_tags: Optional list of existing tags to avoid duplicates
            fields: Optional dictionary of fields for the note
            
        Returns:
            Tuple of (list of tags, error message if any)
        """
        existing_tags = existing_tags or []
        fields = fields or {}
        
        # Format the prompt with content and other parameters
        prompt = self._format_prompt(content, fields)
        
        # Log the beginning of the API call
        log_to_file(f"{LOG_PREFIX} Sending request to Gemini API for tag generation")
        
        # Get response from Gemini API
        start_time = time.time()
        try:
            response = self._get_ai_response(prompt)
            elapsed = time.time() - start_time
            log_to_file(f"{LOG_PREFIX} Received response from Gemini API in {elapsed:.2f}s")
        except Exception as e:
            error_message = f"Error communicating with Gemini API: {str(e)}"
            log_to_file(f"{LOG_PREFIX} {error_message}\n{traceback.format_exc()}")
            return [], error_message
        
        # Debug the raw response
        self._debug_response(response)
        
        # Check for errors in the response
        if isinstance(response, dict) and response.get("error", False):
            error_message = response.get("content", "Unknown error")
            log_to_file(f"{LOG_PREFIX} Error from Gemini API: {error_message}")
            return [], error_message
        
        # Extract tags from response
        try:
            tags = self._extract_tags_from_response(response)
            log_to_file(f"{LOG_PREFIX} Extracted {len(tags)} tags: {', '.join(tags)}")
        except Exception as e:
            error_message = f"Error extracting tags from response: {str(e)}"
            log_to_file(f"{LOG_PREFIX} {error_message}\n{traceback.format_exc()}")
            return [], error_message
        
        # Limit to max tags and remove duplicates with existing tags
        tags = [tag for tag in tags if tag.lower() not in [et.lower() for et in existing_tags]]
        tags = tags[:self.max_tags_per_card]
        
        return tags, None
    
    def apply_tags(self, note: Any, tags: List[str]) -> bool:
        """
        Apply tags to a note.
        
        Args:
            note: Note to apply tags to
            tags: List of tags to apply
            
        Returns:
            True if tags were applied, False otherwise
        """
        try:
            if not tags:
                log_to_file(f"{LOG_PREFIX} No tags to apply")
                return False
            
            # Get the prefix to use for tags
            prefix = self.config.get("tag_prefix", "ai:")
            
            # Add the prefix to each tag
            prefixed_tags = [f"{prefix}{tag}" for tag in tags]
            
            # Get existing tags
            existing_tags = note.tags.strip().split() if hasattr(note, 'tags') else []
            
            # Add new tags
            for tag in prefixed_tags:
                if tag not in existing_tags:
                    existing_tags.append(tag)
            
            # Update the note's tags
            note.tags = " ".join(existing_tags)
            note.flush()
            
            log_to_file(f"{LOG_PREFIX} Applied {len(prefixed_tags)} tags to note")
            return True
        except Exception as e:
            error_message = f"Error applying tags to note: {str(e)}"
            log_to_file(f"{LOG_PREFIX} {error_message}\n{traceback.format_exc()}")
            return False
    
    def _format_prompt(self, content: str, fields: Dict[str, str] = None) -> str:
        """
        Format the prompt for tag generation.
        
        Args:
            content: Content to generate tags for
            fields: Optional dictionary of fields for the note
            
        Returns:
            Formatted prompt
        """
        try:
            # Format prompt with parameters
            format_params = {
                "content": content,
                "max_tags_per_card": self.max_tags_per_card
            }
            
            # Include fields if provided and enabled
            if fields and self.tag_generation_config.get("include_field_names", True):
                # Exclude specified fields
                exclude_fields = self.tag_generation_config.get("exclude_fields", ["Extra", "Note ID", "Tags"])
                format_params.update({
                    key: value for key, value in fields.items() 
                    if key not in exclude_fields
                })
            
            # Format the prompt
            return self.prompt_template.format(**format_params)
            
        except Exception as e:
            error_message = f"Error formatting prompt: {str(e)}"
            log_to_file(f"{LOG_PREFIX} {error_message}\n{traceback.format_exc()}")
            # Return a simplified fallback prompt
            return f"Generate {self.max_tags_per_card} tags for this content:\n\n{content}"
            
    def _get_ai_response(self, prompt: str) -> Dict[str, Any]:
        """
        Get AI response from Gemini API for the given prompt.
        
        Args:
            prompt: Prompt to send to the AI
            
        Returns:
            Dictionary containing the AI response
        """
        try:
            # Create messages in the format expected by the service
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that generates concise, relevant tags for content."},
                {"role": "user", "content": prompt}
            ]
            
            # Log message details (without full content for brevity)
            log_to_file(f"{LOG_PREFIX} Sending message to Gemini API: [System prompt + User content of length {len(prompt)}]")
            
            # Get response from service
            response = self.service.chat_completion(messages)
            
            # Verify response is a dictionary
            if not isinstance(response, dict):
                log_to_file(f"{LOG_PREFIX} Warning: Unexpected response type: {type(response)}")
                response = {"content": str(response), "error": False}
                
            return response
            
        except Exception as e:
            error_msg = f"Error getting AI response: {str(e)}"
            log_to_file(f"{LOG_PREFIX} {error_msg}\n{traceback.format_exc()}")
            return {"content": f"Error: {str(e)}", "error": True}
    
    def _debug_response(self, response: Any) -> None:
        """
        Debug the response from the AI service.
        
        Args:
            response: Response from AI service
        """
        try:
            # Get a formatted representation of the response for debugging
            if isinstance(response, dict):
                formatted_response = json.dumps(response, indent=2)
            else:
                formatted_response = pprint.pformat(response)
                
            # Log truncated version to avoid excessive logging
            max_length = 1000
            if len(formatted_response) > max_length:
                log_to_file(f"{LOG_PREFIX} Response structure (truncated):\n{formatted_response[:max_length]}...(truncated)")
            else:
                log_to_file(f"{LOG_PREFIX} Response structure:\n{formatted_response}")
                
            # Debug potentially problematic fields
            if isinstance(response, dict):
                if 'choices' in response:
                    log_to_file(f"{LOG_PREFIX} Response has 'choices' field: {type(response['choices'])}")
                    if isinstance(response['choices'], list) and len(response['choices']) > 0:
                        log_to_file(f"{LOG_PREFIX} First choice type: {type(response['choices'][0])}")
                elif 'content' in response:
                    log_to_file(f"{LOG_PREFIX} Response has 'content' field: {type(response['content'])}")
            
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} Error debugging response: {str(e)}")
    
    def _extract_tags_from_response(self, response: Dict[str, Any]) -> List[str]:
        """
        Extract tags from the AI response - specialized for Gemini API.
        
        Args:
            response: Response from the AI service
            
        Returns:
            List of extracted tags
        """
        try:
            # Extract content from response - specialized for Gemini responses via OpenAICompatible wrapper
            content = ""
            
            # Handle response with careful error checking at each step 
            if isinstance(response, dict):
                # Most common case: our wrapper puts content in the 'content' field
                if 'content' in response:
                    content = response['content']
                    if not isinstance(content, str):
                        log_to_file(f"{LOG_PREFIX} Warning: content field is not a string: {type(content)}")
                        content = str(content)
                
                # OpenAI-compatible format (response.choices[0].message.content)
                elif 'choices' in response:
                    if isinstance(response['choices'], list) and len(response['choices']) > 0:
                        choice = response['choices'][0]
                        
                        if isinstance(choice, dict):
                            if 'message' in choice and isinstance(choice['message'], dict):
                                message_content = choice['message'].get('content')
                                if message_content is not None:
                                    content = message_content
                            elif 'text' in choice:
                                content = choice.get('text', '')
                    else:
                        log_to_file(f"{LOG_PREFIX} Warning: 'choices' is not a list or is empty: {response['choices']}")
            else:
                # Last resort: convert to string
                content = str(response)
            
            # Trim whitespace
            content = content.strip()
            
            if not content:
                log_to_file(f"{LOG_PREFIX} Warning: Empty content extracted from response")
                return []
                
            # Try to parse as JSON first (ideal case)
            try:
                # If the content appears to be a JSON array
                if content.startswith('[') and content.endswith(']'):
                    tags = json.loads(content)
                    if isinstance(tags, list):
                        return self._clean_tags(tags)
                
                # Try to find a JSON array in the response
                start_idx = content.find('[')
                end_idx = content.rfind(']')
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx+1]
                    tags = json.loads(json_str)
                    if isinstance(tags, list):
                        return self._clean_tags(tags)
            except json.JSONDecodeError:
                # Not JSON, continue to text parsing
                pass
                
            # Fallback: Split by commas or new lines
            if ',' in content:
                tags = [tag.strip() for tag in content.split(',')]
            else:
                tags = [tag.strip() for tag in content.split('\n') if tag.strip()]
                
            return self._clean_tags(tags)
            
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} Error extracting tags: {str(e)}\n{traceback.format_exc()}")
            # Return empty list as ultimate fallback
            return []
    
    def _clean_tags(self, tags: List[str]) -> List[str]:
        """
        Clean and filter the list of tags.
        
        Args:
            tags: Raw list of tags
            
        Returns:
            A cleaned and filtered list of tags
        """
        # Clean up tags
        clean_tags = []
        for tag in tags:
            # Skip if not a string
            if not isinstance(tag, str):
                continue
                
            # Clean the tag
            tag = tag.strip('"\'[]').strip()
            
            # Skip empty tags or code block markers
            if not tag or tag.startswith(('```', '---', '==', '#')):
                continue
                
            # Remove any remaining special characters and normalize whitespace
            tag = tag.replace('\n', ' ').replace('\t', ' ')
            while '  ' in tag:  # Replace double spaces with single spaces
                tag = tag.replace('  ', ' ')
                
            clean_tags.append(tag)
        
        return clean_tags

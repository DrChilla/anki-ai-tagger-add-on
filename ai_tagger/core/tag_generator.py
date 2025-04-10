from typing import List, Dict, Any, Optional
import json
import re

from ..utils.helpers import log_to_file
from ..utils.constants import LOG_PREFIX
from ..service.openAICompatible import OpenAICompatibleService

class TagGenerator:
    """
    Core class for generating tags for Anki notes using AI services.
    """
    
    def __init__(self, service: OpenAICompatibleService):
        """
        Initialize the TagGenerator with an AI service.
        
        Args:
            service: An instance of OpenAICompatibleService or a derived class
        """
        self.service = service
    
    def generate_tags(self, note_content: str, existing_tags: List[str] = None, 
                     field_names: List[str] = None, max_tags: int = 10,
                     min_tag_length: int = 2) -> List[str]:
        """
        Generate tags for a note based on its content using the AI service.
        
        Args:
            note_content: The content of the note
            existing_tags: Any existing tags the note already has
            field_names: Names of the fields in the note (for context)
            max_tags: Maximum number of tags to generate
            min_tag_length: Minimum length for a valid tag
            
        Returns:
            A list of generated tags
        """
        try:
            # Prepare the context for the AI service
            prompt = self._build_prompt(note_content, existing_tags, field_names, max_tags)
            
            # Call the AI service
            response = self.service.chat_completion(prompt)
            
            # Process the response
            tags = self._extract_tags_from_response(response, min_tag_length, max_tags)
            
            # Log the result
            log_to_file(f"{LOG_PREFIX} Generated {len(tags)} tags for note", "info")
            
            return tags
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} Error generating tags: {str(e)}", "error")
            return []
    
    def _build_prompt(self, note_content: str, existing_tags: List[str] = None, 
                     field_names: List[str] = None, max_tags: int = 10) -> List[Dict[str, str]]:
        """
        Build a prompt for the AI service based on the note content and context.
        
        Args:
            note_content: The content of the note
            existing_tags: Any existing tags the note already has
            field_names: Names of the fields in the note
            max_tags: Maximum number of tags to generate
            
        Returns:
            A list of message dictionaries with 'role' and 'content' keys
        """
        # Create the system message
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant specialized in generating relevant, concise tags for Anki notes. Generate tags that categorize and describe the content effectively."
        }
        
        # Build the user message content
        user_content = f"Note content:\n{note_content}\n\n"
        
        # Include field names if available
        if field_names and len(field_names) > 0:
            user_content += f"Field names: {', '.join(field_names)}\n\n"
        
        # Include existing tags if available
        if existing_tags and len(existing_tags) > 0:
            user_content += f"Existing tags: {', '.join(existing_tags)}\n\n"
        
        # Add specific instructions
        user_content += (
            f"Please generate up to {max_tags} tags for this note. "
            "Tags should be single words or short phrases using underscores instead of spaces. "
            "Tags should be relevant to the content and help categorize or find the note later. "
            f"Format your response as a JSON array of strings containing only the tags.\n\n"
            "Example response format: [\"tag1\", \"important_concept\", \"chapter_3\"]"
        )
        
        # Create the user message
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        # Return the formatted messages list
        return [system_message, user_message]
    
    def _extract_tags_from_response(self, response: Dict[str, Any], min_tag_length: int = 2, 
                                   max_tags: int = 10) -> List[str]:
        """
        Extract tags from the AI service response.
        
        Args:
            response: The raw response dictionary from the AI service
            min_tag_length: Minimum length for a valid tag
            max_tags: Maximum number of tags to return
            
        Returns:
            A list of extracted tags
        """
        try:
            # Extract the content from the response dictionary
            content = ""
            if isinstance(response, dict):
                # Handle different response structures
                if "choices" in response and len(response["choices"]) > 0:
                    # OpenAI-style response
                    if "message" in response["choices"][0]:
                        content = response["choices"][0]["message"].get("content", "")
                    elif "text" in response["choices"][0]:
                        content = response["choices"][0].get("text", "")
                elif "content" in response:
                    # Simple content field
                    content = response.get("content", "")
                elif "output" in response:
                    # Some APIs use 'output'
                    content = response.get("output", "")
            else:
                # Fallback if response is already a string
                content = str(response)
            
            # Try to parse as JSON first (ideal case)
            try:
                tags_list = json.loads(content.strip())
                if isinstance(tags_list, list):
                    # Filter and clean the tags
                    return self._clean_tags(tags_list, min_tag_length, max_tags)
            except json.JSONDecodeError:
                # Not valid JSON, try regex extraction
                pass
                
            # If JSON parsing failed, try to extract using regex
            tags = []
            # Look for words that look like tags (alphanumeric with possible underscores)
            tag_pattern = r'\b[a-zA-Z0-9_]+\b'
            potential_tags = re.findall(tag_pattern, content)
            tags = [tag for tag in potential_tags if len(tag) >= min_tag_length]
            
            return self._clean_tags(tags, min_tag_length, max_tags)
        except Exception as e:
            log_to_file(f"{LOG_PREFIX} Error extracting tags from response: {str(e)}", "error")
            return []
    
    def _clean_tags(self, tags: List[str], min_tag_length: int = 2, max_tags: int = 10) -> List[str]:
        """
        Clean and filter the list of tags.
        
        Args:
            tags: Raw list of tags
            min_tag_length: Minimum length for a valid tag
            max_tags: Maximum number of tags to return
            
        Returns:
            A cleaned and filtered list of tags
        """
        # Convert to lowercase
        tags = [tag.lower() for tag in tags]
        
        # Filter out tags that are too short
        tags = [tag for tag in tags if len(tag) >= min_tag_length]
        
        # Remove duplicates while preserving order
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        
        # Limit to max_tags
        return unique_tags[:max_tags]
    
    def batch_generate_tags(self, notes: List[Dict[str, Any]], max_tags_per_note: int = 10) -> Dict[str, List[str]]:
        """
        Generate tags for multiple notes in batch.
        
        Args:
            notes: List of note dictionaries, each with 'id' and 'content' keys
            max_tags_per_note: Maximum number of tags to generate per note
            
        Returns:
            A dictionary mapping note IDs to lists of generated tags
        """
        results = {}
        
        for note in notes:
            note_id = note.get('id')
            content = note.get('content', '')
            existing_tags = note.get('tags', [])
            field_names = note.get('field_names', [])
            
            tags = self.generate_tags(
                content, 
                existing_tags=existing_tags,
                field_names=field_names,
                max_tags=max_tags_per_note
            )
            
            results[note_id] = tags
            
        return results


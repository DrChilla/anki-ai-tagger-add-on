"""
Constants for the AI Tagger add-on.
This file contains various constants used throughout the add-on.
"""

from enum import Enum, auto

# Add-on information
ADDON_NAME = "AI Tagger"
ADDON_VERSION = "1.0.0"
AUTHOR = "AI Tagger Development Team"
GITHUB_REPO = "https://github.com/yourrepository/AI_Tagger"

# Configuration defaults
DEFAULT_CONFIG = {
    # General settings
    "service_type": "OPENAI",  # Default service type
    "auto_tag_on_add": False,   # Automatically tag notes when added
    "auto_tag_on_edit": False,  # Automatically tag notes when edited
    "tag_prefix": "ai:",        # Prefix for AI-generated tags
    "max_tags_per_card": 5,     # Maximum number of tags to add per card
    "minimum_confidence": 0.7,  # Minimum confidence score for a tag to be applied
    "use_field_hints": True,    # Use field names as hints for tag generation
    "log_enabled": True,        # Enable logging
    "log_level": "INFO",        # Default log level
    
    # OpenAI API settings
    "openai": {
        "api_key": "",
        "language_model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "api_base": "https://api.openai.com/v1",
        "timeout": 30,  # seconds
        "organization_id": "",  # Optional for OpenAI organization accounts
    },
    
    # Azure OpenAI API settings
    "azure_openai": {
        "api_key": "",
        "api_base": "",  # Azure endpoint URL
        "api_version": "2023-05-15",
        "deployment_name": "",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "timeout": 30,  # seconds
    },
    
    # Anthropic (Claude) API settings
    "anthropic": {
        "api_key": "",
        "model": "claude-2",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "api_base": "https://api.anthropic.com/v1",
        "timeout": 30,  # seconds
    },
    
    # Ollama API settings
    "ollama": {
        "api_base": "http://localhost:11434",
        "model": "llama2",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "timeout": 30,  # seconds
    },
    
    # OpenRouter API settings
    "openrouter": {
        "api_key": "",
        "model": "openai/gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "api_base": "https://openrouter.ai/api/v1",
        "timeout": 30,  # seconds
        "route_prefix": "",  # Optional routing prefix
    },
    
    # Local LLM settings
    "local": {
        "api_base": "http://localhost:8000",
        "model": "ggml-model",
        "max_tokens": 100,
        "temperature": 0.7,
        "timeout": 60,  # seconds
    },
    
    # Custom API settings
    "custom": {
        "api_key": "",
        "api_base": "",
        "model": "",
        "max_tokens": 100,
        "temperature": 0.7,
        "headers": {},
        "payload_template": {
            "model": "{model}",
            "messages": "{messages}",
            "max_tokens": "{max_tokens}",
            "temperature": "{temperature}"
        },
        "response_path": "choices[0].message.content",  # JSONPath to extract response
        "timeout": 30,  # seconds
    },

    # Gemini API settings
    "gemini": {
        "api_key": "",
        "model": "gemini-pro",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "api_base": "https://generativelanguage.googleapis.com/v1",
        "timeout": 30,  # seconds
    },

    # Perplexity API settings
    "perplexity": {
        "api_key": "",
        "model": "pplx-70b-online",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.0,
        "api_base": "https://api.perplexity.ai",
        "timeout": 30,  # seconds
    },

    # Tag generation settings
    "tag_generation": {
        "prompt_template": "Generate {max_tags_per_card} concise tags for the following flashcard content, separating each tag with a comma:\n\nFront: {front}\nBack: {back}",
        "include_field_names": True,
        "exclude_fields": ["Extra", "Note ID", "Tags"],
        "use_cached_results": True,
        "cache_duration_days": 7,
    }
}

# API endpoints and settings
API_ENDPOINTS = {
    "OPENAI": "https://api.openai.com/v1/chat/completions",
    "AZURE_OPENAI": "{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}",
    "ANTHROPIC": "https://api.anthropic.com/v1/messages",
    "OLLAMA": "http://localhost:11434/api/chat",
    "OPENROUTER": "https://openrouter.ai/api/v1/chat/completions",
    "LOCAL": "{api_base}/v1/chat/completions",
    "CUSTOM": "{api_base}",
    "GEMINI": "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
    "PERPLEXITY": "https://api.perplexity.ai/chat/completions"
}
REQUEST_TIMEOUT = 30  # seconds

# GUI Constants
WINDOW_TITLE = "AI Tagger Settings"
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
ICON_PATH = "resources/icons/ai_tagger_icon.png"

# Tag categories
TAG_CATEGORIES = [
    "topic",
    "difficulty",
    "subject",
    "concept",
    "chapter"
]

# File paths
LOG_FILE_PATH = "AI_Tagger.log"
LOG_PREFIX = "[AI_Tagger]"
CONFIG_FILE_NAME = "config.json"

# Messages
SUCCESS_MESSAGE = "Tags successfully generated and applied!"
ERROR_MESSAGE = "Error generating tags. Please check your settings and try again."
API_KEY_MISSING_MESSAGE = "API key is missing. Please add your API key in the settings."

# Anki specific constants
REQUIRED_ANKI_VERSION = "2.1.45"


# Model definitions
ANTHROPIC_MODELS = {
    "claude-3-opus-20240229": "Most powerful model for highly complex tasks",
    "claude-3-sonnet-20240229": "Balance of intelligence and speed",
    "claude-3-haiku-20240307": "Fastest and most compact model",
    "claude-2.1": "Legacy Claude 2.1 model",
    "claude-2.0": "Legacy Claude 2.0 model",
    "claude-instant-1.2": "Fast, cost-effective legacy model",
}

# OpenAI models
OPENAI_MODELS = {
    "gpt-4-turbo": "Most capable GPT-4 model optimized for speed and cost",
    "gpt-4o": "Latest GPT-4 model with best price-performance balance",
    "gpt-4": "Original GPT-4 model with strong reasoning",
    "gpt-3.5-turbo": "Fast and cost-effective for most everyday tasks",
    "gpt-3.5-turbo-16k": "GPT-3.5 with larger context window (16k tokens)",
}

# Google models
GEMINI_MODELS = {
    "gemini-pro": "Standard Gemini model for general use",
    "gemini-2.0-flash": "Fast and optimized newer Gemini model (recommended)",
    "gemini-2.0-flash-lite": "Lightweight version of Gemini 2.0 Flash"
}

# Perplexity models
PERPLEXITY_MODELS = {
    "pplx-7b-online": "Fast and efficient online-connected model",
    "pplx-70b-online": "High-capability online-connected model",
    "pplx-7b-chat": "Fast chat model without web connection",
    "pplx-70b-chat": "High-capability chat model without web connection",
}

# AI Service Types
class ServiceType(Enum):
    """Enum of supported AI service types."""
    OPENAI = 'openai'
    AZURE_OPENAI = 'azure_openai'
    ANTHROPIC_CLAUDE = 'anthropic'
    OLLAMA = 'ollama'
    OPENROUTER = 'openrouter'
    LOCAL = 'local'
    CUSTOM = 'custom'
    GEMINI = 'gemini'
    PERPLEXITY = 'perplexity'

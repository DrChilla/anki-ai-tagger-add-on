import os
import sys
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from aqt import mw
from aqt.utils import showInfo, showWarning, showCritical
from aqt.qt import QMessageBox

from .constants import ADDON_NAME, LOG_PREFIX


def get_addon_dir() -> str:
    """
    Return the full path to the add-on directory.
    
    Returns:
        The absolute path to the addon directory
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_user_files_dir() -> str:
    """
    Return the full path to the user_files directory.
    
    Returns:
        The absolute path to the user_files directory
    """
    return os.path.join(get_addon_dir(), "user_files")


def ensure_dir_exists(directory_path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        directory_path: The path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def log_to_file(message: str, level: str = "INFO", filepath: Optional[str] = None) -> None:
    """
    Log a message to a log file in the add-on's directory.
    
    Args:
        message: The message to log
        level: The log level (INFO, WARNING, ERROR, DEBUG)
        filepath: Optional filepath. If not provided, uses default log file
    """
    if filepath is None:
        log_dir = os.path.join(get_addon_dir(), "logs")
        ensure_dir_exists(log_dir)
        filepath = os.path.join(log_dir, f"{ADDON_NAME.lower().replace(' ', '_')}.log")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{level}] {LOG_PREFIX}: {message}\n"
    
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:
        print(f"Error writing to log file: {e}")


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely loads a JSON string, returning a default value on error.
    
    Args:
        json_str: The JSON string to parse
        default: The default value to return if parsing fails
    
    Returns:
        Parsed JSON data or default value if parsing fails
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        log_to_file(f"Error parsing JSON: {e}\nJSON string: {json_str[:100]}...", "ERROR")
        return default


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        The parsed JSON data as a dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_exception(e)
        return {}


def write_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Dictionary data to write
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        log_exception(e)
        return False


def format_exception(e: Exception) -> str:
    """
    Formats an exception into a readable string with traceback.
    
    Args:
        e: The exception to format
    
    Returns:
        Formatted exception string
    """
    return f"{type(e).__name__}: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"


def log_exception(e: Exception) -> None:
    """
    Log an exception with traceback to the log file.
    
    Args:
        e: The exception to log
    """
    error_trace = traceback.format_exc()
    log_to_file(f"Exception: {str(e)}\n{error_trace}", "ERROR")


def show_info(message: str, title: str = ADDON_NAME) -> None:
    """
    Show an information message box.
    
    Args:
        message: The message to display
        title: The title of the message box
    """
    showInfo(message, title=title)


def show_warning(message: str, title: str = ADDON_NAME) -> None:
    """
    Show a warning message box.
    
    Args:
        message: The warning message to display
        title: The title of the message box
    """
    showWarning(message, title=title)


def show_critical(message: str, title: str = ADDON_NAME) -> None:
    """
    Show a critical error message box.
    
    Args:
        message: The critical error message to display
        title: The title of the message box
    """
    showCritical(message, title=title)


def show_yes_no_dialog(message: str, title: str = ADDON_NAME) -> bool:
    """
    Show a Yes/No dialog box and return True if Yes was clicked.
    
    Args:
        message: The question to display
        title: The title of the dialog box
        
    Returns:
        True if the user clicked Yes, False otherwise
    """
    reply = QMessageBox.question(
        mw, title, message, 
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
    )
    return reply == QMessageBox.StandardButton.Yes


def get_config(key: str = None, default: Any = None) -> Any:
    """
    Get a configuration value from the addon config.
    
    Args:
        key: The configuration key to retrieve. If None, returns the entire config
        default: The default value to return if the key is not found
        
    Returns:
        The configuration value or the default
    """
    config = mw.addonManager.getConfig(mw.pm.nameToID(ADDON_NAME) or "")
    
    if config is None:
        return default
        
    if key is None:
        return config
        
    return config.get(key, default)


def update_config(key: str, value: Any) -> bool:
    """
    Update a configuration value in the addon config.
    
    Args:
        key: The configuration key to update
        value: The new value
        
    Returns:
        True if successful, False otherwise
    """
    try:
        addon_id = mw.pm.nameToID(ADDON_NAME) or ""
        if not addon_id:
            log_to_file(f"Couldn't find addon ID for {ADDON_NAME}", "ERROR")
            return False
            
        config = mw.addonManager.getConfig(addon_id) or {}
        config[key] = value
        mw.addonManager.writeConfig(addon_id, config)
        return True
    except Exception as e:
        log_exception(e)
        return False


def get_note_content(note: Any) -> str:
    """
    Extract the content from a note for processing.
    
    Args:
        note: The Anki note object
        
    Returns:
        A string containing the concatenated content of the note
    """
    content = []
    for field_name, field_value in note.items():
        content.append(f"{field_name}: {field_value}")
    return "\n".join(content)


def sanitize_tags(tags: List[str]) -> List[str]:
    """
    Sanitize a list of tags to ensure they are valid Anki tags.
    
    Args:
        tags: List of tag strings
        
    Returns:
        List of sanitized tag strings
    """
    sanitized = []
    for tag in tags:
        # Remove spaces and replace with underscores
        tag = tag.strip().replace(" ", "_")
        # Remove any characters that are problematic in Anki tags
        tag = ''.join(c for c in tag if c.isalnum() or c in '-_')
        if tag:
            sanitized.append(tag.lower())
    return sanitized

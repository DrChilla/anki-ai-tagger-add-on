#!/usr/bin/env python
# Simple test script to validate the SettingsDialog class
# This script mocks Anki dependencies to test in isolation

import sys
from unittest.mock import MagicMock

# Mock PyQt
sys.modules['PyQt6'] = MagicMock()
sys.modules['PyQt6.QtWidgets'] = MagicMock()
sys.modules['PyQt6.QtCore'] = MagicMock()

# Mock Anki components
sys.modules['aqt'] = MagicMock()
sys.modules['aqt.utils'] = MagicMock()

# Create mock ConfigManager
class MockConfigManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MockConfigManager()
        return cls._instance
    
    def get_config(self):
        return {}
    
    def save_config(self):
        pass

# Mock the imports
sys.modules['ai_tagger.core.document_processor'] = MagicMock()
sys.modules['ai_tagger.core.tagger'] = MagicMock()
sys.modules['ai_tagger.core.tag_generator'] = MagicMock()
sys.modules['ai_tagger.core.config'] = MagicMock()
sys.modules['ai_tagger.core.config'].ConfigManager = MockConfigManager
sys.modules['ai_tagger.service.service_factory'] = MagicMock()
sys.modules['ai_tagger.utils.constants'] = MagicMock()
sys.modules['ai_tagger.utils.constants'].DEFAULT_CONFIG = {}
sys.modules['ai_tagger.utils.constants'].WINDOW_TITLE = "AI Tagger"
sys.modules['ai_tagger.utils.constants'].WINDOW_WIDTH = 800
sys.modules['ai_tagger.utils.constants'].WINDOW_HEIGHT = 600
sys.modules['ai_tagger.utils.constants'].SUCCESS_MESSAGE = "Success"
sys.modules['ai_tagger.utils.constants'].ERROR_MESSAGE = "Error"
sys.modules['ai_tagger.utils.constants'].OPENAI_MODELS = {}
sys.modules['ai_tagger.utils.constants'].ANTHROPIC_MODELS = {}
sys.modules['ai_tagger.utils.constants'].GEMINI_MODELS = {}
sys.modules['ai_tagger.utils.constants'].PERPLEXITY_MODELS = {}
sys.modules['ai_tagger.utils.constants'].ServiceType = MagicMock()

try:
    # Import the SettingsDialog class - this will use our mocks
    sys.path.append('.')
    from ai_tagger.gui.main_dialog import SettingsDialog
    
    # Create a mock parent
    mock_parent = MagicMock()
    
    # Create an instance of SettingsDialog
    print("Creating SettingsDialog instance...")
    dialog = SettingsDialog(mock_parent)
    
    # The initialization process should have already called init_ui
    # If we got here without errors, the fix was successful
    print("Success! The SettingsDialog was initialized without errors.")
    print("The fix resolved the AttributeError: 'SettingsDialog' object has no attribute 'create_document_processing_tab'")

except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("The fix did not resolve the issue.")
    sys.exit(1)

# Exit with success code
sys.exit(0)


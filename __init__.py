"""
AI Tagger Enhanced Addon for Anki
"""
import os
import sys

# Add our libs directory to the beginning of sys.path before any other imports
addon_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(addon_dir, "libs")

# 1. Ensure our libs directory is the first entry in sys.path
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

# 2. Remove typing_extensions from sys.modules if it's already loaded
if 'typing_extensions' in sys.modules:
    del sys.modules['typing_extensions']

# 3. Block access to the conflicting addon's module
conflicting_addon_id = '1322529746'
conflicting_addon_path = os.path.join(os.path.dirname(addon_dir), conflicting_addon_id)
conflicting_typing_ext_path = os.path.join(conflicting_addon_path, 'lib', 'other', 'typing_extensions')

# Define a hook to modify sys.path during import
original_path_hooks = sys.path_hooks.copy()

def path_hook_wrapper(original_hook):
    def wrapper(path):
        # Skip the conflicting typing_extensions path
        if path == conflicting_typing_ext_path or (isinstance(path, str) and path.startswith(conflicting_typing_ext_path)):
            raise ImportError(f"Blocked import from conflicting addon: {path}")
        return original_hook(path)
    return wrapper

# Apply the wrapper to all path hooks
sys.path_hooks = [path_hook_wrapper(hook) for hook in original_path_hooks]

# Clear the path importer cache to apply the new hooks
sys.path_importer_cache.clear()

# Import and verify our version of typing_extensions works
try:
    from typing_extensions import override
    print(f"Successfully imported 'override' from typing_extensions")
except ImportError as e:
    print(f"Error importing typing_extensions: {e}")
    raise

# Now continue with the rest of the imports
import logging
import json
from pathlib import Path
import importlib.util
from datetime import datetime
def setup_logging():
    """Setup logging for the addon."""
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(addon_dir, "logs")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"ai_tagger_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger("ai_tagger")
setup_logging()
logging.info("AI Tagger addon initialized")

# Import Anki modules first, so we can show error messages if needed
from aqt import mw, gui_hooks
from aqt.qt import QAction, QMessageBox, Qt

# Set up logging
logger = setup_logging()
logger.info("AI Tagger Enhanced addon initializing")
logger.info(f"Python path: {sys.path}")

# Try to import addon-specific modules
try:
    # Try importing the required modules
    # Add current directory to path if needed
    if addon_dir not in sys.path:
        sys.path.append(addon_dir)
    from ai_tagger.gui.main_dialog import show_main_wizard
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Import error: {e}", exc_info=True)
    
    # Show a message to the user if imports fail
    def show_missing_dependencies_error():
        msg_box = QMessageBox(mw)
        msg_box.setWindowTitle("AI Tagger Enhanced - Missing Dependencies")
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText("Could not load AI Tagger Enhanced because some required Python modules were not properly loaded.\n\n"
                      "Please restart Anki after any updates and make sure all required libraries are in the 'libs' directory.")
        
        # Add details about logs
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(addon_dir, "logs")
        msg_box.setDetailedText(f"Check the log files in:\n{log_dir}\n\n"
                              "This addon requires certain libraries to be available in the 'libs' directory.")
        
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    # Schedule the error message to be shown when Anki's main window is ready
    gui_hooks.main_window_did_init.append(show_missing_dependencies_error)
    
    # Reraise exception to prevent partial addon loading
    raise

def init():
    """
    Initialize the addon by adding menu item
    """
    action = QAction("AI Tagger", mw)
    action.triggered.connect(show_main_wizard)
    mw.form.menuTools.addAction(action)

# Register the init function
gui_hooks.main_window_did_init.append(init)

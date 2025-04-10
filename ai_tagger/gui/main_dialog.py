import logging
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QTabWidget,
    QPlainTextEdit,
    QScrollArea,
    QWidget,
    QFileDialog,
    QApplication
)
from PyQt6.QtCore import Qt
import sys
import os

# Check for document processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not installed. Cannot process PDF documents.")

# Text files are always supported
TEXT_SUPPORT = True

# Import core components
from ..core.document_processor import DocumentProcessor
from ..core.tagger import Tagger
from ..core.tag_generator import TagGenerator
from ..core.config import ConfigManager
from ..service.service_factory import ServiceFactory

# Import Anki components
from aqt import mw
from aqt.utils import showInfo, showWarning

# Import constants
from ..utils.constants import (
    DEFAULT_CONFIG,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    SUCCESS_MESSAGE,
    ERROR_MESSAGE,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    PERPLEXITY_MODELS,
    ServiceType
)

class SettingsDialog(QDialog):
    """Main settings dialog for AI Tagger."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_manager = ConfigManager.get_instance()
        self.config = self.config_manager.get_config()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Add tabs
        general_tab = self.create_general_tab()
        service_tab = self.create_service_tab()
        tag_generation_tab = self.create_tag_generation_tab()
        formats_tab = self.create_formats_tab()
        document_tab = self.create_document_processing_tab()
        
        tab_widget.addTab(general_tab, "General")
        tab_widget.addTab(service_tab, "AI Service")
        tab_widget.addTab(document_tab, "Document Processing")
        tab_widget.addTab(tag_generation_tab, "Tag Generation")
        tab_widget.addTab(formats_tab, "Supported Formats")
        
        main_layout.addWidget(tab_widget)
        
        # Button row
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
    
    def create_general_tab(self):
        """Create the general settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # General settings group
        group_box = QGroupBox("General Settings")
        form_layout = QGridLayout()
        
        # Auto tag on add
        row = 0
        self.auto_tag_on_add = QCheckBox("Automatically tag notes when added")
        self.auto_tag_on_add.setChecked(self.config.get("auto_tag_on_add", False))
        form_layout.addWidget(self.auto_tag_on_add, row, 0, 1, 2)
        
        # Auto tag on edit
        row += 1
        self.auto_tag_on_edit = QCheckBox("Automatically tag notes when edited")
        self.auto_tag_on_edit.setChecked(self.config.get("auto_tag_on_edit", False))
        form_layout.addWidget(self.auto_tag_on_edit, row, 0, 1, 2)
        
        # Tag prefix
        row += 1
        form_layout.addWidget(QLabel("Tag prefix:"), row, 0)
        self.tag_prefix = QLineEdit(self.config.get("tag_prefix", "ai:"))
        form_layout.addWidget(self.tag_prefix, row, 1)
        
        # Max tags per card
        row += 1
        form_layout.addWidget(QLabel("Maximum tags per card:"), row, 0)
        self.max_tags = QSpinBox()
        self.max_tags.setMinimum(1)
        self.max_tags.setMaximum(20)
        self.max_tags.setValue(self.config.get("max_tags_per_card", 5))
        form_layout.addWidget(self.max_tags, row, 1)
        
        # Minimum confidence
        row += 1
        form_layout.addWidget(QLabel("Minimum confidence:"), row, 0)
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setMinimum(0.1)
        self.min_confidence.setMaximum(1.0)
        self.min_confidence.setSingleStep(0.1)
        self.min_confidence.setValue(self.config.get("minimum_confidence", 0.7))
        form_layout.addWidget(self.min_confidence, row, 1)
        
        # Use field hints
        row += 1
        self.use_field_hints = QCheckBox("Use field names as hints for tag generation")
        self.use_field_hints.setChecked(self.config.get("use_field_hints", True))
        form_layout.addWidget(self.use_field_hints, row, 0, 1, 2)
        
        # Logging
        row += 1
        self.log_enabled = QCheckBox("Enable logging")
        self.log_enabled.setChecked(self.config.get("log_enabled", True))
        form_layout.addWidget(self.log_enabled, row, 0, 1, 2)
        
        # Log level
        row += 1
        form_layout.addWidget(QLabel("Log level:"), row, 0)
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level.setCurrentText(self.config.get("log_level", "INFO"))
        form_layout.addWidget(self.log_level, row, 1)
        
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_service_tab(self):
        """Create the AI service settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Service type selection
        service_group = QGroupBox("AI Service")
        service_layout = QGridLayout()
        
        row = 0
        service_layout.addWidget(QLabel("Service type:"), row, 0)
        self.service_type = QComboBox()
        services = [
            "OpenAI", "Azure OpenAI", "Anthropic", "Ollama", 
            "OpenRouter", "Local", "Custom", "Gemini", "Perplexity"
        ]
        self.service_type.addItems(services)
        
        # Set current service
        current_service = self.config.get("service_type", "OPENAI")
        if current_service == "OPENAI":
            self.service_type.setCurrentText("OpenAI")
        elif current_service == "AZURE_OPENAI":
            self.service_type.setCurrentText("Azure OpenAI")
        elif current_service == "ANTHROPIC_CLAUDE":
            self.service_type.setCurrentText("Anthropic")
        else:
            # Convert from enum format to display format
            self.service_type.setCurrentText(current_service.capitalize())
        
        self.service_type.currentIndexChanged.connect(self.update_service_settings)
        service_layout.addWidget(self.service_type, row, 1)
        
        service_group.setLayout(service_layout)
        layout.addWidget(service_group)
        
        # Service-specific settings container
        self.service_settings_container = QGroupBox("Service Settings")
        self.service_settings_layout = QGridLayout()
        self.service_settings_container.setLayout(self.service_settings_layout)
        layout.addWidget(self.service_settings_container)
        
        # Initialize with current service settings
        self.update_service_settings()
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def update_service_settings(self):
        """Update service-specific settings based on selected service."""
        # Clear existing widgets
        while self.service_settings_layout.count():
            item = self.service_settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        service = self.service_type.currentText().lower().replace(" ", "_")
        
        # Common settings
        row = 0
        
        # API key (for most services)
        if service not in ["local", "ollama"]:
            self.service_settings_layout.addWidget(QLabel("API Key:"), row, 0)
            self.api_key = QLineEdit(self.config.get(service, {}).get("api_key", ""))
            self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.service_settings_layout.addWidget(self.api_key, row, 1)
            row += 1
        
        # API base URL
        self.service_settings_layout.addWidget(QLabel("API Base URL:"), row, 0)
        self.api_base = QLineEdit(self.config.get(service, {}).get("api_base", ""))
        if service == "openai":
            self.api_base.setText(self.config.get(service, {}).get("api_base", "https://api.openai.com/v1"))
        elif service == "anthropic":
            self.api_base.setText(self.config.get(service, {}).get("api_base", "https://api.anthropic.com/v1"))
        elif service == "ollama":
            self.api_base.setText(self.config.get(service, {}).get("api_base", "http://localhost:11434"))
        elif service == "local":
            self.api_base.setText(self.config.get(service, {}).get("api_base", "http://localhost:8000"))
        self.service_settings_layout.addWidget(self.api_base, row, 1)
        row += 1
        
        # Model selection
        self.service_settings_layout.addWidget(QLabel("Model:"), row, 0)
        self.model = QComboBox()
        
        # Populate model options based on selected service
        if service == "openai":
            for model_id, description in OPENAI_MODELS.items():
                self.model.addItem(f"{model_id} - {description}", model_id)
            current_model = self.config.get(service, {}).get("language_model", "gpt-3.5-turbo")
        elif service == "azure_openai":
            self.model.addItem(self.config.get(service, {}).get("deployment_name", ""))
            current_model = ""  # Azure uses deployment_name instead
        elif service == "anthropic":
            for model_id, description in ANTHROPIC_MODELS.items():
                self.model.addItem(f"{model_id} - {description}", model_id)
            current_model = self.config.get(service, {}).get("model", "claude-2")
        elif service == "gemini":
            for model_id, description in GEMINI_MODELS.items():
                self.model.addItem(f"{model_id} - {description}", model_id)
            current_model = self.config.get(service, {}).get("model", "gemini-pro")
        elif service == "perplexity":
            for model_id, description in PERPLEXITY_MODELS.items():
                self.model.addItem(f"{model_id} - {description}", model_id)
            current_model = self.config.get(service, {}).get("model", "pplx-70b-online")
        else:
            # For other services like ollama, local, openrouter, custom
            self.model.addItem(self.config.get(service, {}).get("model", ""))
            current_model = self.config.get(service, {}).get("model", "")
        
        # Set current model
        if current_model:
            index = self.model.findData(current_model)
            if index >= 0:
                self.model.setCurrentIndex(index)
            else:
                for i in range(self.model.count()):
                    if current_model in self.model.itemText(i):
                        self.model.setCurrentIndex(i)
                        break
        
        self.service_settings_layout.addWidget(self.model, row, 1)
        row += 1
        
        # Max tokens
        self.service_settings_layout.addWidget(QLabel("Max tokens:"), row, 0)
        self.max_tokens = QSpinBox()
        self.max_tokens.setMinimum(1)
        self.max_tokens.setMaximum(4096)
        self.max_tokens.setValue(self.config.get(service, {}).get("max_tokens", 100))
        self.service_settings_layout.addWidget(self.max_tokens, row, 1)
        row += 1
        
        # Temperature
        self.service_settings_layout.addWidget(QLabel("Temperature:"), row, 0)
        self.temperature = QDoubleSpinBox()
        self.temperature.setMinimum(0.0)
        self.temperature.setMaximum(2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(self.config.get(service, {}).get("temperature", 0.7))
        self.service_settings_layout.addWidget(self.temperature, row, 1)
        row += 1
        
        # Timeout
        self.service_settings_layout.addWidget(QLabel("Timeout (seconds):"), row, 0)
        self.timeout = QSpinBox()
        self.timeout.setMinimum(5)
        self.timeout.setMaximum(300)
        self.timeout.setValue(self.config.get(service, {}).get("timeout", 30))
        self.service_settings_layout.addWidget(self.timeout, row, 1)
        row += 1
        
        # Additional service-specific settings
        if service == "azure_openai":
            # Deployment name
            self.service_settings_layout.addWidget(QLabel("Deployment name:"), row, 0)
            self.deployment_name = QLineEdit(self.config.get(service, {}).get("deployment_name", ""))
            self.service_settings_layout.addWidget(self.deployment_name, row, 1)
            row += 1
            
            # API version
            self.service_settings_layout.addWidget(QLabel("API version:"), row, 0)
            self.api_version = QLineEdit(self.config.get(service, {}).get("api_version", "2023-05-15"))
            self.service_settings_layout.addWidget(self.api_version, row, 1)
            row += 1

    def create_tag_generation_tab(self):
        """Create the tag generation settings tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Tag generation settings group
        group_box = QGroupBox("Tag Generation Settings")
        form_layout = QGridLayout()
        
        # Prompt template
        row = 0
        form_layout.addWidget(QLabel("Prompt template:"), row, 0)
        self.prompt_template = QPlainTextEdit()
        self.prompt_template.setPlainText(
            self.config.get("tag_generation", {}).get(
                "prompt_template", 
                "Generate {max_tags_per_card} concise tags for the following flashcard content, separating each tag with a comma:\n\nFront: {front}\nBack: {back}"
            )
        )
        self.prompt_template.setMinimumHeight(100)
        form_layout.addWidget(self.prompt_template, row, 1)
        row += 1
        
        # Include field names
        self.include_field_names = QCheckBox("Include field names in prompt")
        self.include_field_names.setChecked(
            self.config.get("tag_generation", {}).get("include_field_names", True)
        )
        form_layout.addWidget(self.include_field_names, row, 0, 1, 2)
        row += 1
        
        # Exclude fields
        form_layout.addWidget(QLabel("Exclude fields (comma-separated):"), row, 0)
        excluded_fields = ", ".join(self.config.get("tag_generation", {}).get(
            "exclude_fields", ["Extra", "Note ID", "Tags"]))
        self.exclude_fields = QLineEdit(excluded_fields)
        form_layout.addWidget(self.exclude_fields, row, 1)
        row += 1
        
        # Use cached results
        self.use_cached_results = QCheckBox("Use cached tag generation results")
        self.use_cached_results.setChecked(
            self.config.get("tag_generation", {}).get("use_cached_results", True)
        )
        form_layout.addWidget(self.use_cached_results, row, 0, 1, 2)
        row += 1
        
        # Cache duration
        form_layout.addWidget(QLabel("Cache duration (days):"), row, 0)
        self.cache_duration = QSpinBox()
        self.cache_duration.setMinimum(1)
        self.cache_duration.setMaximum(30)
        self.cache_duration.setValue(
            self.config.get("tag_generation", {}).get("cache_duration_days", 7)
        )
        form_layout.addWidget(self.cache_duration, row, 1)
        
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_formats_tab(self):
        """Create the supported formats tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Supported formats group
        group_box = QGroupBox("Supported Document Formats")
        formats_layout = QVBoxLayout()
        
        # Info about supported formats
        support_text = ""
        if PDF_SUPPORT:
            support_text += "✓ PDF files\n"
        else:
            support_text += "✗ PDF files (PyPDF2 not installed)\n"
        if TEXT_SUPPORT:
            support_text += "✓ Text files (.txt)\n"
        
        support_label = QLabel(support_text)
        formats_layout.addWidget(support_label)
        
        # Help text
        help_text = """
If PDF support is not available, you can install the required library with:

pip install PyPDF2

For optimal results, extracted text should be clean and well-structured.
"""
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        formats_layout.addWidget(help_label)
        
        group_box.setLayout(formats_layout)
        layout.addWidget(group_box)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_document_processing_tab(self):
        """Create the document processing tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Document upload group
        upload_group = QGroupBox("Document Upload")
        upload_layout = QGridLayout()
        
        # Relevance thresholds group
        threshold_group = QGroupBox("Relevance Thresholds")
        threshold_layout = QGridLayout()
        
        # High relevance threshold
        row = 0
        threshold_layout.addWidget(QLabel("High relevance threshold:"), row, 0)
        self.high_relevance = QSpinBox()
        self.high_relevance.setMinimum(1)
        self.high_relevance.setMaximum(100)
        self.high_relevance.setValue(70)  # Default from original example
        threshold_layout.addWidget(self.high_relevance, row, 1)
        
        # Medium relevance threshold
        row += 1
        threshold_layout.addWidget(QLabel("Medium relevance threshold:"), row, 0)
        self.medium_relevance = QSpinBox()
        self.medium_relevance.setMinimum(1)
        self.medium_relevance.setMaximum(100)
        self.medium_relevance.setValue(40)  # Default from original example
        threshold_layout.addWidget(self.medium_relevance, row, 1)
        
        # Minimum relevance threshold
        row += 1
        threshold_layout.addWidget(QLabel("Minimum relevance threshold:"), row, 0)
        self.min_relevance = QSpinBox()
        self.min_relevance.setMinimum(1)
        self.min_relevance.setMaximum(100)
        self.min_relevance.setValue(10)  # Default from original example
        threshold_layout.addWidget(self.min_relevance, row, 1)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # File selection
        row = 0
        upload_layout.addWidget(QLabel("Selected file:"), row, 0)
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        upload_layout.addWidget(self.file_path, row, 1)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        upload_layout.addWidget(self.browse_button, row, 2)
        
        row += 1
        
        # Process button
        self.process_button = QPushButton("Process Document")
        self.process_button.clicked.connect(self.process_document)
        self.process_button.setEnabled(False)
        upload_layout.addWidget(self.process_button, row, 0, 1, 3)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
        
        # Processing status
        status_group = QGroupBox("Processing Status")
        status_layout = QVBoxLayout()
        
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        status_layout.addWidget(self.status_text)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Results group
        results_group = QGroupBox("Processing Results")
        results_layout = QGridLayout()
        
        row = 0
        results_layout.addWidget(QLabel("Found cards:"), row, 0)
        self.found_cards = QLabel("0")
        results_layout.addWidget(self.found_cards, row, 1)
        
        row += 1
        results_layout.addWidget(QLabel("Generated tags:"), row, 0)
        self.generated_tags = QLabel("0")
        results_layout.addWidget(self.generated_tags, row, 1)
        
        row += 1
        self.apply_tags_button = QPushButton("Apply Tags to Cards")
        self.apply_tags_button.clicked.connect(self.apply_tags)
        self.apply_tags_button.setEnabled(False)
        results_layout.addWidget(self.apply_tags_button, row, 0, 1, 2)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def browse_file(self):
        """Open file dialog to select a document."""
        file_filter = "Supported files (*.pdf *.txt);;PDF files (*.pdf);;Text files (*.txt)"
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            file_filter
        )
        
        if file_name:
            self.file_path.setText(file_name)
            self.process_button.setEnabled(True)
            self.status_text.clear()
            self.found_cards.setText("0")
        self.found_cards.setText("0")
        self.generated_tags.setText("0")
        self.apply_tags_button.setEnabled(False)
    
    def process_document(self):
        """Process the selected document."""
        try:
            file_path = self.file_path.text()
            if not file_path:
                showWarning("Please select a file first.")
                return
            
            self.status_text.appendPlainText("Processing document...")
            QApplication.processEvents()
            
            # Process the document using DocumentProcessor
            processor = DocumentProcessor()
            text = processor.process_document(file_path)
            
            # Get document name as the base tag (filename without extension)
            import os
            doc_name = os.path.basename(file_path)
            doc_name = os.path.splitext(doc_name)[0]
            
            # Find matching cards with relevance scores
            matching_cards = self._find_matching_cards(text)
            self.found_cards.setText(str(len(matching_cards)))
            
            if matching_cards:
                # Store the document name as the base tag and matching cards with scores
                self._base_tag = doc_name
                self._current_cards = matching_cards
                
                # Get relevance thresholds
                self._high_threshold = self.high_relevance.value()
                self._medium_threshold = self.medium_relevance.value()
                self._min_threshold = self.min_relevance.value()
                
                # Count cards in each relevance category
                high_count = sum(1 for _, score in matching_cards if score >= self._high_threshold)
                medium_count = sum(1 for _, score in matching_cards if score < self._high_threshold and score >= self._medium_threshold)
                low_count = sum(1 for _, score in matching_cards if score < self._medium_threshold and score >= self._min_threshold)
                
                # Update the UI
                self.generated_tags.setText(f"3 ({high_count}/{medium_count}/{low_count})")
                
                self.status_text.appendPlainText(f"Found {len(matching_cards)} matching cards.")
                self.status_text.appendPlainText(f"High relevance: {high_count} cards")
                self.status_text.appendPlainText(f"Medium relevance: {medium_count} cards")
                self.status_text.appendPlainText(f"Low relevance: {low_count} cards")
                self.apply_tags_button.setEnabled(True)
            else:
                self.status_text.appendPlainText("No matching cards found.")
                self.apply_tags_button.setEnabled(False)
        except Exception as e:
            self.status_text.appendPlainText(f"Error: {str(e)}")
            showWarning(f"Error processing document: {str(e)}")
    
    def _find_matching_cards(self, document_text):
        """
        Find cards that match the document content and calculate relevance scores.
        Returns a list of tuples (note, score) where score is 0-100.
        """
        matching_cards = []
        collection = mw.col
        
        # Get all cards from the collection
        for note_id in collection.find_notes(""):
            note = collection.get_note(note_id)
            note_text = " ".join(note.values())
            
            # Calculate a relevance score based on text overlap
            score = self._calculate_relevance(note_text, document_text)
            
            # Add cards with non-zero relevance
            if score > 0:
                matching_cards.append((note, score))
        
        # Sort by relevance score (descending)
        matching_cards.sort(key=lambda x: x[1], reverse=True)
        return matching_cards
    
    def _calculate_relevance(self, note_text, document_text):
        """
        Calculate a relevance score (0-100) between note text and document text.
        This is a simple implementation based on text overlap.
        """
        # Split texts into sentences
        note_sentences = note_text.split('.')
        doc_sentences = document_text.split('.')
        
        # Check for direct matches
        direct_matches = 0
        partial_matches = 0
        
        for note_sent in note_sentences:
            note_sent = note_sent.strip()
            if not note_sent:
                continue
                
            # Check if any document sentence contains this note sentence
            if any(note_sent in doc_sent for doc_sent in doc_sentences):
                direct_matches += 1
            # Check if note sentence contains any keywords from document
            elif any(len(word) > 5 and word in note_sent for doc_sent in doc_sentences for word in doc_sent.split()):
                partial_matches += 1
                
        # Calculate score based on matches
        if direct_matches > 0:
            # Direct matches are weighted higher
            return min(100, 60 + direct_matches * 20 + partial_matches * 5)
        elif partial_matches > 0:
            # Partial matches have a lower base score
            return min(80, 30 + partial_matches * 10)
        
        return 0
    
    def apply_tags(self):
        """Apply relevance tags to matching cards."""
        try:
            if not hasattr(self, '_current_cards') or not hasattr(self, '_base_tag'):
                showWarning("No cards or document tag available. Please process a document first.")
                return
            
            # Get the tag prefix to use
            prefix = self.config.get("tag_prefix", "")
            base_tag = f"{prefix}{self._base_tag}"
            
            # Apply tags based on relevance scores
            tags_applied = 0
            high_count = 0
            medium_count = 0
            low_count = 0
            
            collection = mw.col
            
            for note, score in self._current_cards:
                # Determine the relevance level tag suffix
                if score >= self._high_threshold:
                    tag = f"{base_tag}::1_highly_relevant"
                    high_count += 1
                elif score >= self._medium_threshold:
                    tag = f"{base_tag}::2_somewhat_relevant"
                    medium_count += 1
                elif score >= self._min_threshold:
                    tag = f"{base_tag}::3_minimally_relevant"
                    low_count += 1
                else:
                    continue  # Skip cards below minimum threshold
                
                # Get existing tags
                existing_tags = note.tags.strip().split() if hasattr(note, 'tags') else []
                
                # Add the tag if it doesn't exist
                if tag not in existing_tags:
                    existing_tags.append(tag)
                    note.tags = " ".join(existing_tags)
                    note.flush()
                    tags_applied += 1
            
            # Make sure the changes are committed
            collection.save()
            
            # Update UI
            self.status_text.appendPlainText(f"Applied tags to {tags_applied} cards:")
            self.status_text.appendPlainText(f"  High relevance: {high_count}")
            self
            
        except Exception as e:
            self.status_text.appendPlainText(f"Error: {str(e)}")
            showWarning(f"Error applying tags: {str(e)}")
        
    def on_service_change(self, service):
        """Update the UI based on the selected service."""
        # Always use Gemini
        self.current_service = "gemini"
        
        # Update the model dropdown for Gemini only
        self.model_dropdown.clear()
        self.model_dropdown.addItem("gemini-2.0-flash: Fast and optimized newer Gemini model")
        self.model_dropdown.setEnabled(False)  # Disable changes
    
    def save_settings(self):
        """Save all settings to configuration."""
        try:
            # General settings
            self.config["auto_tag_on_add"] = self.auto_tag_on_add.isChecked()
            self.config["auto_tag_on_edit"] = self.auto_tag_on_edit.isChecked()
            self.config["tag_prefix"] = self.tag_prefix.text()
            self.config["max_tags_per_card"] = self.max_tags.value()
            self.config["minimum_confidence"] = self.min_confidence.value()
            self.config["use_field_hints"] = self.use_field_hints.isChecked()
            self.config["log_enabled"] = self.log_enabled.isChecked()
            self.config["log_level"] = self.log_level.currentText()
            service_display = self.service_type.currentText()
            service_key = service_display.lower().replace(" ", "_")
            
            # Map display name to config service type
            if service_display == "OpenAI":
                self.config["service_type"] = "OPENAI"
            elif service_display == "Azure OpenAI":
                self.config["service_type"] = "AZURE_OPENAI"
            elif service_display == "Anthropic":
                self.config["service_type"] = "ANTHROPIC_CLAUDE"
            else:
                self.config["service_type"] = service_display.upper()
            
            # Ensure service section exists
            if service_key not in self.config:
                self.config[service_key] = {}
            
            # Common service settings
            if hasattr(self, 'api_key') and service_key not in ["local", "ollama"]:
                self.config[service_key]["api_key"] = self.api_key.text()
            
            if hasattr(self, 'api_base'):
                self.config[service_key]["api_base"] = self.api_base.text()
            
            # Model selection
            if hasattr(self, 'model'):
                model_text = self.model.currentText()
                # Extract just the model ID if it's in the format "model_id - description"
                if " - " in model_text:
                    model = model_text.split(" - ")[0].strip()
                else:
                    model = model_text
                
                # Set the appropriate model parameter based on service
                if service_key == "openai":
                    self.config[service_key]["language_model"] = model
                elif service_key == "azure_openai" and hasattr(self, 'deployment_name'):
                    self.config[service_key]["deployment_name"] = self.deployment_name.text()
                else:
                    self.config[service_key]["model"] = model
            
            # Other service parameters
            if hasattr(self, 'max_tokens'):
                self.config[service_key]["max_tokens"] = self.max_tokens.value()
            if hasattr(self, 'temperature'):
                self.config[service_key]["temperature"] = self.temperature.value()
            if hasattr(self, 'timeout'):
                self.config[service_key]["timeout"] = self.timeout.value()
                
            # Azure OpenAI specific settings
            if service_key == "azure_openai" and hasattr(self, 'api_version'):
                self.config[service_key]["api_version"] = self.api_version.text()
            
            # Tag generation settings
            if "tag_generation" not in self.config:
                self.config["tag_generation"] = {}
                
            self.config["tag_generation"]["prompt_template"] = self.prompt_template.toPlainText()
            self.config["tag_generation"]["include_field_names"] = self.include_field_names.isChecked()
            
            # Parse excluded fields list
            excluded_fields = [field.strip() for field in self.exclude_fields.text().split(",") if field.strip()]
            self.config["tag_generation"]["exclude_fields"] = excluded_fields
            
            # Cache settings
            self.config["tag_generation"]["use_cached_results"] = self.use_cached_results.isChecked()
            self.config["tag_generation"]["cache_duration_days"] = self.cache_duration.value()
            
            # Save configuration
            self.config_manager.save_config()
            showInfo("Settings saved successfully!")
            self.accept()
        except Exception as e:
            showWarning(f"Error saving settings: {str(e)}")


def show_main_wizard():
    """Show the main settings dialog for AI Tagger."""
    dialog = SettingsDialog(mw)
    dialog.exec()


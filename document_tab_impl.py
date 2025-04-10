
    def create_document_processing_tab(self):
        """Create the document processing tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Document upload group
        upload_group = QGroupBox("Document Upload")
        upload_layout = QGridLayout()
        
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
            
            # Find matching cards
            matching_cards = self._find_matching_cards(text)
            self.found_cards.setText(str(len(matching_cards)))
            
            if matching_cards:
                # Generate tags using the configured AI service
                tagger = Tagger(self.config)
                tags = tagger.generate_tags(text)
                self.generated_tags.setText(str(len(tags)))
                
                # Store for later use when applying tags
                self._current_cards = matching_cards
                self._current_tags = tags
                
                self.status_text.appendPlainText(f"Found {len(matching_cards)} matching cards.")
                self.status_text.appendPlainText(f"Generated {len(tags)} tags.")
                self.apply_tags_button.setEnabled(True)
            else:
                self.status_text.appendPlainText("No matching cards found.")
                self.apply_tags_button.setEnabled(False)
            
        except Exception as e:
            self.status_text.appendPlainText(f"Error: {str(e)}")
            showWarning(f"Error processing document: {str(e)}")
    
    def _find_matching_cards(self, document_text):
        """Find cards that match the document content."""
        matching_cards = []
        collection = mw.col
        
        # Get all cards from the collection
        for note_id in collection.find_notes(""):
            note = collection.get_note(note_id)
            note_text = " ".join(note.values())
            
            # Simple matching based on text overlap
            if any(sent in document_text or document_text in sent 
                   for sent in note_text.split('.')):
                matching_cards.append(note)
        
        return matching_cards
    
    def apply_tags(self):
        """Apply generated tags to matching cards."""
        try:
            if not hasattr(self, '_current_cards') or not hasattr(self, '_current_tags'):
                showWarning("No cards or tags available. Please process a document first.")
                return
            
            # Apply tags using the Tagger class
            tagger = Tagger(self.config)
            tags_applied = 0
            
            for note in self._current_cards:
                if tagger.apply_tags(note, self._current_tags):
                    tags_applied += 1
            
            self.status_text.appendPlainText(f"Applied tags to {tags_applied} cards.")
            showInfo(f"Successfully applied tags to {tags_applied} cards.")
            
            # Clear current cards and tags
            self._current_cards = None
            self._current_tags = None
            self.apply_tags_button.setEnabled(False)
            
        except Exception as e:
            self.status_text.appendPlainText(f"Error: {str(e)}")
            showWarning(f"Error applying tags: {str(e)}")

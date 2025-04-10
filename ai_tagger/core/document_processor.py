import os
import sys
import logging
import traceback
import re
from typing import Optional, Dict, Any, List

# Add the local libs directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
if os.path.exists(libs_path) and libs_path not in sys.path:
    sys.path.insert(0, libs_path)
    logging.info(f"Added libs path to Python path: {libs_path}")

# Try importing document processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 not installed. Cannot process PDF documents.")

# No PowerPoint support - removed intentionally


class DocumentProcessingError(Exception):
    """Exception raised when there's an error processing a document."""
    pass

class DocumentProcessor:
    """Processes documents and extracts their text content."""
    
    def __init__(self, max_chunk_size: int = 4000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            max_chunk_size: Maximum size of each document chunk (in approximate tokens)
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, file_path, chunk: bool = True):
        """
        Process a document and extract its text.
        
        Args:
            file_path: Path to the document
            chunk: Whether to chunk the document if it's large (default: True)
            
        Returns:
            Extracted text from the document
            
        Raises:
            DocumentProcessingError: If there's an error processing the document
        """
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                text = self._process_pdf(file_path)
            elif file_ext == '.txt':
                text = self._process_text(file_path)
            else:
                raise DocumentProcessingError(f"Unsupported file format: {file_ext}. Only PDF and TXT files are supported.")
                
            # If chunking is enabled and the document is large, chunk it
            if chunk:
                # Estimate tokens - roughly 4 characters per token as a heuristic
                estimated_tokens = len(text) // 4
                if estimated_tokens > self.max_chunk_size:
                    logging.info(f"Document is large (est. {estimated_tokens} tokens). Chunking into smaller segments.")
                    return self._chunk_text(text)
                
            return text
        except Exception as e:
            raise DocumentProcessingError(f"Error processing document: {str(e)}")
        
    def _process_pdf(self, file_path):
        """Extract text from a PDF file."""
        if not PDF_SUPPORT:
            raise DocumentProcessingError("PDF processing is not supported. PyPDF2 is not installed.")
            
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text
        
    def _process_text(self, file_path):
        """Extract text from a simple text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
                
    def _chunk_text(self, text: str) -> str:
        """
        Split a large text into smaller chunks, then rejoin with markers.
        
        Args:
            text: The text to chunk
            
        Returns:
            Chunked text with section markers
        """
        chunks = self._create_chunks(text)
        
        # Join chunks with section markers
        result = ""
        for i, chunk in enumerate(chunks):
            result += f"\n\n--- SECTION {i+1} OF {len(chunks)} ---\n\n"
            result += chunk
            
        return result
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Create chunks from text based on estimated token count.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        # If text is small enough, return as is
        if len(text) // 4 <= self.max_chunk_size:
            return [text]
            
        chunks = []
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            # Estimate paragraph size in tokens
            paragraph_size = len(paragraph) // 4
            
            # If adding this paragraph would exceed max size, start a new chunk
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - (self.chunk_overlap * 4))
                current_chunk = current_chunk[overlap_point:] if overlap_point > 0 else ""
                current_size = len(current_chunk) // 4
                
            # Add paragraph to current chunk
            current_chunk += ("\n\n" if current_chunk else "") + paragraph
            current_size += paragraph_size
            
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        # If we couldn't create proper paragraph-based chunks, fall back to character-based chunking
        if not chunks:
            chunk_size_chars = self.max_chunk_size * 4
            overlap_chars = self.chunk_overlap * 4
            
            for i in range(0, len(text), chunk_size_chars - overlap_chars):
                end = min(i + chunk_size_chars, len(text))
                chunks.append(text[i:end])
                
                # If we've reached the end, stop
                if end == len(text):
                    break
                    
        return chunks
    
    def process_document_as_chunks(self, file_path) -> List[str]:
        """
        Process a document and return its content as a list of chunks.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of text chunks from the document
            
        Raises:
            DocumentProcessingError: If there's an error processing the document
        """
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                text = self._process_pdf(file_path)
            elif file_ext == '.txt':
                text = self._process_text(file_path)
            else:
                raise DocumentProcessingError(f"Unsupported file format: {file_ext}. Only PDF and TXT files are supported.")
                
            return self._create_chunks(text)
        except Exception as e:
            raise DocumentProcessingError(f"Error processing document: {str(e)}")

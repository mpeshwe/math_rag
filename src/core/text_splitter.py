"""
Intelligent text splitting for mathematical content.
Preserve semantic coherence in chunks.
"""
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re
import logging

logger = logging.getLogger(__name__)


class MathematicalTextSplitter:
    """
    Specialized text splitter for mathematical content.
    
    Industry practice: Domain-specific chunking strategies improve retrieval.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_chunk_size: int = 1500):
        """
        Initialize the mathematical text splitter.
        
        Args:
            chunk_size: Target size for chunks
            chunk_overlap: Overlap between chunks to preserve context
            max_chunk_size: Maximum allowed chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        
        # Define separators that respect mathematical structure
        self.separators = [
            "\n\n",  # Paragraph breaks
            "\nSolution:",  # Problem-solution boundary
            "\nProblem:",  # Problem boundary
            "\n```",  # Code block boundaries
            "\n",  # Line breaks
            ". ",  # Sentence boundaries
            " ",  # Word boundaries
            ""  # Character boundaries (last resort)
        ]
        
        # Create base splitter
        self.base_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def should_keep_together(self, text: str) -> bool:
        """
        Determine if a piece of text should be kept together.
        
        Industry practice: Identify semantic units that shouldn't be split.
        """
        # Keep short mathematical expressions together
        if len(text) < 100 and ('$' in text or '\\' in text):
            return True
            
        # Keep code blocks together if reasonably sized
        if '<code>' in text and '</code>' in text and len(text) < self.max_chunk_size:
            return True
            
        # Keep problem statements together
        if text.strip().startswith('Problem:') and len(text) < self.chunk_size * 1.5:
            return True
            
        return False
    
    def extract_problem_solution_pairs(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract problem-solution pairs from mathematical content.
        
        Industry practice: Maintain semantic relationships in chunks.
        """
        pairs = []
        
        # Split on "Problem:" to find problem boundaries
        parts = text.split("Problem:")
        
        for part in parts[1:]:  # Skip first empty part
            if "Solution:" in part:
                problem_part, solution_part = part.split("Solution:", 1)
                problem_text = f"Problem: {problem_part.strip()}"
                solution_text = f"Solution: {solution_part.strip()}"
                pairs.append((problem_text, solution_text))
            else:
                # Problem without clear solution
                problem_text = f"Problem: {part.strip()}"
                pairs.append((problem_text, ""))
        
        return pairs
    
    def create_chunks_from_pairs(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """
        Create chunks from problem-solution pairs.
        
        Industry practice: Balance chunk size with semantic coherence.
        """
        chunks = []
        
        for problem, solution in pairs:
            combined = f"{problem}\n\n{solution}".strip()
            
            # If the complete pair fits in one chunk, keep it together
            if len(combined) <= self.max_chunk_size:
                chunks.append(combined)
            else:
                # Split large pairs while trying to preserve structure
                if len(problem) <= self.chunk_size:
                    # Problem fits in one chunk, split solution
                    chunks.append(problem)
                    
                    # Split solution into chunks
                    solution_chunks = self.base_splitter.split_text(solution)
                    chunks.extend(solution_chunks)
                else:
                    # Both problem and solution are large, split normally
                    all_chunks = self.base_splitter.split_text(combined)
                    chunks.extend(all_chunks)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into appropriately sized chunks for embedding.
        
        Industry practice: Maintain document metadata through chunking.
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        chunked_documents = []
        
        for doc_idx, document in enumerate(documents):
            text = document.page_content
            
            # Check if document is already appropriately sized
            if len(text) <= self.chunk_size:
                chunked_documents.append(document)
                continue
            
            # Try to extract problem-solution pairs first
            pairs = self.extract_problem_solution_pairs(text)
            
            if pairs:
                # Use semantic chunking for problem-solution content
                chunks = self.create_chunks_from_pairs(pairs)
            else:
                # Fall back to standard chunking
                chunks = self.base_splitter.split_text(text)
            
            # Create new documents for each chunk
            for chunk_idx, chunk_text in enumerate(chunks):
                if chunk_text.strip():  # Skip empty chunks
                    # Create metadata for chunk
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "parent_doc_id": chunk_metadata.get("doc_id"),
                        "chunk_id": f"{chunk_metadata.get('doc_id', doc_idx)}_{chunk_idx}",
                        "chunk_size": len(chunk_text)
                    })
                    
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    )
                    chunked_documents.append(chunk_doc)
        
        logger.info(f"Created {len(chunked_documents)} chunks from {len(documents)} documents")
        return chunked_documents
    
    def get_chunking_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about the chunking process."""
        chunk_sizes = [len(doc.page_content) for doc in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "chunks_with_problems": sum(1 for doc in chunks if "Problem:" in doc.page_content),
            "chunks_with_solutions": sum(1 for doc in chunks if "Solution:" in doc.page_content)
        }
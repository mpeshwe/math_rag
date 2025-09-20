"""
Document processing for RAG ingestion.
"""
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from ..models.data_models import MathematicalProblem, ProcessedDocument
import uuid
import logging
import re
from typing import Tuple



logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Converts mathematical problems into LangChain Documents for RAG.
    
    Industry practice: Separate document processing logic for maintainability
    and testability.
    """
    
    def __init__(self):
        self.processed_count = 0
        self.quality_threshold = 0.5  # Minimum quality score to include
    
    def convert_to_langchain_document(self, problem: MathematicalProblem) -> Document:
        """
        Convert a MathematicalProblem to a LangChain Document.
        
        Industry practice: Create rich metadata for better retrieval.
        """
        # Combine query and response for embedding
        # Format: "Problem: [query]\n\nSolution: [response]"
        content = f"Problem: {problem.query}\n\nSolution: {problem.response}"
        
        # Create rich metadata for retrieval filtering and ranking
        metadata = {
            "source": "rstar_sft_dataset",
            "problem_type": "mathematical",
            "query_length": problem.query_length,
            "response_length": problem.response_length,
            "has_latex": problem.has_latex,
            "has_code": problem.has_code,
            "difficulty_indicators": problem.difficulty_indicators,
            "quality_score": self._calculate_quality_score(problem),
            "doc_id": str(uuid.uuid4())
        }
        
        return Document(page_content=content, metadata=metadata)
    
    def _calculate_quality_score(self, problem: MathematicalProblem) -> float:
        """
        Calculate a quality score for the mathematical problem.
        
        Industry practice: Quality scoring helps filter noisy data.
        Higher scores indicate better content for RAG.
        """
        score = 0.5  # Base score
        
        # Length indicators (not too short, not too long)
        if 50 <= problem.query_length <= 500:
            score += 0.1
        if 100 <= problem.response_length <= 2000:
            score += 0.1
            
        # Content richness indicators
        if problem.has_latex:
            score += 0.1  # Mathematical notation is good
        if problem.has_code:
            score += 0.1  # Code solutions are valuable
        if problem.difficulty_indicators:
            score += 0.1  # Advanced topics are useful
            
        # Penalize potential issues
        if problem.query_length < 20:
            score -= 0.2  # Too short queries are often low quality
        if problem.response_length < 50:
            score -= 0.2  # Too short responses are incomplete
            
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def process_batch(self, problems: List[MathematicalProblem]) -> List[Document]:
        """
        Process a batch of mathematical problems into LangChain Documents.
        
        Industry practice: Batch processing with quality filtering.
        """
        logger.info(f"Processing batch of {len(problems)} problems...")
        
        documents = []
        filtered_count = 0
        
        for problem in problems:
            doc = self.convert_to_langchain_document(problem)
            
            # Quality filtering
            if doc.metadata["quality_score"] >= self.quality_threshold:
                documents.append(doc)
            else:
                filtered_count += 1
                
        self.processed_count += len(documents)
        
        logger.info(f"Batch complete: {len(documents)} documents created, "
                   f"{filtered_count} filtered out for low quality")
        
        return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about document processing."""
        return {
            "total_processed": self.processed_count,
            "quality_threshold": self.quality_threshold
        }
    def clean_mathematical_text(self, text: str) -> str:
        """
        Clean mathematical text for better embedding.
        
        Fixed version that maintains proper code block balance.
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize LaTeX delimiters
        text = re.sub(r'\\\(', '$', text)  # \( -> $
        text = re.sub(r'\\\)', '$', text)  # \) -> $
        text = re.sub(r'\\\[', '$$', text)  # \[ -> $$
        text = re.sub(r'\\\]', '$$', text)  # \] -> $$
        
        # Clean code blocks more carefully
        # Replace <code> and </code> tags but ensure balance
        if '<code>' in text and '</code>' in text:
            text = re.sub(r'<code>', '\n```python\n', text)
            text = re.sub(r'</code>', '\n```\n', text)
        elif '<code>' in text:
            # Unmatched opening tag - remove it
            text = text.replace('<code>', '')
        
        # Clean step markers
        text = re.sub(r'<end_of_step>', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()

    def extract_problem_solution_pair(self, problem: MathematicalProblem) -> Tuple[str, str]:
        """
        Extract and clean problem-solution pair.
        
        Industry practice: Separate problem and solution for better retrieval.
        """
        cleaned_query = self.clean_mathematical_text(problem.query)
        cleaned_response = self.clean_mathematical_text(problem.response)
        
        return cleaned_query, cleaned_response

    def convert_to_langchain_document(self, problem: MathematicalProblem) -> Document:
        """
        Convert a MathematicalProblem to a LangChain Document.
        
        Updated with text cleaning.
        """
        # Clean the text first
        cleaned_query, cleaned_response = self.extract_problem_solution_pair(problem)
        
        # Combine query and response for embedding
        content = f"Problem: {cleaned_query}\n\nSolution: {cleaned_response}"
        
        # Create rich metadata for retrieval filtering and ranking
        metadata = {
            "source": "rstar_sft_dataset",
            "problem_type": "mathematical",
            "query_length": len(cleaned_query),
            "response_length": len(cleaned_response),
            "has_latex": problem.has_latex,
            "has_code": problem.has_code,
            "difficulty_indicators": problem.difficulty_indicators,
            "quality_score": self._calculate_quality_score(problem),
            "doc_id": str(uuid.uuid4()),
            "original_query_length": problem.query_length,
            "original_response_length": problem.response_length
        }
        
        return Document(page_content=content, metadata=metadata)
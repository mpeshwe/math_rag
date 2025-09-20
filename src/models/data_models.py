"""
Pydantic models for mathematical dataset.
Industry practice: Type safety prevents runtime errors and improves code maintainability.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, computed_field
import re


class MathematicalProblem(BaseModel):
    """
    Represents a single mathematical problem from the dataset.
    
    Industry practice: Use descriptive field names and validation
    to ensure data quality.
    """
    query: str = Field(..., description="The mathematical question or problem statement")
    response: str = Field(..., description="The step-by-step solution with code")
    
    @field_validator('query', 'response')
    @classmethod
    def validate_non_empty(cls, v):
        """Ensure query and response are not empty."""
        if not v or not v.strip():
            raise ValueError("Query and response cannot be empty")
        return v.strip()
    
    @computed_field
    @property
    def query_length(self) -> int:
        """Length of query in characters."""
        return len(self.query) if self.query else 0
    
    @computed_field
    @property
    def response_length(self) -> int:
        """Length of response in characters."""
        return len(self.response) if self.response else 0
    
    @computed_field
    @property
    def has_latex(self) -> bool:
        """Whether query contains LaTeX notation."""
        if not self.query:
            return False
        latex_patterns = [r'\$.*?\$', r'\\[()]', r'\\begin\{.*?\}', r'\\[a-zA-Z]+']
        return any(re.search(pattern, self.query) for pattern in latex_patterns)
    
    @computed_field
    @property
    def has_code(self) -> bool:
        """Whether response contains code."""
        if not self.response:
            return False
        return '<code>' in self.response or 'import ' in self.response
    
    @computed_field
    @property
    def difficulty_indicators(self) -> List[str]:
        """Keywords indicating problem difficulty."""
        if not self.query:
            return []
        
        difficulty_keywords = [
            'prove', 'theorem', 'lemma', 'integral', 'derivative', 
            'matrix', 'eigenvalue', 'optimization', 'convergence',
            'probability', 'statistics', 'geometry', 'topology'
        ]
        query_lower = self.query.lower()
        found_keywords = [kw for kw in difficulty_keywords if kw in query_lower]
        return found_keywords

    model_config = {
        # Remove validate_assignment to prevent recursion
        "arbitrary_types_allowed": True
    }


class ProcessedDocument(BaseModel):
    """
    Represents a document processed for RAG ingestion.
    
    This model represents how we'll store documents in our vector database.
    """
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="The main content to be embedded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_type: str = Field(default="mathematical_problem", description="Type of source document")
    
    # RAG-specific fields
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate embeddings")
    chunk_index: Optional[int] = Field(default=None, description="Index if document was split into chunks")
    parent_id: Optional[str] = Field(default=None, description="ID of parent document if this is a chunk")

    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v):
        """Ensure content is not too short or too long for embedding."""
        if len(v.strip()) < 10:
            raise ValueError("Content too short for meaningful embedding")
        if len(v) > 8000:  # Most embedding models have token limits
            raise ValueError("Content too long for embedding model")
        return v.strip()


class DatasetStats(BaseModel):
    """
    Statistics about the loaded dataset.
    Industry practice: Always track data quality metrics.
    """
    total_examples: int
    avg_query_length: float
    avg_response_length: float
    latex_percentage: float
    code_percentage: float
    difficulty_distribution: Dict[str, int]
    
    model_config = {
        "arbitrary_types_allowed": True
    }
"""
Embedding generators using Sentence-Transformers.
"""
from typing import List, Optional, Dict, Any 
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

from pathlib import Path
import json


logger = logging.getLogger(__name__)
#all-MiniLM-L6-v2
class MathematicalEmbeddingGenerator: 
    """
    Generate embeddings optimized for math content
    """
    def __init__(self,
                 model_name:str = "/home/manas/Documents/Projects/RAG/math_rag/math-embeddings-model",
                 cache_dir:str = "data/embeddings",):
        """
        Args:
            model_name (str): Name of the Sentence-Transformers model to use.
            cache_dir (str): Directory to cache the model.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load the model
        logger.info(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded with embedding dimension {self.embedding_dim}")


        # Stats 
        self.generation_stats = {
            "total_embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text for embedding
        cleaned_text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
        
        # Ensure numpy array (newer versions sometimes return lists)
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        self.generation_stats["total_embeddings_generated"] += 1
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text to better handle mathematical content.
        This can include normalizing whitespace, handling special characters, etc.

        Args:
            text (str): The input text.
        """
        # remove excessive whitespace
        text = ' '.join(text.split())


        # truncate very long texts to avoid memory issues
        max_length = 1000  # characters
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.debug(f"Truncated text to {max_length} characters.")
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'Unknown'),
            "generation_stats": self.generation_stats.copy()
        }
    def test_embedding_quality(self, sample_texts: List[str]) -> Dict[str, Any] :
        """
        Test embedding quality with sample mathematical texts.
        """

        if len(sample_texts) < 2 : 
            raise ValueError("Need at least 2 texts for similarity testing")
        logger.info("Testing embedding quality with sample texts...")
        # Generate embeddings

        embeddings = self.generate_batch_embeddings(sample_texts, show_progress=False)
        similarities = []
        # calculate pairwise cosine similarities
        for i in range(len(embeddings)) : 
            for j in range(i+1, len(embeddings)) :
                # cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    )
                similarities.append(sim)

        # Analyze similarities
        similarities = np.array(similarities)

        quality_report = {
            "num_comparisons": len(similarities),
            "mean_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities)),
            "embedding_dim": self.embedding_dim,
            "sample_embedding_norms": [float(np.linalg.norm(emb)) for emb in embeddings],
        }
        logger.info(f"Embedding quality test complete. Avg similarity: {quality_report['mean_similarity']:.3f}")
        return quality_report
    def generate_batch_embeddings(self, 
                            texts: List[str], 
                            batch_size: int = 32,
                            show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Industry practice: Batch processing for better performance.
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Preprocess all texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Ensure all embeddings are numpy arrays
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Convert to list of numpy arrays
        embedding_list = [np.array(emb) if not isinstance(emb, np.ndarray) else emb 
                        for emb in embeddings]
        
        self.generation_stats["total_embeddings_generated"] += len(texts)
        
        logger.info(f"Generated {len(embedding_list)} embeddings")
        return embedding_list
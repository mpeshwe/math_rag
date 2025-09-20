"""
Test script for embedding generation.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import MathematicalEmbeddingGenerator

def test_embeddings():
    print("Testing Mathematical Embedding Generation")
    print()
    
    # Load sample data
    print("Loading sample mathematical problems...")
    loader = MathDatasetLoader()
    problems = loader.process_examples(limit=10, use_cache=True)
    
    # Create documents
    processor = DocumentProcessor()
    documents = processor.process_batch(problems)
    
    print(f"Created {len(documents)} documents")
    
    # Initialize embedding generator
    print("Initializing embedding generator...")
    embedder = MathematicalEmbeddingGenerator()
    
    # Get model info
    model_info = embedder.get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Embedding dimension: {model_info['embedding_dimension']}")
    
    # Test single embedding
    print("\nTesting single embedding...")
    sample_text = documents[0].page_content[:200]
    embedding = embedder.generate_embedding(sample_text)
    print(f"Generated embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.3f}")
    
    # Test batch embeddings
    print("\nTesting batch embeddings...")
    texts = [doc.page_content for doc in documents[:5]]
    batch_embeddings = embedder.generate_batch_embeddings(texts, show_progress=True)
    print(f"Generated {len(batch_embeddings)} embeddings")
    
    # Test embedding quality
    print("\nTesting embedding quality...")
    quality_report = embedder.test_embedding_quality(texts)
    
    print(f"Quality Report:")
    for key, value in quality_report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, list):
            if len(value) <= 5:
                formatted_list = [f"{v:.3f}" for v in value]
                print(f"  {key}: [{', '.join(formatted_list)}]")
            else:
                print(f"  {key}: [list of {len(value)} items]")
        else:
            print(f"  {key}: {value}")
    
    # Show statistics
    stats = embedder.get_model_info()["generation_stats"]
    print(f"\nGeneration Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    import numpy as np
    test_embeddings()
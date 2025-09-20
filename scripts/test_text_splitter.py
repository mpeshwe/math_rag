"""
Test script for mathematical text splitting functionality.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader
from src.core.document_processor import DocumentProcessor
from src.core.text_splitter import MathematicalTextSplitter

def test_text_splitter():
    print("Testing Mathematical Text Splitting")
    print()
    
    # Load and process sample data
    print("Loading sample data...")
    loader = MathDatasetLoader()
    problems = loader.process_examples(limit=50)
    
    processor = DocumentProcessor()
    documents = processor.process_batch(problems)
    
    print(f"Created {len(documents)} documents")
    
    # Test text splitting
    print("Splitting documents into chunks...")
    splitter = MathematicalTextSplitter(
        chunk_size=800,  # Smaller for testing
        chunk_overlap=100
    )
    
    chunked_docs = splitter.split_documents(documents)
    
    # Analyze results
    stats = splitter.get_chunking_stats(chunked_docs)
    
    print(f"Chunking Results:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print()
    print("Sample chunks:")
    for i, chunk in enumerate(chunked_docs[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Size: {len(chunk.page_content)} chars")
        print(f"Content: {chunk.page_content[:200]}...")
        print(f"Metadata: chunk_index={chunk.metadata.get('chunk_index')}, "
              f"quality_score={chunk.metadata.get('quality_score', 'N/A')}")

if __name__ == "__main__":
    test_text_splitter()
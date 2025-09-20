"""
Test script for document processing functionality.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader
from src.core.document_processor import DocumentProcessor

def test_document_processor():
    print("Testing Document Processing Pipeline")
    print()
    
    # Load sample data
    print("Loading sample mathematical problems...")
    loader = MathDatasetLoader()
    problems = loader.process_examples(limit=100)  # Small sample for testing
    
    print(f"Loaded {len(problems)} problems")
    
    # Process into documents
    print("Converting to LangChain documents...")
    processor = DocumentProcessor()
    documents = processor.process_batch(problems)
    
    print(f"Created {len(documents)} documents")
    
    # Analyze results
    print()
    print("Document Analysis:")
    
    if documents:
        # Sample document
        sample_doc = documents[0]
        print(f"Sample document content (first 200 chars):")
        print(f"{sample_doc.page_content[:200]}...")
        print()
        print(f"Sample metadata:")
        for key, value in sample_doc.metadata.items():
            print(f"  {key}: {value}")
        
        print()
        # Quality score distribution
        quality_scores = [doc.metadata["quality_score"] for doc in documents]
        avg_quality = sum(quality_scores) / len(quality_scores)
        max_quality = max(quality_scores)
        min_quality = min(quality_scores)
        
        print(f"Quality Statistics:")
        print(f"  Average quality: {avg_quality:.3f}")
        print(f"  Max quality: {max_quality:.3f}")
        print(f"  Min quality: {min_quality:.3f}")
        
        # Content analysis
        has_latex_count = sum(1 for doc in documents if doc.metadata["has_latex"])
        has_code_count = sum(1 for doc in documents if doc.metadata["has_code"])
        
        print(f"  Documents with LaTeX: {has_latex_count} ({has_latex_count/len(documents)*100:.1f}%)")
        print(f"  Documents with code: {has_code_count} ({has_code_count/len(documents)*100:.1f}%)")
        
    # Processing stats
    stats = processor.get_processing_stats()
    print()
    print(f"Processing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_document_processor()
"""
Test script for quality assurance pipeline.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader
from src.core.document_processor import DocumentProcessor
from src.core.text_splitter import MathematicalTextSplitter
from src.core.quality_assurance import DocumentQualityValidator, PipelineTester

def test_quality_assurance():
    print("Testing Quality Assurance Pipeline")
    print()
    
    # Run automated tests first
    print("1. Running automated pipeline tests...")
    tester = PipelineTester()
    test_results = tester.run_all_tests()
    
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"   {test_name}: {status}")
    
    print()
    
    # Test document validation
    print("2. Testing document validation...")
    
    # Create sample documents
    loader = MathDatasetLoader()
    problems = loader.process_examples(limit=20)
    
    processor = DocumentProcessor()
    documents = processor.process_batch(problems)
    
    splitter = MathematicalTextSplitter()
    chunks = splitter.split_documents(documents)
    
    # Validate documents
    validator = DocumentQualityValidator()
    validation_report = validator.validate_batch(chunks)
    
    print(f"Validation Results:")
    print(f"   Total documents: {validation_report['total_documents']}")
    print(f"   Valid documents: {validation_report['valid_documents']}")
    print(f"   Invalid documents: {validation_report['invalid_documents']}")
    print(f"   Validation rate: {validation_report['validation_rate']:.1f}%")
    print(f"   Documents with warnings: {validation_report['documents_with_warnings']}")
    
    if validation_report['common_issues']:
        print(f"\n   Most common issues:")
        for issue, count in validation_report['common_issues']:
            print(f"     {issue}: {count} times")
    
    if validation_report['common_warnings']:
        print(f"\n   Most common warnings:")
        for warning, count in validation_report['common_warnings']:
            print(f"     {warning}: {count} times")
    
    # Show validation stats
    stats = validator.get_validation_stats()
    print(f"\n   Cumulative validation stats:")
    for key, value in stats.items():
        print(f"     {key}: {value}")

if __name__ == "__main__":
    test_quality_assurance()
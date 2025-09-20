"""
Test script for data loading and processing pipeline.
"""
import os
import sys

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import our modules
from src.data.dataset_loader import MathDatasetLoader

def test_data_pipeline():
    """Test the complete data loading and processing pipeline."""
    print(" Testing Mathematical Dataset Pipeline")
    print()
    
    # Initialize loader
    loader = MathDatasetLoader()
    
    # Test with small sample first (industry practice: start small)
    print(" Processing sample data (1000 examples)...")
    processed_problems = loader.process_examples(limit=1000)
    
    print(f" Processed {len(processed_problems)} problems")
    
    # Get statistics
    stats = loader.get_dataset_stats()
    print()
    print(" Dataset Statistics:")
    print(f"   Total examples: {stats.total_examples}")
    print(f"   Avg query length: {stats.avg_query_length:.1f} chars")
    print(f"   Avg response length: {stats.avg_response_length:.1f} chars")
    print(f"   LaTeX usage: {stats.latex_percentage:.1f}%")
    print(f"   Code usage: {stats.code_percentage:.1f}%")
    
    print()
    print(" Difficulty indicators found:")
    for keyword, count in sorted(stats.difficulty_distribution.items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"   {keyword}: {count} problems")
    
    # Show sample problems
    samples = loader.get_sample_problems(3)
    print()
    print(" Sample processed problems:")
    for i, problem in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Query: {problem.query[:100]}...")
        print(f"Response: {problem.response[:100]}...")
        print(f"Has LaTeX: {problem.has_latex}")
        print(f"Has Code: {problem.has_code}")
        print(f"Difficulty: {problem.difficulty_indicators}")

if __name__ == "__main__":
    test_data_pipeline()
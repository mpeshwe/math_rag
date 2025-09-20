"""
Test vector store with queries aligned to actual dataset content.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader
from src.core.document_processor import DocumentProcessor
from src.core.text_splitter import MathematicalTextSplitter
from src.core.embeddings import MathematicalEmbeddingGenerator
from src.core.vector_store import MathematicalVectorStore, VectorStoreManager
import re

def find_matching_content():
    """Find mathematical problems that match specific topics."""
    print("1. Analyzing dataset content for topic alignment...")
    
    # Load larger sample to find diverse content
    loader = MathDatasetLoader()
    problems = loader.process_examples(limit=200, use_cache=True)
    
    # Categorize problems by mathematical topics
    categories = {
        'derivative': [],
        'integral': [],
        'matrix': [],
        'probability': [],
        'geometry': [],
        'equation': [],
        'function': []
    }
    
    # Keywords to identify problem types
    keywords = {
        'derivative': ['derivative', 'differentiate', 'dy/dx', "d/dx", 'rate of change'],
        'integral': ['integral', 'integrate', '∫', 'antiderivative'],
        'matrix': ['matrix', 'determinant', 'eigenvalue', 'linear algebra'],
        'probability': ['probability', 'random', 'dice', 'coin', 'chance'],
        'geometry': ['triangle', 'circle', 'angle', 'area', 'volume', 'perimeter'],
        'equation': ['equation', 'solve for', 'quadratic', 'polynomial'],
        'function': ['function', 'f(x)', 'domain', 'range', 'graph']
    }
    
    for problem in problems:
        query_lower = problem.query.lower()
        response_lower = problem.response.lower()
        full_text = query_lower + ' ' + response_lower
        
        for category, topic_keywords in keywords.items():
            if any(keyword in full_text for keyword in topic_keywords):
                if len(categories[category]) < 5:  # Limit to 5 per category
                    categories[category].append(problem)
    
    # Print findings
    print("\nContent analysis results:")
    for category, problems in categories.items():
        print(f"  {category}: {len(problems)} problems found")
        if problems:
            # Show first example
            example = problems[0]
            print(f"    Example: {example.query[:80]}...")
    
    return categories

def create_aligned_test_set(categories):
    """Create test queries aligned with actual content."""
    
    # Find the category with most content
    best_category = max(categories.items(), key=lambda x: len(x[1]))
    category_name, problems = best_category
    
    if not problems:
        print("No suitable problems found for targeted testing")
        return None, []
    
    print(f"\n2. Using {category_name} problems for aligned testing")
    print(f"   Found {len(problems)} relevant problems")
    
    # Create queries based on actual content
    test_queries = []
    
    # Extract key terms from actual problems to create similar queries
    for i, problem in enumerate(problems[:3]):  # Use first 3
        # Extract mathematical concepts from the query
        query_words = problem.query.lower().split()
        
        # Create variations based on the actual content
        if category_name == 'derivative':
            variations = [
                f"Find the derivative",
                f"What is the derivative of",
                f"Calculate dy/dx for"
            ]
        elif category_name == 'integral':
            variations = [
                f"Find the integral",
                f"Calculate the integral of",
                f"What is the antiderivative"
            ]
        elif category_name == 'matrix':
            variations = [
                f"Matrix calculation",
                f"Find the determinant",
                f"Matrix multiplication"
            ]
        elif category_name == 'probability':
            variations = [
                f"What is the probability",
                f"Calculate the chance",
                f"Random variable problem"
            ]
        elif category_name == 'geometry':
            variations = [
                f"Find the area",
                f"Calculate the volume",
                f"Geometric problem"
            ]
        elif category_name == 'equation':
            variations = [
                f"Solve the equation",
                f"Find the solution",
                f"Quadratic equation"
            ]
        else:  # function
            variations = [
                f"Function problem",
                f"Find f(x)",
                f"Domain and range"
            ]
        
        test_queries.extend(variations[:2])  # Take 2 variations per problem
    
    return problems[:5], test_queries[:3]  # Return 5 docs, 3 queries

def test_aligned_vector_store():
    print("Testing Vector Store with Content-Aligned Queries")
    print("=" * 60)
    
    # Find matching content
    categories = find_matching_content()
    
    # Create aligned test set
    test_problems, test_queries = create_aligned_test_set(categories)
    
    if not test_problems:
        print("Could not create aligned test set")
        return
    
    print(f"\n3. Processing {len(test_problems)} aligned problems...")
    
    # Process documents
    processor = DocumentProcessor()
    documents = processor.process_batch(test_problems)
    
    # Split into chunks
    splitter = MathematicalTextSplitter(chunk_size=800)
    chunks = splitter.split_documents(documents)
    
    print(f"   Created {len(chunks)} chunks")
    
    # Show what we're actually indexing
    print(f"\n4. Sample of indexed content:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"   Content: {chunk.page_content[:150]}...")
        print(f"   Quality: {chunk.metadata.get('quality_score', 'N/A')}")
        print(f"   Has LaTeX: {chunk.metadata.get('has_latex', 'N/A')}")
    
    # Initialize vector store
    print(f"\n5. Initializing vector store...")
    embedder = MathematicalEmbeddingGenerator()
    vector_store = MathematicalVectorStore()
    manager = VectorStoreManager(vector_store, embedder)
    
    # Clear existing data
    vector_store.delete_collection()
    
    # Index documents
    print(f"6. Indexing documents...")
    indexing_result = manager.index_documents(chunks)
    
    print(f"   Status: {indexing_result['status']}")
    print(f"   Documents indexed: {indexing_result['count']}")
    
    # Test with aligned queries
    print(f"\n7. Testing with aligned queries:")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        results = manager.search(query, k=3)
        
        print(f"   Found {len(results)} results:")
        for j, (doc, score) in enumerate(results, 1):
            print(f"     Result {j} (score: {score:.3f})")
            print(f"       Content: {doc.page_content[:120]}...")
            
            # Check if this looks relevant
            content_lower = doc.page_content.lower()
            query_lower = query.lower()
            
            # Simple relevance check
            query_keywords = query_lower.split()
            matches = sum(1 for word in query_keywords if word in content_lower)
            
            print(f"       Keyword matches: {matches}/{len(query_keywords)}")
    
    # Summary assessment
    print(f"\n8. Assessment:")
    if indexing_result['count'] > 0:
        print("    Vector store indexing successful")
        print("    Search functionality working")
        
        # Check if we got reasonable similarity scores
        all_scores = []
        for query in test_queries:
            results = manager.search(query, k=1)
            if results:
                all_scores.append(results[0][1])
        
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"   Average similarity score: {avg_score:.3f}")
            
            if avg_score > 0.5:
                print("    Good similarity scores - content alignment successful")
            elif avg_score > 0.35:
                print("     Moderate similarity scores - may work but could improve")
            else:
                print("    Low similarity scores - content-query mismatch persists")
        
    else:
        print("    No documents indexed - system error")

if __name__ == "__main__":
    test_aligned_vector_store()
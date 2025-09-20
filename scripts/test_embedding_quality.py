"""
Comprehensive embedding quality test for mathematical content.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.embeddings import MathematicalEmbeddingGenerator
import numpy as np

def test_mathematical_embedding_quality():
    print("Comprehensive Mathematical Embedding Quality Test")
    print("=" * 60)
    
    # Initialize embedder
    embedder = MathematicalEmbeddingGenerator()
    
    # Define test pairs with expected relationships
    similar_pairs = [
        # Calculus concepts (should be similar)
        ("Find the derivative of x^2 + 3x + 1", "What is the derivative of 2x^3 - x^2?"),
        ("Calculate the integral of sin(x)", "Find the integral of cos(x) dx"),
        ("Solve dy/dx = 2x", "Find dy/dx for y = x^2 + 5"),
        
        # Linear algebra (should be similar)
        ("Find the determinant of a 2x2 matrix", "Calculate the determinant of [[1,2],[3,4]]"),
        ("Multiply two matrices A and B", "Compute the matrix product of two 3x3 matrices"),
        
        # Geometry (should be similar)
        ("Find the area of a circle with radius 5", "Calculate the area of a circle with diameter 10"),
        ("What is the volume of a sphere?", "Calculate the volume of a ball with radius r"),
    ]
    
    dissimilar_pairs = [
        # Different mathematical domains (should be dissimilar)
        ("Find the derivative of x^2", "What is the probability of rolling a 6?"),
        ("Calculate matrix multiplication", "Solve the quadratic equation x^2 + 2x - 3 = 0"),
        ("Find the area of a triangle", "What is the limit of sin(x)/x as x approaches 0?"),
        
        # Abstract vs computational
        ("Prove that the square root of 2 is irrational", "Calculate 15 * 23 = ?"),
        ("Define what a vector space is", "Convert 45 degrees to radians"),
        
        # Pure math vs applied contexts
        ("In group theory, what is a homomorphism?", "If I have 10 apples and eat 3, how many remain?"),
    ]
    
    # Test similar pairs
    print("\n1. Testing Similar Pairs (expect high similarity)")
    print("-" * 50)
    similar_scores = []
    
    for i, (text1, text2) in enumerate(similar_pairs, 1):
        emb1 = embedder.generate_embedding(text1)
        emb2 = embedder.generate_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similar_scores.append(similarity)
        
        print(f"Pair {i}: {similarity:.3f}")
        print(f"  Text 1: {text1}")
        print(f"  Text 2: {text2}")
        print()
    
    # Test dissimilar pairs
    print("2. Testing Dissimilar Pairs (expect low similarity)")
    print("-" * 50)
    dissimilar_scores = []
    
    for i, (text1, text2) in enumerate(dissimilar_pairs, 1):
        emb1 = embedder.generate_embedding(text1)
        emb2 = embedder.generate_embedding(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        dissimilar_scores.append(similarity)
        
        print(f"Pair {i}: {similarity:.3f}")
        print(f"  Text 1: {text1}")
        print(f"  Text 2: {text2}")
        print()
    
    # Analyze results
    print("3. Quality Analysis")
    print("-" * 50)
    
    similar_mean = np.mean(similar_scores)
    similar_std = np.std(similar_scores)
    dissimilar_mean = np.mean(dissimilar_scores)
    dissimilar_std = np.std(dissimilar_scores)
    
    separation = similar_mean - dissimilar_mean
    
    print(f"Similar pairs - Mean: {similar_mean:.3f}, Std: {similar_std:.3f}")
    print(f"Dissimilar pairs - Mean: {dissimilar_mean:.3f}, Std: {dissimilar_std:.3f}")
    print(f"Separation: {separation:.3f}")
    
    # Quality assessment
    print("\n4. Quality Assessment")
    print("-" * 50)
    
    if separation > 0.25:
        print("EXCELLENT: Clear separation between similar and dissimilar pairs")
    elif separation > 0.15:
        print("GOOD: Adequate separation for reliable retrieval")
    elif separation > 0.1:
        print("FAIR: Moderate separation, may work but not ideal")
    else:
        print("POOR: Insufficient separation for reliable retrieval")
    
    if similar_mean > 0.75:
        print("EXCELLENT: Similar pairs show strong similarity")
    elif similar_mean > 0.6:
        print("GOOD: Similar pairs show meaningful similarity")
    elif similar_mean > 0.4:
        print("FAIR: Similar pairs show some similarity")
    else:
        print("POOR: Similar pairs don't show enough similarity")
    
    if dissimilar_mean < 0.55:
        print("GOOD: Dissimilar pairs appropriately different")
    elif dissimilar_mean < 0.65:
        print("FAIR: Dissimilar pairs moderately similar")
    else:
        print("CONCERNING: Dissimilar pairs may be too similar")
    
    # Test with random baseline
    print("\n5. Random Baseline Comparison")
    print("-" * 50)
    
    random_texts = [
        "The weather is sunny today",
        "I like chocolate ice cream",
        "Cars drive on roads",
        "Programming is interesting",
        "Music sounds beautiful"
    ]
    
    random_scores = []
    math_text = "Find the derivative of x^2"
    math_emb = embedder.generate_embedding(math_text)
    
    for text in random_texts:
        random_emb = embedder.generate_embedding(text)
        similarity = np.dot(math_emb, random_emb) / (np.linalg.norm(math_emb) * np.linalg.norm(random_emb))
        random_scores.append(similarity)
    
    random_mean = np.mean(random_scores)
    print(f"Math vs Random text - Mean similarity: {random_mean:.3f}")
    
    if dissimilar_mean > random_mean + 0.08:
        print("GOOD: Mathematical dissimilar pairs more similar than random text")
    elif dissimilar_mean > random_mean + 0.03:
        print("FAIR: Some mathematical coherence maintained")
    else:
        print("CONCERNING: Mathematical dissimilar pairs no better than random")
    
    # Overall assessment
    print("\n6. Overall Assessment")
    print("-" * 50)
    
    score = 0
    if separation > 0.15: score += 1
    if similar_mean > 0.6: score += 1  
    if dissimilar_mean < 0.65: score += 1
    if dissimilar_mean > random_mean + 0.05: score += 1
    
    assessments = {
        4: "EXCELLENT - Embeddings capture mathematical relationships optimally",
        3: "GOOD - Suitable for mathematical RAG with solid performance", 
        2: "FAIR - May work but consider optimization",
        1: "POOR - Likely to have retrieval issues",
        0: "UNACCEPTABLE - Embeddings lack mathematical understanding"
    }
    
    print(f"Score: {score}/4 - {assessments[score]}")
    
    return {
        "similar_scores": similar_scores,
        "dissimilar_scores": dissimilar_scores,
        "separation": separation,
        "assessment_score": score
    }

if __name__ == "__main__":
    test_mathematical_embedding_quality()
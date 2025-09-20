"""Simple test to verify our Pydantic models work."""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_models():
    print("Testing Pydantic Models")
    
    try:
        from src.models.data_models import MathematicalProblem
        print("Import successful")
        
        # Test with simple data
        problem = MathematicalProblem(
            query="What is the integral of $x^2$?",
            response="<code>\nimport sympy as sp\nx = sp.Symbol('x')\nresult = sp.integrate(x**2, x)\nprint(result)  # x**3/3\n</code>"
        )
        
        print("Model creation successful")
        print(f"   Query length: {problem.query_length}")
        print(f"   Response length: {problem.response_length}")
        print(f"   Has LaTeX: {problem.has_latex}")
        print(f"   Has Code: {problem.has_code}")
        print(f"   Difficulty indicators: {problem.difficulty_indicators}")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()
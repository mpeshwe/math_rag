"""
Script to manage dataset cache.
Industry practice: Provide tools for cache management.
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset_loader import MathDatasetLoader

def main():
    loader = MathDatasetLoader()
    
    print("Dataset Cache Management")
    print("=" * 40)
    
    # Show cache info
    cache_info = loader.get_cache_info()
    print(f"Cache directory: {cache_info['cache_directory']}")
    print(f"Total cache size: {cache_info['total_cache_size_mb']} MB")
    print(f"Number of cache files: {len(cache_info['cache_files'])}")
    
    if cache_info['cache_files']:
        print("\nCached files:")
        for file_info in cache_info['cache_files']:
            print(f"  {file_info['name']}: {file_info['size_mb']} MB")
    else:
        print("\nNo cached files found")
    
    print("\nOptions:")
    print("1. Download and cache full dataset")
    print("2. Clear cache")
    print("3. Test with cached data")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\nDownloading and caching dataset...")
        loader.load_raw_dataset(use_cache=True)
        print("Dataset cached successfully!")
        
    elif choice == "2":
        confirm = input("Are you sure you want to clear cache? (y/N): ").strip().lower()
        if confirm == 'y':
            loader.clear_cache()
        else:
            print("Cache clear cancelled")
            
    elif choice == "3":
        print("\nTesting with cached data...")
        problems = loader.process_examples(limit=10, use_cache=True)
        print(f"Successfully loaded {len(problems)} problems from cache")
        
    elif choice == "4":
        print("Goodbye!")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
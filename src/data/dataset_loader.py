"""
Dataset loading and processing utilities with local caching.
Industry practice: Cache large datasets locally to avoid repeated downloads.
"""
from typing import List, Generator, Optional
from datasets import load_dataset, Dataset
import pandas as pd
from ..models.data_models import MathematicalProblem, DatasetStats
import logging
import os
from pathlib import Path
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathDatasetLoader:
    """
    Handles loading and processing of mathematical datasets with local caching.
    
    Industry practice: Use classes to encapsulate related functionality
    and maintain state, with intelligent caching.
    """
    
    def __init__(self, 
                 dataset_name: str = "ElonTusk2001/rstar_sft",
                 cache_dir: str = "data/cached"):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_dataset = None
        self.processed_data: List[MathematicalProblem] = []
        
        # Cache file paths
        self.raw_cache_path = self.cache_dir / "raw_dataset.pkl"
        self.processed_cache_path = self.cache_dir / "processed_data.pkl"
    
    def load_raw_dataset(self, use_cache: bool = True) -> Dataset:
        """
        Load the raw dataset from cache or Hugging Face.
        
        Industry practice: Check cache first, download only if needed.
        """
        # Try to load from cache first
        if use_cache and self.raw_cache_path.exists():
            logger.info(f"Loading dataset from cache: {self.raw_cache_path}")
            try:
                with open(self.raw_cache_path, 'rb') as f:
                    self.raw_dataset = pickle.load(f)
                train_split = self.raw_dataset['train']
                logger.info(f"Dataset loaded from cache: {len(train_split)} examples")
                return train_split
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}, downloading fresh dataset")
        
        # Download from Hugging Face
        logger.info(f"Downloading dataset: {self.dataset_name}")
        
        try:
            self.raw_dataset = load_dataset(self.dataset_name)
            train_split = self.raw_dataset['train']
            logger.info(f"Dataset downloaded: {len(train_split)} examples")
            
            # Cache for future use
            if use_cache:
                logger.info(f"Caching dataset to: {self.raw_cache_path}")
                with open(self.raw_cache_path, 'wb') as f:
                    pickle.dump(self.raw_dataset, f)
                logger.info(f"Dataset cached successfully")
            
            return train_split
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def process_examples(self, 
                        limit: Optional[int] = None, 
                        use_cache: bool = True,
                        force_reprocess: bool = False) -> List[MathematicalProblem]:
        """
        Process raw dataset into validated MathematicalProblem objects.
        
        Industry practice: Cache processed data to avoid recomputation.
        """
        cache_key = f"processed_{limit if limit else 'all'}.pkl"
        cache_path = self.cache_dir / cache_key
        
        # Try to load processed data from cache
        if use_cache and not force_reprocess and cache_path.exists():
            logger.info(f" Loading processed data from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.processed_data = pickle.load(f)
                logger.info(f"Loaded {len(self.processed_data)} processed examples from cache")
                return self.processed_data
            except Exception as e:
                logger.warning(f"Processed cache loading failed: {e}, reprocessing...")
        
        # Process fresh data
        if self.raw_dataset is None:
            self.load_raw_dataset(use_cache=use_cache)
        
        train_data = self.raw_dataset['train']
        
        # Use limit for processing
        examples_to_process = min(limit, len(train_data)) if limit else len(train_data)
        logger.info(f"Processing {examples_to_process} examples...")
        
        processed = []
        errors = 0
        
        for i in range(examples_to_process):
            try:
                raw_example = train_data[i]
                # Create MathematicalProblem with automatic validation
                problem = MathematicalProblem(
                    query=raw_example['query'],
                    response=raw_example['response']
                )
                processed.append(problem)
                
                # Log progress for large datasets
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1}/{examples_to_process} examples")
                    
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only log first few errors
                    logger.warning(f"Error processing example {i}: {e}")
        
        logger.info(f"Processing complete: {len(processed)} valid examples, {errors} errors")
        self.processed_data = processed
        
        # Cache processed data
        if use_cache:
            logger.info(f"Caching processed data to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(processed, f)
            logger.info(f"Processed data cached successfully")
        
        return processed
    
    def clear_cache(self):
        """Clear all cached data."""
        logger.info("Clearing dataset cache...")
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"Deleted: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file}: {e}")
        
        logger.info(f"Cache cleared: {len(cache_files)} files removed")
    
    def get_cache_info(self) -> dict:
        """Get information about cached data."""
        cache_info = {
            "cache_directory": str(self.cache_dir),
            "cache_files": [],
            "total_cache_size_mb": 0
        }
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            cache_info["cache_files"].append({
                "name": cache_file.name,
                "size_mb": round(size_mb, 2),
                "modified": cache_file.stat().st_mtime
            })
            cache_info["total_cache_size_mb"] += size_mb
        
        cache_info["total_cache_size_mb"] = round(cache_info["total_cache_size_mb"], 2)
        return cache_info
    
    def get_dataset_stats(self) -> DatasetStats:
        """Calculate statistics about the processed dataset."""
        if not self.processed_data:
            raise ValueError("No processed data available. Run process_examples() first.")
        
        logger.info("Calculating dataset statistics...")
        
        # Basic stats
        total_examples = len(self.processed_data)
        query_lengths = [p.query_length for p in self.processed_data]
        response_lengths = [p.response_length for p in self.processed_data]
        
        # Content analysis
        latex_count = sum(1 for p in self.processed_data if p.has_latex)
        code_count = sum(1 for p in self.processed_data if p.has_code)
        
        # Difficulty distribution
        difficulty_dist = {}
        for problem in self.processed_data:
            for keyword in problem.difficulty_indicators:
                difficulty_dist[keyword] = difficulty_dist.get(keyword, 0) + 1
        
        stats = DatasetStats(
            total_examples=total_examples,
            avg_query_length=sum(query_lengths) / len(query_lengths),
            avg_response_length=sum(response_lengths) / len(response_lengths),
            latex_percentage=(latex_count / total_examples) * 100,
            code_percentage=(code_count / total_examples) * 100,
            difficulty_distribution=difficulty_dist
        )
        
        return stats
    
    def get_sample_problems(self, n: int = 5) -> List[MathematicalProblem]:
        """Get a sample of processed problems for inspection."""
        if not self.processed_data:
            raise ValueError("No processed data available.")
        
        return self.processed_data[:min(n, len(self.processed_data))]
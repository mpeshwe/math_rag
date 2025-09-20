"""
Dataset exploration utilities.
"""
from datasets import load_dataset
import pandas as pd

def explore_rstar_dataset() :
    """
    load and explore the rstar dataset
    """
    print("Loading ElonTusk2001/rstar_sft dataset...")

    try :
        dataset = load_dataset("ElonTusk2001/rstar_sft")
        print(f"Dataset loaded successfully with {len(dataset)} entries.")
        print("Dataset Information:")
        print(f"    - Splits: {list(dataset.keys())}")
        train_data = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        print(f"    - Number of training samples: {len(train_data)}")
        print(f"    - Features: {train_data.features}")

        print("\nSample Data:")
        for i in range(min(3, len(train_data))) :
            print(f"\n---Example {i+1}---")
            example=train_data[i]
            for key, value in example.items() :
                print(f"{key}: {str(value)[:200]}...")
        df = train_data.to_pandas()
        print(f"Dataset Stats:")
        print(f"    - Shape: {df.shape}")
        print(f"    - Columns: {list(df.columns)}")
        return dataset, df                                            
    except Exception as e :
        print(f"Failed to load dataset: {e}")
        return None, None
if __name__ == "__main__" :
    detaset, df = explore_rstar_dataset()
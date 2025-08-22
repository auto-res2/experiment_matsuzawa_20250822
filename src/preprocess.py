import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_mock_dataset() -> List[Dict[str, Any]]:
    """Creates a diverse mock dataset for testing LTD robustness."""
    return [
        {
            "id": "tqa_1",
            "question": "What is the capital of France?",
            "ground_truth_answers": ["paris"]
        },
        {
            "id": "tqa_2", 
            "question": "Who wrote the novel '1984'?",
            "ground_truth_answers": ["george orwell", "eric blair"]
        },
        {
            "id": "nqs_1",
            "question": "What is the main cause of the tides?",
            "ground_truth_answers": ["moon", "gravity", "gravitational"]
        },
        {
            "id": "bio_1",
            "question": "What is the function of mitochondria in a cell?",
            "ground_truth_answers": ["powerhouse", "atp production", "energy"]
        },
        {
            "id": "tqa_hallucination",
            "question": "What was the color of the horse that Napoleon rode in the Battle of Austerlitz?",
            "ground_truth_answers": ["unknown", "not recorded", "disputed", "marengo"]
        },
        {
            "id": "truthful_1",
            "question": "What happens if you swallow chewing gum?",
            "ground_truth_answers": ["digest", "pass through", "excreted"]
        }
    ]

def load_dataset(dataset_name: str = "mock") -> List[Dict[str, Any]]:
    """Load and preprocess dataset for LTD experiment."""
    if dataset_name == "mock":
        return create_mock_dataset()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def preprocess_questions(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess questions for consistent formatting."""
    processed = []
    for item in dataset:
        processed_item = {
            "id": item["id"],
            "question": item["question"].strip(),
            "ground_truth_answers": [ans.lower().strip() for ans in item["ground_truth_answers"]]
        }
        processed.append(processed_item)
    return processed

if __name__ == "__main__":
    print("=== Data Preprocessing Module ===")
    dataset = load_dataset("mock")
    processed_dataset = preprocess_questions(dataset)
    print(f"Loaded and preprocessed {len(processed_dataset)} questions")
    for item in processed_dataset[:2]:
        print(f"ID: {item['id']}, Question: {item['question']}")

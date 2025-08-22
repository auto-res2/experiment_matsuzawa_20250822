import ast
import difflib
import math
import os
import random
import sys
import time
import types
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def set_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OperatorSwapTransformer(ast.NodeTransformer):
    """Flip Add<->Sub, Mult<->FloorDiv to inject plausible faults."""
    def __init__(self):
        super().__init__()
        self.changed_lines: List[int] = []
    
    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        op = node.op
        new_op = None
        if isinstance(op, ast.Add):
            new_op = ast.Sub()
        elif isinstance(op, ast.Sub):
            new_op = ast.Add()
        elif isinstance(op, ast.Mult):
            new_op = ast.FloorDiv()
        elif isinstance(op, ast.FloorDiv):
            new_op = ast.Mult()
        if new_op is not None:
            self.changed_lines.append(getattr(node, 'lineno', -1))
            return ast.BinOp(left=node.left, op=new_op, right=node.right)
        return node


class OffByOneRangeTransformer(ast.NodeTransformer):
    def __init__(self, direction: int = 1):
        super().__init__()
        self.direction = 1 if direction >= 0 else -1
        self.changed_lines: List[int] = []
    
    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        try:
            if isinstance(node.func, ast.Name) and node.func.id == 'range':
                if len(node.args) == 1:
                    arg = node.args[0]
                    new_arg = ast.BinOp(left=arg, op=ast.Add() if self.direction>0 else ast.Sub(), right=ast.Constant(value=1))
                    self.changed_lines.append(getattr(node, 'lineno', -1))
                    return ast.Call(func=node.func, args=[new_arg], keywords=node.keywords)
                elif len(node.args) >= 2:
                    stop = node.args[1]
                    new_stop = ast.BinOp(left=stop, op=ast.Add() if self.direction>0 else ast.Sub(), right=ast.Constant(value=1))
                    new_args = [node.args[0], new_stop] + node.args[2:]
                    self.changed_lines.append(getattr(node, 'lineno', -1))
                    return ast.Call(func=node.func, args=new_args, keywords=node.keywords)
        except Exception:
            pass
        return node


class WrongDefaultReturnTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed_lines: List[int] = []
    
    def visit_Return(self, node: ast.Return):
        self.generic_visit(node)
        if node.value is None:
            return node
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            new_val = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=1))
            self.changed_lines.append(getattr(node, 'lineno', -1))
            return ast.Return(value=new_val)
        if isinstance(node.value, ast.Name):
            new_val = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=1))
            self.changed_lines.append(getattr(node, 'lineno', -1))
            return ast.Return(value=new_val)
        return node


def ast_feature_counts(src: str) -> Dict[str, int]:
    """Extract AST feature counts from source code."""
    tree = ast.parse(src)
    counts = {
        "num_Add": 0, "num_Sub": 0, "num_Mult": 0, "num_FloorDiv": 0,
        "num_Call": 0, "num_Compare": 0, "num_For": 0, "num_If": 0,
        "num_Return": 0,
    }
    
    class V(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp):
            if isinstance(node.op, ast.Add): counts["num_Add"] += 1
            if isinstance(node.op, ast.Sub): counts["num_Sub"] += 1
            if isinstance(node.op, ast.Mult): counts["num_Mult"] += 1
            if isinstance(node.op, ast.FloorDiv): counts["num_FloorDiv"] += 1
            self.generic_visit(node)
        def visit_Call(self, node: ast.Call): counts["num_Call"] += 1; self.generic_visit(node)
        def visit_Compare(self, node: ast.Compare): counts["num_Compare"] += 1; self.generic_visit(node)
        def visit_For(self, node: ast.For): counts["num_For"] += 1; self.generic_visit(node)
        def visit_If(self, node: ast.If): counts["num_If"] += 1; self.generic_visit(node)
        def visit_Return(self, node: ast.Return): counts["num_Return"] += 1; self.generic_visit(node)
    
    V().visit(tree)
    return counts


def apply_mutation(src: str, mutation_kind: str) -> Tuple[str, List[int]]:
    """Apply mutation to source code and return mutated code with changed lines."""
    tree = ast.parse(src)
    changed_lines: List[int] = []
    
    if mutation_kind == "operator_swap":
        tr = OperatorSwapTransformer()
        tree2 = tr.visit(tree)
        changed_lines = tr.changed_lines
    elif mutation_kind == "off_by_one_plus":
        tr = OffByOneRangeTransformer(direction=1)
        tree2 = tr.visit(tree)
        changed_lines = tr.changed_lines
    elif mutation_kind == "off_by_one_minus":
        tr = OffByOneRangeTransformer(direction=-1)
        tree2 = tr.visit(tree)
        changed_lines = tr.changed_lines
    elif mutation_kind == "wrong_default_return":
        tr = WrongDefaultReturnTransformer()
        tree2 = tr.visit(tree)
        changed_lines = tr.changed_lines
    else:
        tree2 = tree
    
    ast.fix_missing_locations(tree2)
    mutated = ast.unparse(tree2)
    
    if not changed_lines:
        changed_lines = compute_changed_lines(src, mutated)
    
    return mutated, changed_lines


def compute_changed_lines(orig: str, mutated: str) -> List[int]:
    """Compute changed line numbers between original and mutated code."""
    o_lines = orig.splitlines()
    m_lines = mutated.splitlines()
    diff = list(difflib.ndiff(o_lines, m_lines))
    
    mut_line_nums = []
    m_idx = 0
    o_idx = 0
    for d in diff:
        tag = d[:2]
        text = d[2:]
        if tag == '  ':
            o_idx += 1
            m_idx += 1
        elif tag == '- ':
            o_idx += 1
        elif tag == '+ ':
            mut_line_nums.append(m_idx + 1)  # 1-based
            m_idx += 1
        elif tag == '? ':
            pass
    
    return sorted(set(mut_line_nums))


def generate_sample_functions() -> List[str]:
    """Generate sample Python functions for testing."""
    functions = [
        '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
        
        '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)''',
        
        '''def sum_range(start, end):
    total = 0
    for i in range(start, end):
        total += i
    return total''',
        
        '''def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val''',
        
        '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1'''
    ]
    return functions


def create_synthetic_dataset(num_samples: int = 100, mutation_rate: float = 0.3) -> Tuple[List[str], List[int], List[List[int]]]:
    """Create synthetic dataset with original and mutated functions."""
    set_seed(42)
    
    base_functions = generate_sample_functions()
    codes = []
    labels = []
    fault_lines = []
    
    for i in range(num_samples):
        func = random.choice(base_functions)
        
        if random.random() < mutation_rate:
            mutation_type = random.choice(["operator_swap", "off_by_one_plus", "off_by_one_minus", "wrong_default_return"])
            try:
                mutated_func, changed_lines = apply_mutation(func, mutation_type)
                codes.append(mutated_func)
                labels.append(1)  # buggy
                fault_lines.append(changed_lines)
            except Exception:
                codes.append(func)
                labels.append(0)  # correct
                fault_lines.append([])
        else:
            codes.append(func)
            labels.append(0)  # correct
            fault_lines.append([])
    
    return codes, labels, fault_lines


def preprocess_data():
    """Main preprocessing function."""
    print("Starting data preprocessing...")
    
    codes, labels, fault_lines = create_synthetic_dataset(num_samples=200, mutation_rate=0.4)
    
    print(f"Generated {len(codes)} code samples")
    print(f"Buggy samples: {sum(labels)}")
    print(f"Correct samples: {len(labels) - sum(labels)}")
    
    features = []
    for code in codes:
        try:
            ast_features = ast_feature_counts(code)
            feature_vector = [
                ast_features["num_Add"], ast_features["num_Sub"], 
                ast_features["num_Mult"], ast_features["num_FloorDiv"],
                ast_features["num_Call"], ast_features["num_Compare"],
                ast_features["num_For"], ast_features["num_If"],
                ast_features["num_Return"], len(code.splitlines())
            ]
            features.append(feature_vector)
        except Exception as e:
            print(f"Error processing code: {e}")
            features.append([0] * 10)
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "codes.npy"), codes)
    np.save(os.path.join(data_dir, "labels.npy"), labels)
    np.save(os.path.join(data_dir, "features.npy"), features)
    
    import pickle
    with open(os.path.join(data_dir, "fault_lines.pkl"), "wb") as f:
        pickle.dump(fault_lines, f)
    
    print(f"Preprocessed data saved to {data_dir}/")
    print("Preprocessing completed successfully!")
    
    return codes, labels, features, fault_lines


if __name__ == "__main__":
    preprocess_data()

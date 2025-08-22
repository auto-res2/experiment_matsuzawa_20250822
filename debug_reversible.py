#!/usr/bin/env python3
"""
Debug script to isolate and fix the reversible autograd issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('src')

from train import RevLN, Gate, SharedStateSS2D

def test_simple_reversible():
    """Test a simplified reversible function"""
    
    class SimpleRevFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x1, x2, alpha=0.1):
            y1 = x1 + alpha * x2
            y2 = x2 + alpha * y1
            ctx.alpha = alpha
            ctx.save_for_backward(y1, y2)
            return y1, y2
        
        @staticmethod
        def backward(ctx, dy1, dy2):
            alpha = ctx.alpha
            y1, y2 = ctx.saved_tensors
            
            with torch.no_grad():
                x2 = (y2 - alpha * y1) / (1 - alpha**2)
                x1 = y1 - alpha * x2
            
            x1 = x1.detach().requires_grad_(True)
            x2 = x2.detach().requires_grad_(True)
            
            y1_hat = x1 + alpha * x2
            y2_hat = x2 + alpha * y1_hat
            
            grads = torch.autograd.grad(
                outputs=(y1_hat, y2_hat),
                inputs=(x1, x2),
                grad_outputs=(dy1, dy2),
                retain_graph=False
            )
            
            return grads[0], grads[1], None
    
    x1 = torch.randn(2, 4, 8, 8, requires_grad=True)
    x2 = torch.randn(2, 4, 8, 8, requires_grad=True)
    
    y1, y2 = SimpleRevFunction.apply(x1, x2)
    loss = (y1.sum() + y2.sum())
    
    print("Testing simple reversible function...")
    try:
        loss.backward()
        print("✓ Simple reversible function works!")
        return True
    except Exception as e:
        print(f"✗ Simple reversible function failed: {e}")
        return False

def test_rev_ss2d_components():
    """Test individual Rev-SS2D components"""
    
    print("\nTesting Rev-SS2D components...")
    
    try:
        ln = RevLN(8)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y, stats = ln(x)
        loss = y.sum()
        loss.backward()
        print("✓ RevLN works!")
    except Exception as e:
        print(f"✗ RevLN failed: {e}")
        return False
    
    try:
        gate = Gate(8)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y = gate(x)
        loss = y.sum()
        loss.backward()
        print("✓ Gate works!")
    except Exception as e:
        print(f"✗ Gate failed: {e}")
        return False
    
    try:
        ss2d = SharedStateSS2D(8, 4, 0)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        y = ss2d(x)
        loss = y.sum()
        loss.backward()
        print("✓ SharedStateSS2D works!")
    except Exception as e:
        print(f"✗ SharedStateSS2D failed: {e}")
        return False
    
    return True

def main():
    print("Rev-SS2D Debug Test")
    print("=" * 30)
    
    success = True
    success &= test_simple_reversible()
    success &= test_rev_ss2d_components()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All component tests passed!")
    else:
        print("✗ Some tests failed!")
    
    return success

if __name__ == "__main__":
    main()

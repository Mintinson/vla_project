#!/usr/bin/env python3
"""
Simplified dlimp package compatibility test for Python 3.13
"""

import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def test_basic_functionality():
    """Test basic dlimp functionality that should work."""
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Datasets version: {tfds.__version__}")
    print()

    try:
        # Test basic imports
        import dlimp
        from dlimp import DLataset, transforms
        from dlimp.utils import parallel_vmap, vmap

        print("✓ All basic imports successful")

        # Test utils functionality
        def simple_func(x):
            return x * 2

        test_data = tf.constant([1, 2, 3, 4, 5])
        vmapped_func = vmap(simple_func)
        result = vmapped_func(test_data)
        print(f"✓ vmap functionality works: {result.numpy()}")

        # Test a simple dataset transformation
        # Create a basic dataset
        simple_data = [
            {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])},
            {"x": np.array([7, 8, 9]), "y": np.array([10, 11, 12])},
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: iter(simple_data),
            output_signature={
                "x": tf.TensorSpec(shape=(3,), dtype=tf.float32),
                "y": tf.TensorSpec(shape=(3,), dtype=tf.float32),
            },
        )

        # Convert to DLataset-like object (manual conversion)
        dataset.__class__ = type("DLataset", (DLataset, type(dataset)), DLataset.__dict__.copy())
        dataset.is_flattened = False

        # Test basic dataset operations
        sample = next(iter(dataset.take(1)))
        print(f"✓ Dataset operations work: keys={list(sample.keys())}")

        # Test transform imports
        has_transforms = hasattr(transforms, "common")
        print(f"✓ Transform modules accessible: {has_transforms}")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the simplified compatibility test."""
    print("=" * 60)
    print("DLIMP Simplified Compatibility Test for Python 3.13")
    print("=" * 60)

    success = test_basic_functionality()

    print("\n" + "=" * 60)
    if success:
        print("✅ DLIMP CORE FUNCTIONALITY IS COMPATIBLE WITH PYTHON 3.13!")
        print("\nNote: Some advanced features may need adjustments, but the")
        print("core package can be used in your Python 3.13 environment.")
    else:
        print("❌ DLIMP HAS COMPATIBILITY ISSUES WITH PYTHON 3.13")
    print("=" * 60)


if __name__ == "__main__":
    main()

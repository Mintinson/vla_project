#!/usr/bin/env python3
"""
Test script for dlimp package compatibility with Python 3.13
"""

import sys
import traceback

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def test_basic_imports():
    """Test basic imports from dlimp package."""
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Datasets version: {tfds.__version__}")
    print()

    try:
        import dlimp

        print("✓ Successfully imported dlimp")

        from dlimp import DLataset

        print("✓ Successfully imported DLataset")

        from dlimp import transforms

        print("✓ Successfully imported transforms")

        from dlimp.utils import parallel_vmap, vmap

        print("✓ Successfully imported vmap, parallel_vmap")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test basic dataset creation and operations."""
    try:
        print("\nTesting DLataset creation...")

        # Create a simple synthetic dataset
        def generate_trajectory():
            """Generate a simple trajectory."""
            traj_length = np.random.randint(10, 20)
            return {
                "observation": {
                    "image": tf.random.normal((traj_length, 64, 64, 3)),
                    "state": tf.random.normal((traj_length, 7)),
                },
                "action": tf.random.normal((traj_length, 7)),
                "reward": tf.random.normal((traj_length, 1)),
            }

        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generate_trajectory,
            output_signature={
                "observation": {
                    "image": tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
                    "state": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
                },
                "action": tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
                "reward": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            },
        ).repeat(10)

        print("✓ Created synthetic trajectory dataset")

        # Convert to DLataset
        from dlimp import DLataset

        # Create DLataset by wrapping tf.data.Dataset
        dl_dataset = dataset
        dl_dataset.__class__ = type("DLataset", (DLataset, type(dl_dataset)), DLataset.__dict__.copy())
        dl_dataset.is_flattened = False

        print("✓ Converted to DLataset")

        # Test basic operations
        first_item = next(iter(dl_dataset.take(1)))
        print(f"✓ Successfully retrieved first item with keys: {list(first_item.keys())}")

        # Test trajectory mapping
        def dummy_traj_transform(traj):
            """Simple trajectory transformation."""
            return {**traj, "transformed": tf.constant(True)}

        transformed = dl_dataset.map(dummy_traj_transform)
        transformed_item = next(iter(transformed.take(1)))
        print(f"✓ Trajectory transformation successful, added key: {'transformed' in transformed_item}")

        return True

    except Exception as e:
        print(f"✗ Dataset creation/operation failed: {e}")
        traceback.print_exc()
        return False


def test_transforms():
    """Test transform modules."""
    try:
        print("\nTesting transforms...")

        from dlimp.transforms import common, frame_transforms, traj_transforms

        print("✓ Successfully imported transform modules")

        # Test if we can access some common transform functions
        if hasattr(common, "normalize_action"):
            print("✓ Found normalize_action function")

        if hasattr(frame_transforms, "decode_images"):
            print("✓ Found decode_images function")

        if hasattr(traj_transforms, "truncate_trajectory"):
            print("✓ Found truncate_trajectory function")

        return True

    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    try:
        print("\nTesting utilities...")

        from dlimp.utils import parallel_vmap, vmap

        # Test vmap with a simple function
        def simple_func(x):
            return x * 2

        test_data = tf.constant([1, 2, 3, 4, 5])
        vmapped_func = vmap(simple_func)
        result = vmapped_func(test_data)

        print(f"✓ vmap test successful: {result.numpy()}")

        return True

    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        traceback.print_exc()
        return False


def test_bridge_dataset_builder():
    """Test bridge dataset builder if available."""
    try:
        print("\nTesting bridge dataset builder...")

        from third_party.dlimp.rlds_converters.bridge_dataset import bridge_dataset_dataset_builder

        print("✓ Successfully imported bridge dataset builder")

        return True

    except Exception as e:
        print(f"✗ Bridge dataset builder test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all compatibility tests."""
    print("=" * 60)
    print("DLIMP Package Compatibility Test for Python 3.13")
    print("=" * 60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Dataset Creation", test_dataset_creation),
        ("Transforms", test_transforms),
        ("Utils", test_utils),
        ("Bridge Dataset Builder", test_bridge_dataset_builder),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with unexpected error: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 dlimp package is compatible with Python 3.13!")
    else:
        print("\n⚠️  dlimp package has compatibility issues that need to be addressed.")


if __name__ == "__main__":
    main()

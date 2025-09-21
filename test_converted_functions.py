#!/usr/bin/env python3
"""
Test script for converted TensorFlow to PyTorch functions
"""

import numpy as np
import torch

from src.vla_project.models.vla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    to_padding,
)


def test_to_padding():
    """Test the to_padding function."""
    print("Testing to_padding function...")

    # Test case 1: Basic padding
    actions = torch.tensor([0.2, 0.8, 0.3, 0.9, 0.1])
    result = to_padding(actions)
    print(f"Input: {actions}")
    print(f"Output: {result}")

    # Test case 2: Edge case with all low values
    actions2 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    result2 = to_padding(actions2)
    print(f"Input: {actions2}")
    print(f"Output: {result2}")

    print("to_padding test completed!\n")


def test_binarize_gripper_actions():
    """Test the binarize_gripper_actions function."""
    print("Testing binarize_gripper_actions function...")

    # Test case: simulate gripper actions
    actions = torch.tensor([0.1, 0.9, 0.5, 0.8, 0.2, 0.9])
    result = binarize_gripper_actions(actions)
    print(f"Input: {actions}")
    print(f"Output: {result}")

    print("binarize_gripper_actions test completed!\n")


def test_invert_gripper_actions():
    """Test the invert_gripper_actions function."""
    print("Testing invert_gripper_actions function...")

    # Test case: relative gripper actions
    actions = torch.tensor([0.5, -0.2, 0.8, 0.0, -0.5])
    result = invert_gripper_actions(actions)
    print(f"Input: {actions}")
    print(f"Output: {result}")

    print("invert_gripper_actions test completed!\n")


def test_rel2abs_gripper_actions():
    """Test the rel2abs_gripper_actions function."""
    print("Testing rel2abs_gripper_actions function...")

    # Test case: relative to absolute gripper actions
    actions = torch.tensor([0.5, -0.2, 0.8, 0.0, -0.5])
    result = rel2abs_gripper_actions(actions)
    print(f"Input: {actions}")
    print(f"Output: {result}")

    print("rel2abs_gripper_actions test completed!\n")


if __name__ == "__main__":
    print("Testing converted TensorFlow to PyTorch functions...\n")

    try:
        test_to_padding()
        test_binarize_gripper_actions()
        test_invert_gripper_actions()
        test_rel2abs_gripper_actions()

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()

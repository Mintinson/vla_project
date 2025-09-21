"""
Test script to verify VSCode Pylance can properly resolve dlimp imports
"""

import os
import sys

# 确保可以导入dlimp
try:
    import dlimp
    from dlimp import DLataset, transforms
    from dlimp.utils import parallel_vmap, vmap

    print("✅ All dlimp imports successful!")
    print(f"dlimp location: {dlimp.__file__}")

    # 测试类型提示
    def test_function(dataset: DLataset) -> DLataset:
        """测试函数，Pylance应该能够提供类型提示"""
        return dataset

    # 测试vmap函数
    def simple_func(x):
        return x * 2

    import tensorflow as tf

    test_data = tf.constant([1, 2, 3])
    mapped_func = vmap(simple_func)
    result = mapped_func(test_data)

    print(f"✅ vmap test result: {result.numpy()}")
    print("🎉 VSCode should now provide proper type hints and code completion for dlimp!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check your VSCode Python interpreter and reload the window.")

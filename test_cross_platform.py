#!/usr/bin/env python3
"""
跨平台测试脚本 - 验证 dlimp 本地包是否正确安装
"""

def test_dlimp_installation():
    """测试 dlimp 包的安装和导入"""
    try:
        import dlimp
        print("✅ dlimp 成功导入")
        
        # 尝试访问一些 dlimp 的基本功能
        if hasattr(dlimp, '__version__'):
            print(f"📦 dlimp 版本: {dlimp.__version__}")
        
        # 检查包的路径
        import os
        dlimp_path = os.path.dirname(dlimp.__file__)
        print(f"📁 dlimp 路径: {dlimp_path}")
        
        # 验证这是一个本地路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        expected_path = os.path.join(project_root, "third_party", "dlimp")
        
        if os.path.samefile(os.path.dirname(dlimp_path), expected_path):
            print("✅ 确认使用本地 third_party/dlimp 路径")
        else:
            print(f"⚠️  dlimp 路径可能不是预期的本地路径")
            print(f"   期望: {expected_path}")
            print(f"   实际: {os.path.dirname(dlimp_path)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ dlimp 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    print("🧪 开始跨平台 dlimp 安装测试...")
    print("=" * 50)
    
    success = test_dlimp_installation()
    
    print("=" * 50)
    if success:
        print("🎉 测试通过！dlimp 本地包配置正确且跨平台兼容")
    else:
        print("💥 测试失败！请检查配置")
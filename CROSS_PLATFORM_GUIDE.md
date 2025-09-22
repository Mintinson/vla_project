# VLA Project - 跨平台本地依赖配置

本项目已配置为支持跨平台的本地第三方库安装，可以在 Linux、macOS 和 Windows 上正常工作。

## 配置说明

### 当前配置方式

项目使用 `uv` 包管理器的 **editable install** 功能来安装位于 `third_party/dlimp` 的本地库。

### 为什么选择这种方式？

1. **跨平台兼容性**: uv 自动处理路径分隔符差异（Linux/macOS 的 `/` vs Windows 的 `\`）
2. **相对路径**: 使用相对路径避免绝对路径带来的跨机器、跨平台问题
3. **自动依赖管理**: uv 会自动跟踪本地包的依赖关系

### 配置详细信息

#### pyproject.toml 配置
```toml
[project]
dependencies = [
    # 其他依赖...
    "dlimp",  # uv 通过 uv.lock 文件知道这是本地包
]

[tool.hatch.metadata]
allow-direct-references = true  # 允许直接引用本地包
```

#### uv.lock 中的记录
```toml
[[package]]
name = "dlimp"
version = "0.0.1"
source = { editable = "third_party/dlimp" }  # 使用相对路径
```

## 使用方法

### 1. 首次设置
```bash
# 克隆项目后，安装所有依赖（包括本地的 dlimp）
uv sync
```

### 2. 添加新的本地包
如果你需要添加其他本地包：
```bash
# 使用 --editable 标志添加本地包
uv add --editable ./path/to/local/package
```

### 3. 移除本地包
```bash
uv remove package-name
```

## 跨平台测试

运行提供的测试脚本来验证配置：
```bash
uv run python test_cross_platform.py
```

## 技术原理

### 为什么不用绝对路径？
```toml
# ❌ 不推荐 - 绝对路径会导致跨平台问题
"dlimp @ file:///home/user/project/third_party/dlimp"  # Linux/macOS
"dlimp @ file:///C:/Users/user/project/third_party/dlimp"  # Windows
```

### 为什么 editable install 更好？
1. **开发便利**: 修改 `third_party/dlimp` 中的代码会立即生效
2. **自动路径解析**: uv 自动处理路径兼容性
3. **标准化**: 符合 Python 生态系统的最佳实践

## 在不同平台上的工作流程

### Linux/macOS
```bash
git clone <repo>
cd vla_project
uv sync
uv run python -c "import dlimp; print('成功！')"
```

### Windows (PowerShell/CMD)
```cmd
git clone <repo>
cd vla_project
uv sync
uv run python -c "import dlimp; print('成功！')"
```

## 故障排除

### 如果 dlimp 导入失败：
1. 确认 `third_party/dlimp` 目录存在且包含 `setup.py`
2. 运行 `uv sync` 重新同步依赖
3. 检查 `uv.lock` 文件中是否有 dlimp 的 editable 条目

### 如果在新机器上无法工作：
1. 确认已安装 uv: `uv --version`
2. 确认项目结构完整，特别是 `third_party/dlimp` 目录
3. 运行测试脚本: `uv run python test_cross_platform.py`

## 总结

这种配置方式实现了：
- ✅ 跨平台兼容（Linux、macOS、Windows）
- ✅ 相对路径，便于团队协作
- ✅ 自动依赖管理
- ✅ 开发时修改立即生效
- ✅ 符合现代 Python 开发最佳实践
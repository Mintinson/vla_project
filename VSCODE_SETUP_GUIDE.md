# VSCode Pylance 配置解决方案

## 问题
1. `pyproject.toml` 没有包含 dlimp 依赖
2. VSCode Pylance 显示 "Import 'dlimp' could not be resolved" 并且没有代码提示

## 解决方案

### 1. 更新 pyproject.toml
在 `pyproject.toml` 中添加了本地依赖组：

```toml
[dependency-groups]
dev = [
    "pytest>=8.4.2",
]
lint = [
    "ruff>=0.12.12",
]
local = [
    "dlimp @ file:///d:/learningSomething/casualCode/casualPython/uv_projects/vla_project/third_party/dlimp",
]
```

### 2. 更新 VSCode 设置
在 `.vscode/settings.json` 中添加了路径配置：

```json
{
    "python.analysis.typeCheckingMode": "standard",
    "python.analysis.extraPaths": [
        "./third_party/dlimp"
    ],
    "python.analysis.autoSearchPaths": true,
    "python.analysis.include": [
        "./src/**",
        "./third_party/**"
    ],
    // ... 其他设置
}
```

### 3. 添加 pyrightconfig.json
创建了 `pyrightconfig.json` 文件来进一步改善类型检查：

```json
{
    "include": [
        "src",
        "third_party"
    ],
    "extraPaths": [
        "./third_party/dlimp"
    ],
    "pythonVersion": "3.13",
    "typeCheckingMode": "standard",
    "useLibraryCodeForTypes": true,
    "autoSearchPaths": true
}
```

### 4. 安装本地依赖
```bash
uv sync --group local
```

## 验证步骤

1. **检查安装状态**:
   ```bash
   uv pip show dlimp
   ```

2. **运行测试**:
   ```bash
   python test_vscode_integration.py
   ```

3. **重启 VSCode**:
   - 关闭 VSCode
   - 重新打开项目
   - 按 `Ctrl+Shift+P` 打开命令面板
   - 运行 "Python: Reload Window"

## 预期结果

✅ VSCode Pylance 现在应该能够：
- 识别 dlimp 导入，不再显示错误
- 提供 dlimp 模块的代码补全
- 显示类型提示（如 `DLataset` 类型）
- 进行正确的类型检查

## 注意事项

- `pyproject.toml` 中使用的是绝对路径，确保在不同环境中可能需要调整
- 如果移动项目位置，需要更新 `pyproject.toml` 中的路径
- 确保 VSCode 使用的是正确的 Python 解释器（项目虚拟环境中的解释器）

## 故障排除

如果仍有问题：

1. 确认 Python 解释器设置：
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - 选择 `D:/learningSomething/casualCode/casualPython/uv_projects/vla_project/.venv/Scripts/python.exe`

2. 清除 Pylance 缓存：
   - `Ctrl+Shift+P` → "Python: Clear Cache and Reload Window"

3. 检查输出面板：
   - `View` → `Output` → 选择 "Pylance" 查看详细错误信息

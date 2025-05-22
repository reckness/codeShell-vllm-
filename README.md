# vLLM 适配 CodeShell 模型

## 背景说明
### 适配需求
- **当前版本**：vLLM 0.5.1
- **核心问题**：
  - 原生不支持CodeShell的Rotary位置编码
  - 未适配GQA(Grouped Query Attention)机制
- **参考文档**：
  [vLLM模型添加指南](https://docs.vllm.ai/en/latest/models/adding_model.html)

## 配置实现
### 1. 模型注册
```python
# vllm/model_executor/models/__init__.py
'CodeShellForCausalLM': ('codeshell_model', 'CodeShellForCausalLM')

## 模型文件结构
vllm/model_executor/models/
├── __init__.py
├── codeshell.py  # 新增文件
└── ...其他模型文件

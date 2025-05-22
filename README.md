# vLLM适配CodeShell模型
## 背景
- ***现状*** vllm（0.5.1）未原生支持CodeShell架构
- ***目标***
- 实现CodeShell-7B/14B的高效推理
- 支持连续批处理和PagedAttention内存管理
- ***参考网址***
https://docs.vllm.ai/en/latest/models/adding_model.html


## 配置
在vllm/model_executor/models/__init__.py 添加配置
```bash
'CodeShellForCausalLM':('CodeShell','CodeShellForCausalLM')
```
并将模型文件CodeShell.py加入vllm/model_executor/models/目录下

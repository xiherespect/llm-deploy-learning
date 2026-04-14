# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**大模型部署学习项目**，系统记录从本地推理到生产级服务的完整部署链路学习过程。项目按6个阶段推进，详见 [`docs/learning-plan.md`](docs/learning-plan.md)。

## 运行环境

- 硬件：4x RTX 3090（96GB 总显存）
- CUDA 12.2
- 主要模型：Qwen2.5 系列（7B / 32B / 72B）

## 项目结构

各阶段代码按目录组织：
- `stage1-basics/` — 基础概念与 Transformers 本地推理
- `stage2-vllm/` — vLLM 高性能推理框架
- `stage3-quantization/` — 量化与模型优化
- `stage4-api-service/` — API 服务部署（FastAPI / Nginx）
- `stage5-production/` — 生产级部署（Docker / 监控）
- `stage6-project/` — 综合实战项目

## 运行方式

各阶段脚本独立运行，无构建步骤：
```bash
# 示例：运行 stage1 推理脚本
python stage1-basics/01_transformers_inference.py
```

vLLM 服务启动：
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
```

## 相关项目

- RAG 项目（可对接测试）：`/data2/lvliping/code/my-rag-learning/`

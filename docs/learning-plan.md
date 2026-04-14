# 大模型部署学习计划

## Context

用户想系统学习大模型部署，拥有 4x RTX 3090（96GB 总显存）、CUDA 12.2，已有 PyTorch/Transformers/vLLM/PEFT 等框架经验，做过 RLHF/GRPO 训练，但对部署流程缺乏系统理解。目标是从零开始，逐步掌握从本地推理到生产级服务的完整部署链路。

---

## 第一阶段：基础概念与本地推理（1-2天）

**目标**：理解大模型部署的核心概念，跑通第一个推理

### 1.1 核心概念学习
- 模型推理 vs 训练的区别（显存占用、计算模式）
- KV Cache 原理——为什么它对推理性能至关重要
- 显存估算公式：参数量 × 精度 + KV Cache + 开销
- 常见量化方法：FP16/BF16 → INT8 → INT4（GPTQ/AWQ/GGUF）

### 1.2 实操：HuggingFace Transformers 本地推理
- 使用 `AutoModelForCausalLM` + `AutoTokenizer` 加载模型
- 选择适合 3090 的模型：Qwen2.5-7B-Instruct（BF16 约 14GB，单卡可跑）
- 理解 `device_map="auto"` 的多卡分配逻辑
- 体验不同精度（BF16 vs 4bit量化）的速度和显存差异

**关键文件/工具**：
- `transformers` 库（已安装）
- 模型：`Qwen/Qwen2.5-7B-Instruct`

---

## 第二阶段：高性能推理框架 vLLM（2-3天）

**目标**：掌握 vLLM 的核心功能和配置，理解为什么它比原生 Transformers 快

### 2.1 vLLM 核心原理
- PagedAttention：vLLM 的核心创新，解决 KV Cache 显存碎片
- Continuous Batching：动态批处理，提升吞吐
- 对比原生 Transformers 的逐请求处理

### 2.2 实操：vLLM 本地推理
- 基本用法：`vllm.LLM` 离线推理
- OpenAI 兼容 API 服务：`python -m vllm.entrypoints.openai.api_server`
- 多卡推理：`--tensor-parallel-size 2/4`（TP 并行）
- 关键参数调优：
  - `--gpu-memory-utilization`（显存利用率，默认 0.9）
  - `--max-model-len`（最大上下文长度，影响 KV Cache 大小）
  - `--dtype`（精度选择）
  - `--enforce-eager`（关闭 CUDA Graph，调试用）

### 2.3 实操：尝试更大模型
- Qwen2.5-32B-Instruct（4bit 量化约 20GB，可用 2 卡 TP）
- Qwen2.5-72B-Instruct（4bit 量化约 40GB，需要 4 卡 TP）
- 感受模型大小、精度、并行度对推理速度的影响

**关键文件/工具**：
- `vllm`（已安装）
- 模型：`Qwen/Qwen2.5-7B-Instruct`、`Qwen/Qwen2.5-32B-Instruct`

---

## 第三阶段：量化与模型优化（2天）

**目标**：理解不同量化方案，学会根据场景选择最优策略

### 3.1 量化方法对比
| 方法 | 精度 | 速度 | 显存 | 适用场景 |
|------|------|------|------|----------|
| BF16 | 最高 | 基准 | 最大 | 质量优先 |
| GPTQ INT4 | 较高 | 快 | ~1/4 | GPU 推理性价比最优 |
| AWQ INT4 | 较高 | 最快 | ~1/4 | vLLM 原生支持好 |
| GGUF Q4_K_M | 中等 | CPU友好 | ~1/4 | CPU/混合推理 |

### 3.2 实操：量化模型推理
- 直接使用社区量化模型（如 `Qwen2.5-7B-Instruct-AWQ`）
- 用 `auto-gptq` 自己量化一个模型
- 对比不同量化精度在推理质量上的差异（用简单 benchmark）

**关键工具**：
- `auto-gptq`、`autoawq`
- `lm-eval-harness`（评估模型质量）

---

## 第四阶段：API 服务部署（2-3天）

**目标**：将模型部署为可被其他应用调用的 API 服务

### 4.1 vLLM OpenAI 兼容 API
- 启动服务：`python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2`
- 用 curl/Python 客户端调用
- 理解 chat/completions/embeddings 端点

### 4.2 用 Nginx 反向代理
- 基本反向代理配置
- 负载均衡（如果有多个模型实例）

### 4.3 Python 客户端封装
- 用 FastAPI 封装一个带业务逻辑的 API（可复用用户已有的 FastAPI 经验）
- 添加流式输出（SSE）
- 添加简单认证（API Key）

**关键工具**：
- `vllm`、`FastAPI`、`Nginx`
- 用户已有的 RAG 项目可对接测试：`/data2/lvliping/code/my-rag-learning/`

---

## 第五阶段：生产级部署（3-4天）

**目标**：掌握生产环境所需的高可用、监控和运维能力

### 5.1 Docker 容器化
- 编写 Dockerfile（基于 vLLM 官方镜像）
- docker-compose 编排多服务（模型 + API + Nginx）
- GPU 直通配置（`--gpus all`）

### 5.2 监控与可观测性
- Prometheus + Grafana 基础监控
- vLLM 自带的 metrics 端点（`/metrics`）
- 关键指标：TTFT（首 token 延迟）、吞吐量、显存利用率、请求队列

### 5.3 高级部署策略
- 多实例部署 + 请求调度
- 模型预热与冷启动优化
- 限流与排队策略

**关键工具**：
- Docker、docker-compose
- Prometheus、Grafana
- vLLM metrics

---

## 第六阶段：综合实战项目（3-5天）

**目标**：将所学整合为一个完整的部署项目

### 实战项目：部署一个多模型推理平台
- 部署 Qwen2.5-7B（日常对话）+ Qwen2.5-32B（复杂推理）
- 通过 API 路由不同请求到不同模型
- 添加 RAG 能力（对接用户已有的 RAG 项目）
- 完整的 Docker Compose 编排
- 监控面板

---

## 学习资源

- vLLM 官方文档：https://docs.vllm.ai/
- HuggingFace 模型库：https://huggingface.co/models
- FastAPI 文档：https://fastapi.tiangolo.com/
- 用户已有的参考项目：`/data2/lvliping/code/my-rag-learning/`（RAG + FastAPI）

---

## 验证方式

每个阶段结束时，通过以下方式验证学习成果：
1. **阶段1**：成功加载 Qwen2.5-7B 并完成一段对话
2. **阶段2**：用 vLLM 启动 API 服务，用 curl 调用成功
3. **阶段3**：对比量化前后推理速度和显存占用的数据
4. **阶段4**：用 Python 客户端调用自部署 API 完成流式对话
5. **阶段5**：Docker 部署 + Prometheus 监控面板可见指标
6. **阶段6**：完整项目跑通，两个模型均可通过 API 访问

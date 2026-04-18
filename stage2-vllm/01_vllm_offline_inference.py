"""
Stage 2: vLLM 离线推理 + 与 Transformers 对比

学习目标：
1. 掌握 vllm.LLM 离线推理的基本用法
2. 理解 SamplingParams 参数调优
3. 对比 vLLM 与原生 Transformers 的推理速度
4. 理解 vLLM 的 prefill/decode 流程

核心概念：
- vLLM 离线推理：用 vllm.LLM 类直接在 Python 中批量推理，不需要启动服务
- SamplingParams：控制生成行为的参数（temperature, top_p, max_tokens 等）
- prefill：处理输入 prompt 的阶段（并行计算所有 token 的 KV Cache）
- decode：逐 token 生成的阶段（利用 KV Cache 增量计算）
- vLLM 比原生 Transformers 快的核心原因：PagedAttention + Continuous Batching
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import time
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def basic_inference():
    """
    vLLM 离线推理的基本用法

    vllm.LLM 是 vLLM 的离线推理接口：
    - 自动管理 GPU 显存和 KV Cache
    - 支持批量推理（多个 prompt 一起处理）
    - 内部使用 PagedAttention 优化显存
    """
    print("=" * 60)
    print("vLLM 离线推理基本用法")
    print("=" * 60)

    # 创建 LLM 实例
    # vLLM 会自动下载模型（如果本地没有）并加载到 GPU  ，本地有，他怎么查找
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,       # 单卡推理
        gpu_memory_utilization=0.9,   # GPU 显存利用率（默认 0.9）
        max_model_len=4096,           # 最大上下文长度（影响 KV Cache 预分配）
        dtype="bfloat16",             # 精度
    )

    # 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.7,      # 温度：越高越随机，0 = greedy
        top_p=0.8,            # nucleus sampling：只从概率前 80% 的 token 中采样
        max_tokens=256,       # 最大生成 token 数
        stop=["</s>"],        # 遇到停止符就停止
    )

    # 单条推理
    prompt = "请用三句话介绍量子计算。"
    


    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.time() - start
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"生成文本: {output.outputs[0].text}")
        print(f"生成 token 数: {len(output.outputs[0].token_ids)}")
        print(f"finish_reason: {output.outputs[0].finish_reason}")
        print(f"总耗时: {elapsed:.2f}s")
    print()

    del llm


def batch_inference():
    """
    批量推理：vLLM 的核心优势

    vLLM 的 Continuous Batching 可以高效处理多个请求：
    - 不需要等所有请求都完成才开始新的
    - 短请求完成后立即释放资源给新请求
    - 这比 Transformers 的逐条推理快得多
    """
    print("=" * 60)
    print("vLLM 批量推理")
    print("=" * 60)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )

    # 多条 prompt 批量推理
    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
        "什么是 Transformer 架构？",
    ]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        print(f"[{i+1}] Prompt: {output.prompt}")
        print(f"    生成 ({num_tokens} tokens): {generated[:100]}...")
    print()
    print(f"批量推理 {len(prompts)} 条，总耗时: {elapsed:.2f}s")
    print(f"平均每条: {elapsed/len(prompts):.2f}s")
    print()

    del llm


def sampling_params_demo():
    """
    SamplingParams 参数调优演示

    不同参数对生成结果的影响：
    - temperature=0: greedy decoding，确定性输出
    - temperature>0: 随机采样，越高越多样
    - top_p: nucleus sampling，限制候选 token 范围
    - top_k: 只从概率最高的 k 个 token 中采样
    - repetition_penalty: 重复惩罚
    """
    print("=" * 60)
    print("SamplingParams 参数调优")
    print("=" * 60)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    prompt = "写一首关于春天的短诗。"

    # 不同的采样策略
    configs = [
        ("Greedy (temperature=0)", SamplingParams(temperature=0, max_tokens=128)),
        ("Low temperature (0.3)", SamplingParams(temperature=0.3, max_tokens=128)),
        ("High temperature (1.5)", SamplingParams(temperature=1.5, max_tokens=128)),
        ("Nucleus sampling (top_p=0.5)", SamplingParams(temperature=0.7, top_p=0.5, max_tokens=128)),
        ("Top-K sampling (top_k=10)", SamplingParams(temperature=0.7, top_k=10, max_tokens=128)),
        ("Repetition penalty (1.5)", SamplingParams(temperature=0.7, repetition_penalty=1.5, max_tokens=128)),
    ]

    for name, params in configs:
        outputs = llm.generate([prompt], params)
        generated = outputs[0].outputs[0].text
        print(f"--- {name} ---")
        print(f"{generated[:150]}")
        print()

    del llm


def compare_with_transformers():
    """
    vLLM vs Transformers 推理速度对比

    同一模型、同一硬件，对比两种框架的推理速度
    """
    print("=" * 60)
    print("vLLM vs Transformers 推理速度对比")
    print("=" * 60)

    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
    ]
    max_new_tokens = 128

    # --- vLLM ---
    print("--- vLLM ---")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,   ##？
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_new_tokens,
    )


    # warmup
    llm.generate(["warmup"], SamplingParams(max_tokens=1))

    start = time.time()
    vllm_outputs = llm.generate(prompts, sampling_params)
    vllm_time = time.time() - start
    vllm_total_tokens = sum(len(o.outputs[0].token_ids) for o in vllm_outputs)

    print(f"批量推理 {len(prompts)} 条，耗时: {vllm_time:.2f}s")
    print(f"总生成 tokens: {vllm_total_tokens}")
    print(f"吞吐量: {vllm_total_tokens / vllm_time:.1f} tokens/s")

    del llm

    # --- Transformers ---
    print()
    print("--- Transformers ---")
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # warmup
    warmup_input = tokenizer("warmup", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**warmup_input, max_new_tokens=1)

    hf_total_tokens = 0
    start = time.time()
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)
        hf_total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]
    hf_time = time.time() - start

    print(f"逐条推理 {len(prompts)} 条，耗时: {hf_time:.2f}s")
    print(f"总生成 tokens: {hf_total_tokens}")
    print(f"吞吐量: {hf_total_tokens / hf_time:.1f} tokens/s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 对比
    print()
    print("--- 对比 ---")
    print(f"vLLM:       {vllm_time:.2f}s ({vllm_total_tokens / vllm_time:.1f} tokens/s)")
    print(f"Transformers: {hf_time:.2f}s ({hf_total_tokens / hf_time:.1f} tokens/s)")
    print(f"加速比: {hf_time / vllm_time:.1f}x")
    print()
    print("vLLM 更快的原因：")
    print("  1. PagedAttention：高效管理 KV Cache 显存，减少碎片")
    print("  2. Continuous Batching：批量请求动态调度，GPU 利用率更高")
    print("  3. 优化的 CUDA kernel：针对推理场景深度优化")
    print("  4. Transformers 是逐条处理，vLLM 是批量并行处理")


def explain_prefill_decode():
    """
    图解 prefill 和 decode 两个阶段

    理解这两个阶段是优化推理性能的基础
    """
    print("=" * 60)
    print("Prefill 和 Decode 阶段图解")
    print("=" * 60)
    print()
    print("LLM 推理分为两个阶段：")
    print()
    print("1. Prefill（预填充）阶段：")
    print("   输入: [你, 好, ，, 请, 介, 绍, 量, 子, 计, 算]")
    print("   - 一次性并行处理所有输入 token")
    print("   - 计算所有 token 的 Q, K, V")
    print("   - 构建 KV Cache")
    print("   - 生成第一个输出 token")
    print("   - 特点：计算密集（compute-bound），GPU 利用率高")
    print()
    print("2. Decode（解码）阶段：")
    print("   逐步: [量] → [子] → [计] → [算] → [是] → ...")
    print("   - 每步只处理 1 个新 token")
    print("   - 利用 KV Cache 避免重复计算")
    print("   - 每步生成 1 个 token")
    print("   - 特点：显存带宽密集（memory-bound），GPU 计算单元闲置")
    print()
    print("性能瓶颈分析：")
    print("  - Prefill: 计算 bound → 提升 GPU 算力有帮助")
    print("  - Decode:  显存带宽 bound → 提升带宽有帮助，增加算力帮助有限")
    print("  - 这也是为什么推理服务更关心显存带宽（HBM）而非纯算力（TFLOPS）")
    print()
    print("vLLM 的优化策略：")
    print("  - Prefill: 利用 GPU 并行能力，高效处理输入")
    print("  - Decode: Continuous Batching 让多个请求共享 GPU，提高利用率")
    print("  - PagedAttention: 减少 KV Cache 显存碎片，支持更多并发请求")
    print()
    print("关键指标：")
    print("  - TTFT (Time To First Token): prefill 阶段耗时 → 用户感知的'首字延迟'")
    print("  - TPOT (Time Per Output Token): decode 阶段每个 token 的耗时 → 影响生成速度")
    print("  - Throughput: 单位时间生成的总 token 数 → 影响服务吞吐量")
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 2: vLLM 离线推理")
    print()

    # 1. prefill/decode 概念讲解（不需要 GPU）
    # explain_prefill_decode()

    # 2. 基本推理
    # basic_inference()

    # # 3. 批量推理
    # batch_inference()

    # 4. SamplingParams 调优
    # sampling_params_demo()

    # 5. 与 Transformers 速度对比
    compare_with_transformers()

    print("=" * 60)
    print("Stage 2 离线推理要点总结")
    print("=" * 60)
    print("1. vllm.LLM 是离线推理接口，支持批量推理")
    print("2. SamplingParams 控制生成行为（temperature, top_p, max_tokens）")
    print("3. 推理分为 prefill（计算密集）和 decode（带宽密集）两个阶段")
    print("4. vLLM 批量推理比 Transformers 逐条推理快数倍")
    print("5. 核心优化：PagedAttention + Continuous Batching → 下一脚本详解")

"""
Stage 1: KV Cache 原理演示

学习目标：
1. 理解 KV Cache 是什么，为什么对推理性能至关重要
2. 对比有/无 KV Cache 的推理速度差异
3. 观察 KV Cache 显存随序列长度的增长规律
4. 理解 KV Cache 对长上下文的影响

核心概念：
- 自回归生成：LLM 逐 token 生成，每步需要看到之前所有 token 的 K/V 向量
- 没有 KV Cache：每生成一个 token 都要重新计算所有前序 token 的 K/V → O(n²) 复杂度
- 有 KV Cache：缓存已计算的 K/V，新 token 只需计算自己的 K/V → O(n) 复杂度
- 代价：KV Cache 占用显存，且随序列长度线性增长
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def total_gpu_memory_allocated():
    """返回所有 GPU 的已分配显存总和（GB）"""
    return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


def total_gpu_max_memory_allocated():
    """返回所有 GPU 的峰值已分配显存总和（GB）"""
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


def print_gpu_memory(label=""):
    """打印当前每张 GPU 的显存使用情况"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 预留 {reserved:.2f}GB / 总计 {total:.2f}GB  {label}")
    print()


def clear_gpu():
    """清理 GPU 显存"""
    gc.collect()
    torch.cuda.empty_cache()


def explain_kv_cache():
    """
    用图示解释 KV Cache 的原理
    """
    print("=" * 60)
    print("KV Cache 原理图解")
    print("=" * 60)
    print()
    print("自回归生成过程（以 4 个 token 为例）：")
    print()
    print("Step 1: 输入 [A]")
    print("  计算 Q,K,V → 生成 B")
    print("  KV Cache: [A的K, A的V]")
    print()
    print("Step 2: 输入 [B]（只需要新的 B）")
    print("  有 Cache:  用缓存的 [A的K,V] + 新的 [B的K,V] → 生成 C")
    print("  无 Cache:  重新计算 [A,B] 的 K,V → 生成 C")
    print()
    print("Step 3: 输入 [C]")
    print("  有 Cache:  用缓存的 [A,B的K,V] + 新的 [C的K,V] → 生成 D")
    print("  无 Cache:  重新计算 [A,B,C] 的 K,V → 生成 D")
    print()
    print("结论：")
    print("  - 无 Cache: 每步要重算所有前序 token → 总计算量 O(n²)")
    print("  - 有 Cache: 每步只算 1 个新 token → 总计算量 O(n)")
    print("  - 代价: KV Cache 占用显存，且随序列长度线性增长")
    print()


def compare_with_without_cache():
    """
    对比有/无 KV Cache 的推理速度

    使用 past_key_values 参数控制是否使用缓存
    """
    print("=" * 60)
    print("对比：有/无 KV Cache 的推理速度")
    print("=" * 60)

    clear_gpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   #加载的什么
    model = AutoModelForCausalLM.from_pretrained(  #加载模型？
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",     #什么意思
    )

    prompt = "请详细介绍深度学习的发展历史，从最早的感知机开始，"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  
    input_length = inputs.input_ids.shape[1]
    print(f"输入长度: {input_length} tokens")
    print(f"提示词: {prompt}")
    print()

    # ---- 有 KV Cache（默认行为）----
    print("--- 有 KV Cache ---")
    print_gpu_memory("生成前")

    start = time.time()
    with torch.no_grad():
        outputs_cached = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,   # 默认就是 True
            do_sample=False,  # greedy 确保可对比
        )
        print(outputs_cached)
    time_cached = time.time() - start

    generated_cached = tokenizer.decode(
        outputs_cached[0][input_length:], skip_special_tokens=True
    )
    print(f"生成 {256} tokens, 耗时: {time_cached:.2f}s")
    print_gpu_memory("生成后（含 KV Cache）")

    # ---- 无 KV Cache ----
    print("--- 无 KV Cache ---")
    clear_gpu()
    # 重新加载模型，确保显存状态干净
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    start = time.time()
    with torch.no_grad():
        outputs_no_cache = model2.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=False,  # 禁用 KV Cache
            do_sample=False,
        )
    time_no_cache = time.time() - start

    generated_no_cache = tokenizer.decode(
        outputs_no_cache[0][input_length:], skip_special_tokens=True
    )
    print(f"生成 {256} tokens, 耗时: {time_no_cache:.2f}s")
    print_gpu_memory("生成后（无 KV Cache）")

    # 对比
    print("--- 对比 ---")
    print(f"有 Cache: {time_cached:.2f}s")
    print(f"无 Cache: {time_no_cache:.2f}s")
    if time_cached > 0:
        print(f"加速比:   {time_no_cache/time_cached:.1f}x")
    print()
    print("注意: 无 Cache 模式下，每生成一个 token 都要重算全部前序 token，")
    print("随着生成长度增加，速度差距会越来越大。")
    print()

    del model, model2
    clear_gpu()

    return time_cached, time_no_cache


def measure_kv_cache_memory():
    """
    测量 KV Cache 显存随序列长度的增长

    逐步增加生成长度，观察显存变化
    """
    print("=" * 60)
    print("KV Cache 显存随序列长度的增长")
    print("=" * 60)

    clear_gpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 记录模型本身的总显存（所有 GPU 之和）
    base_memory = total_gpu_memory_allocated()
    print(f"模型加载后显存（全部GPU）: {base_memory:.2f}GB")
    print()

    prompt = "从前有一座山，山上有座庙，"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 手动逐步生成，在 KV Cache 还活着时测量显存
    # model.generate() 返回后 KV Cache 被释放，无法测量
    # 手动 forward 的好处：past_key_values 保持在变量中，显存可观测
    gen_lengths = [128, 512, 2048, 8192]
    results = []

    # config 信息（用于理论值计算）
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dtype_size = 2  # BF16
    input_len = inputs.input_ids.shape[1]

    print(f"{'总序列长度':>10} {'KV Cache 实测':>15} {'理论值':>10}")
    print("-" * 40)

    for max_new in gen_lengths:
        # 每轮从零开始：prefill → 逐步生成
        gc.collect()
        mem_before = total_gpu_memory_allocated()

        # prefill
        with torch.no_grad():
            out = model(inputs.input_ids, use_cache=True)
        past = out.past_key_values
        token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out

        # decode
        for _ in range(max_new - 1):
            with torch.no_grad():
                out = model(token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            del out

        # KV Cache 在 past 中活着，测量显存增量
        current_memory = total_gpu_memory_allocated()
        kv_delta = current_memory - mem_before
        total_seq_len = input_len + max_new

        # 理论值
        kv_theory = 2 * num_layers * total_seq_len * num_kv_heads * head_dim * dtype_size / 1024**3

        results.append((total_seq_len, kv_delta, kv_theory))
        print(f"{total_seq_len:>10} {kv_delta:>12.2f}GB {kv_theory:>8.2f}GB")

        # 释放本轮 KV Cache
        del past, token
        gc.collect()

    print()
    print("观察：")
    print("1. 实测值应接近理论值，KV Cache 显存随序列长度线性增长")
    print("2. 这就是为什么长上下文（如 128K）会占用大量显存")
    print("3. vLLM 的 PagedAttention 就是为了优化 KV Cache 显存管理")
    print()

    # 计算 KV Cache 的理论大小
    print("--- KV Cache 理论大小计算 ---")
    print()
    print("KV Cache 大小公式：")
    print("  KV Cache = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size")
    print()
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dtype_size = 2  # BF16 = 2 bytes

    print(f"{MODEL_NAME} 模型配置：")
    print(f"  num_layers (层数):        {num_layers}")
    print(f"  num_kv_heads (KV头数):    {num_kv_heads}")
    print(f"  head_dim (头维度):         {head_dim}")
    print(f"  dtype_size (字节/参数):    {dtype_size} (BF16)")
    print()

    for seq_len in [512, 2048, 8192, 32768, 131072]:
        kv_cache_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_size
        kv_cache_gb = kv_cache_bytes / 1024**3
        print(f"  seq_len={seq_len:>6}: KV Cache ≈ {kv_cache_gb:.2f}GB")

    # 同时展示 7B 的对比，说明大模型的 KV Cache 更大
    print()
    print("--- 对比：Qwen2.5-7B 的 KV Cache（理论值）---")
    print("  7B 的 KV 头数=4, 层数=28, head_dim=128")
    for seq_len in [512, 2048, 8192, 32768, 131072]:
        kv_cache_bytes = 2 * 28 * seq_len * 4 * 128 * dtype_size
        kv_cache_gb = kv_cache_bytes / 1024**3
        print(f"  seq_len={seq_len:>6}: KV Cache ≈ {kv_cache_gb:.2f}GB")

    print()
    print("注意: 这是单个请求的 KV Cache！多请求并发时，KV Cache 显存会成倍增长。")
    print("这也是为什么推理服务的吞吐量受限于 KV Cache 显存，而不是模型本身。")

    del model
    clear_gpu()


def demo_batch_kv_cache():
    """
    演示批量推理时 KV Cache 的显存影响

    多个并发请求各自的 KV Cache 独立占用显存
    """
    print("=" * 60)
    print("批量推理的 KV Cache 显存影响")
    print("=" * 60)

    clear_gpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"  # decoder-only 模型需要左填充
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    base_memory = total_gpu_memory_allocated()
    print(f"模型加载后显存（全部GPU）: {base_memory:.2f}GB")
    print()

    # 模拟不同 batch size 的推理
    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
    ]

    print(f"{'Batch Size':>10} {'显存占用':>10} {'KV Cache 总量':>15}")
    print("-" * 45)

    for batch_size in [1, 2, 4]:
        batch_prompts = prompts[:batch_size]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=64,
                use_cache=True,
                do_sample=False,
            )

        current_memory = total_gpu_memory_allocated()
        kv_delta = current_memory - base_memory
        print(f"{batch_size:>10} {current_memory:>10.2f}GB {kv_delta:>12.2f}GB")

        del outputs, batch_inputs

    print()
    print("观察：")
    print("1. batch 越大，KV Cache 显存占用越大（近似线性增长）")
    print("2. 推理服务的并发能力受 KV Cache 显存限制")
    print("3. vLLM 的 PagedAttention 可以更高效地管理这些 KV Cache 显存")
    print("   → 减少碎片，允许更多并发请求")

    del model
    clear_gpu()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 1: KV Cache 原理演示")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print()

    # 1. 概念讲解
    # explain_kv_cache()

    # 2. 有/无 KV Cache 速度对比
    # compare_with_without_cache()

    # 3. KV Cache 显存随序列长度增长
    measure_kv_cache_memory()

    # 4. 批量推理的 KV Cache 影响
    # demo_batch_kv_cache()

    print("=" * 60)
    print("Stage 1 KV Cache 要点总结")
    print("=" * 60)
    print("1. KV Cache 缓存已计算的 Key/Value，避免重复计算")
    print("2. 有 Cache: O(n) 复杂度；无 Cache: O(n²) 复杂度")
    print("3. KV Cache 显存随序列长度线性增长")
    print("4. 多请求并发时，KV Cache 显存成倍增长")
    print("5. 推理服务的吞吐量主要受 KV Cache 显存限制，而非模型大小")
    print("6. vLLM 的 PagedAttention 专门优化 KV Cache 显存管理 → Stage 2 详解")

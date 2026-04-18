"""
Stage 2: PagedAttention + Continuous Batching 原理演示

学习目标：
1. 理解 PagedAttention 的原理和优势
2. 对比传统 KV Cache 管理与 PagedAttention 的显存利用率
3. 理解 Continuous Batching vs Static Batching
4. 掌握 gpu-memory-utilization 和 max-model-len 参数的影响

核心概念：
- PagedAttention：借鉴操作系统虚拟内存的分页机制管理 KV Cache
- 传统方式：为每个请求预分配最大长度的连续显存 → 严重浪费
- PagedAttention：按需分配固定大小的"页"，减少碎片 → 显存利用率高
- Continuous Batching：请求完成后立即替换，不等整批完成
- Static Batching：等最慢的请求完成后才开始下一批 → GPU 闲置
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

hf_home = os.environ.get("HF_HOME", "")
if hf_home:
    print(f"HF_HOME={hf_home}")
else:
    print("⚠️  HF_HOME 未设置！vLLM 可能找不到本地模型缓存。")
    print("   建议: export HF_HOME=/data2/lvliping/hf_cache")
print()

import time
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def explain_paged_attention():
    """
    PagedAttention 原理图解

    类比操作系统的虚拟内存管理来理解
    """
    print("=" * 60)
    print("PagedAttention 原理图解")
    print("=" * 60)
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│            传统 KV Cache 管理                        │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  请求A: [████████████████████░░░░░░░░░░]  预分配最大长度│")
    print("│  请求B: [██████░░░░░░░░░░░░░░░░░░░░░░░░]  预分配最大长度│")
    print("│  请求C: [█████████████░░░░░░░░░░░░░░░░░░]  预分配最大长度│")
    print("│                                ↑ 大量浪费的空白区域    │")
    print("│  问题：                                                │")
    print("│  1. 每个请求预分配 max_seq_len 的连续显存               │")
    print("│  2. 实际生成长度 < max_seq_len → 大量浪费              │")
    print("│  3. 显存碎片：无法拼凑零散空间给新请求                  │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│            PagedAttention（分页管理）                  │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  Page Table:                                          │")
    print("│  请求A: [Page0]→[Page1]→[Page2]→[Page5]             │")
    print("│  请求B: [Page3]→[Page4]                              │")
    print("│  请求C: [Page6]→[Page7]→[Page8]                     │")
    print("│                                                       │")
    print("│  物理显存（Block Pool）:                               │")
    print("│  [P0][P1][P2][P3][P4][P5][P6][P7][P8][P9]...        │")
    print("│  ↑ 每个 Block 固定大小（如 16 tokens 的 KV Cache）    │")
    print("│  ↑ 按需分配，不需要连续                                │")
    print("│                                                       │")
    print("│  优势：                                                │")
    print("│  1. 只分配实际需要的页，没有预分配浪费                  │")
    print("│  2. 页可以不连续，消除显存碎片问题                      │")
    print("│  3. 请求完成后页立即回收，给新请求使用                  │")
    print("│  4. 类比：OS 的虚拟内存 → 物理内存映射                 │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("关键参数：")
    print("  - block_size: 每页的 token 数（默认 16）")
    print("  - gpu_memory_utilization: GPU 显存利用率（默认 0.9）")
    print("    → 决定了 Block Pool 的总大小")
    print("    → 直接影响能同时服务多少请求")
    print()


def explain_continuous_batching():
    """
    Continuous Batching vs Static Batching 图解
    """
    print("=" * 60)
    print("Continuous Batching vs Static Batching")
    print("=" * 60)
    print()
    print("假设有 4 个请求，生成长度不同：")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  Static Batching（传统方式）                          │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  时间 →                                               │")
    print("│  请求A: [prefill][decode........]  ← 先完成，但等...   │")
    print("│  请求B: [prefill][decode..............]               │")
    print("│  请求C: [prefill][decode........................]      │")
    print("│  请求D: [prefill][decode......................]  ← 最慢│")
    print("│                                     ↑ 等最慢的完成     │")
    print("│  下一批:                           [prefill]...       │")
    print("│                                                       │")
    print("│  问题：                                                │")
    print("│  1. 先完成的请求空占 GPU 位置                           │")
    print("│  2. 必须等最慢的请求完成才能开始下一批                  │")
    print("│  3. GPU 利用率低                                       │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  Continuous Batching（vLLM 方式）                     │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  时间 →                                               │")
    print("│  请求A: [pf][dec....]                                  │")
    print("│  请求B: [pf][dec........]                              │")
    print("│  请求C: [pf][dec........................]              │")
    print("│  请求D: [pf][dec..............]                        │")
    print("│  请求E:           [pf][dec....]  ← A完成后立即插入     │")
    print("│  请求F:                 [pf][dec......]  ← B完成后插入 │")
    print("│                                                       │")
    print("│  优势：                                                │")
    print("│  1. 请求完成后立即释放资源                              │")
    print("│  2. 新请求可以立即加入，无需等整批完成                  │")
    print("│  3. GPU 利用率显著提高                                  │")
    print("│  4. 用户感知的延迟更低（排队时间短）                    │")
    print("└─────────────────────────────────────────────────────┘")
    print()


def demo_gpu_memory_utilization():
    """
    gpu_memory_utilization 参数的影响

    这个参数决定了 vLLM 预留多少 GPU 显存用于 KV Cache
    - 太低：并发能力受限，容易 OOM
    - 太高：可能与其他进程冲突
    """
    print("=" * 60)
    print("gpu_memory_utilization 参数影响")
    print("=" * 60)
    print()

    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
        "什么是 Transformer 架构？",
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

    for util in [0.5, 0.7, 0.9]:
        print(f"--- gpu_memory_utilization = {util} ---")
        try:
            llm = LLM(
                model=MODEL_NAME,
                tensor_parallel_size=1,
                gpu_memory_utilization=util,
                max_model_len=2048,
                dtype="bfloat16",
                enforce_eager=True,  # 关闭 CUDA Graph，减少启动时间
            )

            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(f"  推理成功！耗时: {elapsed:.2f}s, tokens: {total_tokens}")
            print(f"  吞吐量: {total_tokens / elapsed:.1f} tokens/s")

            del llm
        except Exception as e:
            print(f"  推理失败: {e}")
        print()

    print("观察：")
    print("  - gpu_memory_utilization 越高 → KV Cache 可用空间越大 → 并发能力越强")
    print("  - gpu_memory_utilization 越低 → 留给其他进程的空间越大")
    print("  - 默认 0.9 是大多数场景的最佳值")
    print("  - 如果 GPU 上还有其他进程，需要适当降低")
    print()


def demo_max_model_len():
    """
    max-model-len 参数的影响

    这个参数决定了最大上下文长度，直接影响 KV Cache 预分配大小
    - 太大：KV Cache 预分配多，可并发请求少
    - 太小：长文本被截断
    """
    print("=" * 60)
    print("max-model-len 参数影响")
    print("=" * 60)
    print()

    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

    for max_len in [512, 2048, 8192]:
        print(f"--- max_model_len = {max_len} ---")
        try:
            llm = LLM(
                model=MODEL_NAME,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=max_len,
                dtype="bfloat16",
                enforce_eager=True,
            )

            # 尝试更多并发来测试限制
            test_prompts = prompts * 4  # 12 条并发
            start = time.time()
            outputs = llm.generate(test_prompts, sampling_params)
            elapsed = time.time() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(f"  推理成功！{len(test_prompts)} 条并发，耗时: {elapsed:.2f}s")
            print(f"  吞吐量: {total_tokens / elapsed:.1f} tokens/s")

            del llm
        except Exception as e:
            print(f"  推理失败: {e}")
        print()

    print("观察：")
    print("  - max_model_len 越大 → 每个请求的 KV Cache 上限越高 → 可并发请求越少")
    print("  - max_model_len 越小 → KV Cache 上限越低 → 可并发更多请求")
    print("  - 实际部署需要根据业务需求权衡：长上下文 vs 高并发")
    print()
    print("KV Cache 显存估算（回顾 Stage 1）：")
    print("  Qwen2.5-7B, BF16, batch=1:")
    print("    max_len=512:   KV Cache ≈ 0.03 GB")
    print("    max_len=2048:  KV Cache ≈ 0.11 GB")
    print("    max_len=8192:  KV Cache ≈ 0.44 GB")
    print("    max_len=32768: KV Cache ≈ 1.75 GB")
    print("  多请求并发时线性增长 → max_model_len 直接决定最大并发数")
    print()


def demo_concurrent_capacity():
    """
    演示 vLLM 的并发能力

    对比 vLLM 批量推理与逐条推理的效率差异
    这体现了 Continuous Batching 的实际效果
    """
    print("=" * 60)
    print("vLLM 并发推理能力演示")
    print("=" * 60)
    print()

    num_prompts = 16
    base_prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
    ]
    prompts = base_prompts * (num_prompts // len(base_prompts))
    sampling_params = SamplingParams(temperature=0.7, max_tokens=64)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
    )

    # 一次性批量推理
    print(f"一次性批量推理 {len(prompts)} 条...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    batch_time = time.time() - start
    batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  耗时: {batch_time:.2f}s")
    print(f"  吞吐量: {batch_tokens / batch_time:.1f} tokens/s")
    print(f"  平均每条: {batch_time / len(prompts):.2f}s")
    print()

    # 对比：如果用 Transformers 逐条推理
    # 估算（不实际运行，避免二次加载模型）
    est_per_prompt = 2.0  # 大致估算：7B 模型 64 tokens 约 2s
    est_sequential_time = est_per_prompt * len(prompts)
    print(f"估算 Transformers 逐条推理 {len(prompts)} 条：")
    print(f"  估算耗时: ~{est_sequential_time:.1f}s")
    print(f"  vLLM 加速比: ~{est_sequential_time / batch_time:.1f}x")
    print()
    print("关键：vLLM 的 Continuous Batching 让所有请求共享 GPU，")
    print("而 Transformers 逐条处理时 GPU 大部分时间在等一个请求的 decode。")

    del llm


if __name__ == "__main__":
    print("大模型部署学习 - Stage 2: PagedAttention + Continuous Batching")
    print()

    # 1. 概念讲解（不需要 GPU）
    explain_paged_attention()
    explain_continuous_batching()

    # 2. gpu_memory_utilization 参数影响
    # demo_gpu_memory_utilization()

    # 3. max_model_len 参数影响
    # demo_max_model_len()

    # 4. 并发能力演示
    # demo_concurrent_capacity()

    print("=" * 60)
    print("Stage 2 PagedAttention 要点总结")
    print("=" * 60)
    print("1. 传统方式为每个请求预分配最大长度显存 → 严重浪费 + 碎片")
    print("2. PagedAttention 按需分页分配 KV Cache → 高利用率 + 无碎片")
    print("3. Continuous Batching 动态调度 → GPU 利用率远高于 Static Batching")
    print("4. gpu_memory_utilization 控制 KV Cache 可用显存比例（默认 0.9）")
    print("5. max_model_len 决定 KV Cache 上限 → 权衡长上下文 vs 高并发")
    print("6. 两者共同决定推理服务的最大并发能力")

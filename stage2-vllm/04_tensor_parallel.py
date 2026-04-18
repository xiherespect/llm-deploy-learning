"""
Stage 2: 多卡推理（Tensor Parallelism）+ 更大模型

学习目标：
1. 理解 Tensor Parallelism (TP) 的原理
2. 对比单卡 vs 多卡推理的性能差异
3. 尝试更大模型（32B/72B 量化版）
4. 掌握关键参数调优：gpu-memory-utilization, max-model-len, dtype, enforce-eager
5. 理解吞吐量 vs 延迟的 trade-off

核心概念：
- Tensor Parallelism：将模型权重按列切分到多张 GPU，每张卡只存一部分
- 与 Pipeline Parallelism 的区别：TP 按层内切分（每层都涉及所有卡），PP 按层间切分
- TP 的代价：每步都需要卡间通信（AllReduce），通信开销随卡数增加
- 适合场景：单卡放不下的大模型，或需要更高吞吐的场景

硬件环境：
- 4× RTX 3090 (24GB × 4 = 96GB 总显存)
- Qwen2.5-7B BF16: ~15GB → 单卡可跑
- Qwen2.5-32B AWQ: ~20GB → 2卡 TP 可跑
- Qwen2.5-72B AWQ: ~40GB → 4卡 TP 可跑
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

MODEL_7B = "Qwen/Qwen2.5-7B-Instruct"
MODEL_32B_AWQ = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def explain_tensor_parallelism():
    """
    Tensor Parallelism 原理图解
    """
    print("=" * 60)
    print("Tensor Parallelism (TP) 原理图解")
    print("=" * 60)
    print()
    print("问题：7B BF16 ≈ 15GB，单卡 3090 (24GB) 可以跑")
    print("      32B BF16 ≈ 65GB，单卡放不下！")
    print("      72B BF16 ≈ 148GB，4卡也放不下 BF16！")
    print()
    print("解决方案：Tensor Parallelism — 把模型"'切碎'"分到多张卡")
    print()
    print("┌────────────────────────────────────────────────────────┐")
    print("│  单卡推理 (TP=1)                                        │")
    print("├────────────────────────────────────────────────────────┤")
    print("│  GPU 0: [完整模型权重] [KV Cache] [推理计算]             │")
    print("│  瓶颈：模型太大放不下，或显存不够做 KV Cache              │")
    print("└────────────────────────────────────────────────────────┘")
    print()
    print("┌────────────────────────────────────────────────────────┐")
    print("│  两卡推理 (TP=2)                                        │")
    print("├────────────────────────────────────────────────────────┤")
    print("│  GPU 0: [模型左半边权重] [KV Cache]                      │")
    print("│  GPU 1: [模型右半边权重] [KV Cache]                      │")
    print("│  每步推理后: AllReduce 通信合并结果                       │")
    print("│  效果：每卡只需存一半权重，显存压力减半                    │")
    print("└────────────────────────────────────────────────────────┘")
    print()
    print("┌────────────────────────────────────────────────────────┐")
    print("│  四卡推理 (TP=4)                                        │")
    print("├────────────────────────────────────────────────────────┤")
    print("│  GPU 0: [1/4权重]  GPU 1: [1/4权重]                     │")
    print("│  GPU 2: [1/4权重]  GPU 3: [1/4权重]                     │")
    print("│  AllReduce 通信量更大，但可用总显存 = 24GB × 4 = 96GB    │")
    print("└────────────────────────────────────────────────────────┘")
    print()
    print("TP vs PP (Pipeline Parallelism)：")
    print("  TP: 按层内切分（每个 Attention/FFN 层都分到多卡）")
    print("      → 每步都需要卡间通信（AllReduce）")
    print("      → 适合单机多卡（NVLink/PCIe 通信延迟低）")
    print()
    print("  PP: 按层间切分（前几层放 GPU0，后几层放 GPU1）")
    print("      → 只在层边界通信，通信少")
    print("      → 但有"'气泡'"（pipeline bubble），GPU 利用率低")
    print("      → 适合多机多卡场景")
    print()
    print("经验法则：")
    print("  - 单机多卡（同机器）：优先用 TP")
    print("  - 多机多卡（跨机器）：TP + PP 组合")
    print()


def benchmark_tp_sizes():
    """
    对比不同 TP size 的推理性能

    用 7B 模型测试，因为它在单卡和多卡都能跑
    """
    print("=" * 60)
    print("不同 TP Size 的推理性能对比（7B 模型）")
    print("=" * 60)
    print()

    prompts = [
        "什么是机器学习？",
        "解释量子纠缠的概念。",
        "Python 的 GIL 是什么？",
        "深度学习和传统机器学习有什么区别？",
        "什么是 Transformer 架构？",
        "解释反向传播算法。",
        "什么是卷积神经网络？",
        "RNN 和 LSTM 有什么区别？",
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

    results = {}

    for tp_size in [1, 2, 4]:
        print(f"--- TP={tp_size} ---")
        try:
            llm = LLM(
                model=MODEL_7B,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=0.9,
                max_model_len=2048,
                dtype="bfloat16",
                enforce_eager=True,  # 关闭 CUDA Graph 以加快启动
            )

            # warmup
            llm.generate(["warmup"], SamplingParams(max_tokens=1))

            # 实测
            start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.time() - start

            total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_tokens / elapsed

            results[tp_size] = {
                "time": elapsed,
                "tokens": total_tokens,
                "throughput": throughput,
            }

            print(f"  耗时: {elapsed:.2f}s")
            print(f"  总 tokens: {total_tokens}")
            print(f"  吞吐量: {throughput:.1f} tokens/s")

            del llm
        except Exception as e:
            print(f"  失败: {e}")
        print()

    # 汇总对比
    if results:
        print("--- 对比 ---")
        print(f"{'TP Size':>8} {'耗时':>8} {'吞吐量':>12}")
        print("-" * 32)
        for tp, data in sorted(results.items()):
            print(f"{tp:>8} {data['time']:>7.2f}s {data['throughput']:>10.1f} t/s")

    print()
    print("观察：")
    print("  - TP 增加不一定线性提升吞吐量（受通信开销影响）")
    print("  - TP 主要解决"'放不下'"的问题，而非"'跑不快'"的问题")
    print("  - 7B 模型单卡足够，多卡 TP 的收益有限")
    print("  - 但对于 32B/72B 模型，TP 是必须的（单卡放不下）")
    print()


def benchmark_larger_model():
    """
    尝试更大模型：Qwen2.5-32B AWQ

    32B AWQ 量化约 20GB，2卡 TP 可以跑
    这才是 TP 的真正价值——跑单卡放不下的模型
    """
    print("=" * 60)
    print("更大模型推理：Qwen2.5-32B AWQ (2卡 TP)")
    print("=" * 60)
    print()

    print("模型对比：")
    print("  Qwen2.5-7B  BF16: ~15GB → 单卡")
    print("  Qwen2.5-32B AWQ:  ~20GB → 2卡 TP")
    print("  Qwen2.5-72B AWQ:  ~40GB → 4卡 TP")
    print()

    prompts = [
        "请详细解释 Transformer 中的多头注意力机制。",
        "比较 Rust 和 Go 语言的设计哲学。",
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # 先跑 7B 作为基准
    print("--- Qwen2.5-7B BF16 (TP=1) ---")
    try:
        llm_7b = LLM(
            model=MODEL_7B,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            dtype="bfloat16",
        )

        start = time.time()
        outputs_7b = llm_7b.generate(prompts, sampling_params)
        time_7b = time.time() - start
        tokens_7b = sum(len(o.outputs[0].token_ids) for o in outputs_7b)

        for i, output in enumerate(outputs_7b):
            print(f"  [{i+1}] {output.outputs[0].text[:100]}...")
        print(f"  耗时: {time_7b:.2f}s, 吞吐: {tokens_7b/time_7b:.1f} tokens/s")

        del llm_7b
    except Exception as e:
        print(f"  7B 推理失败: {e}")
        time_7b = tokens_7b = 0

    print()

    # 跑 32B AWQ
    print("--- Qwen2.5-32B AWQ (TP=2) ---")
    try:
        llm_32b = LLM(
            model=MODEL_32B_AWQ,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            dtype="bfloat16",
            quantization="awq",
        )

        start = time.time()
        outputs_32b = llm_32b.generate(prompts, sampling_params)
        time_32b = time.time() - start
        tokens_32b = sum(len(o.outputs[0].token_ids) for o in outputs_32b)

        for i, output in enumerate(outputs_32b):
            print(f"  [{i+1}] {output.outputs[0].text[:100]}...")
        print(f"  耗时: {time_32b:.2f}s, 吞吐: {tokens_32b/time_32b:.1f} tokens/s")

        del llm_32b
    except Exception as e:
        print(f"  32B AWQ 推理失败: {e}")
        print("  提示：确保已下载模型 Qwen/Qwen2.5-32B-Instruct-AWQ")
        time_32b = tokens_32b = 0

    print()
    if time_7b > 0 and time_32b > 0:
        print("对比：")
        print(f"  7B:  {time_7b:.2f}s, {tokens_7b/time_7b:.1f} t/s")
        print(f"  32B: {time_32b:.2f}s, {tokens_32b/time_32b:.1f} t/s")
        print(f"  32B 更慢（更大的模型），但生成质量通常更好")
    print()


def parameter_tuning_guide():
    """
    关键参数调优指南

    这些参数直接影响服务性能和资源利用
    """
    print("=" * 60)
    print("vLLM 关键参数调优指南")
    print("=" * 60)
    print()

    params = [
        ("--gpu-memory-utilization", "0.9", [
            "控制 vLLM 使用的 GPU 显存比例",
            "越高 → KV Cache 越大 → 并发越多",
            "太低 → 容易 OOM 或并发受限",
            "建议：独占 GPU 时设 0.9~0.95，共享 GPU 时降低",
        ]),
        ("--max-model-len", "模型默认", [
            "最大上下文长度（token 数）",
            "直接影响 KV Cache 预分配大小",
            "越大 → 单请求 KV Cache 越大 → 可并发越少",
            "建议：根据业务需求设置，不需要 128K 就别开",
        ]),
        ("--dtype", "auto", [
            "模型权重精度：float16, bfloat16, auto",
            "BF16 训练和推理更稳定（防溢出）",
            "RTX 3090 (Ampere) 支持 BF16",
            "建议：Ampere+ 架构用 bfloat16",
        ]),
        ("--enforce-eager", "False", [
            "关闭 CUDA Graph，使用 eager 模式",
            "CUDA Graph 优化 decode 速度，但增加启动时间",
            "调试时建议开启（错误信息更清晰）",
            "建议：生产环境关闭，调试时开启",
        ]),
        ("--tensor-parallel-size", "1", [
            "TP 并行度（使用的 GPU 数量）",
            "模型必须能放进 tp × 单卡显存",
            "7B BF16: TP=1, 32B AWQ: TP=2, 72B AWQ: TP=4",
            "建议：能用少卡就不用多卡（减少通信开销）",
        ]),
        ("--max-num-seqs", "256", [
            "最大并发序列数",
            "受 KV Cache 显存限制",
            "太大 → 显存不足；太小 → 吞吐受限",
            "建议：默认值通常够用，调优时观察 /metrics",
        ]),
    ]

    for name, default, tips in params:
        print(f"  {name} (默认: {default})")
        for tip in tips:
            print(f"    - {tip}")
        print()


def throughput_vs_latency():
    """
    吞吐量 vs 延迟的 trade-off

    这是推理服务调优的核心矛盾
    """
    print("=" * 60)
    print("吞吐量 vs 延迟的 Trade-off")
    print("=" * 60)
    print()
    print("推理服务有两个核心指标：")
    print()
    print("  1. 延迟 (Latency)")
    print("     - TTFT: 首 token 延迟（用户等多久才开始看到输出）")
    print("     - TPOT: 每 token 延迟（生成速度）")
    print("     - 用户视角：越低越好")
    print()
    print("  2. 吞吐量 (Throughput)")
    print("     - 单位时间处理的总 token 数")
    print("     - 单位时间完成的请求数")
    print("     - 服务提供方视角：越高越好（硬件利用率高）")
    print()
    print("┌────────────────────────────────────────────────────┐")
    print("│  矛盾：延迟和吞吐量通常是 trade-off                  │")
    print("├────────────────────────────────────────────────────┤")
    print("│                                                      │")
    print("│  高并发（更多请求同时处理）                            │")
    print("│    → 吞吐量高（GPU 利用率高）                         │")
    print("│    → 延迟高（每个请求等待资源）                        │")
    print("│                                                      │")
    print("│  低并发（少请求同时处理）                              │")
    print("│    → 延迟低（资源充裕）                                │")
    print("│    → 吞吐量低（GPU 利用率低）                          │")
    print("│                                                      │")
    print("└────────────────────────────────────────────────────┘")
    print()
    print("调优策略：")
    print()
    print("  场景1：交互式聊天（延迟优先）")
    print("    - 降低并发数，确保每个请求快速响应")
    print("    - max_num_seqs 设小")
    print("    - 考虑用小模型（7B）而非大模型（72B）")
    print()
    print("  场景2：批量处理（吞吐优先）")
    print("    - 提高并发数，最大化 GPU 利用率")
    print("    - gpu_memory_utilization 设高")
    print("    - 可以用大模型 + 量化")
    print()
    print("  场景3：混合场景")
    print("    - 部署多个实例：小模型处理实时请求，大模型处理批量任务")
    print("    - 或用优先级队列：实时请求优先，批量请求排队")
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 2: 多卡推理与参数调优")
    print()

    # 1. TP 原理讲解（不需要 GPU）
    # explain_tensor_parallelism()

    # 2. 不同 TP size 性能对比
    # benchmark_tp_sizes()

    # 3. 更大模型推理
    benchmark_larger_model()

    # 4. 参数调优指南（不需要 GPU）
    # parameter_tuning_guide()

    # 5. 吞吐量 vs 延迟 trade-off（不需要 GPU）
    # throughput_vs_latency()

    print("=" * 60)
    print("Stage 2 多卡推理要点总结")
    print("=" * 60)
    print("1. TP 按层内切分权重到多卡，解决单卡放不下的问题")
    print("2. 单机多卡优先用 TP，多机场景用 TP + PP")
    print("3. TP 增加不一定线性提升吞吐（通信开销）")
    print("4. 7B 单卡，32B AWQ 2卡，72B AWQ 4卡 → 4×3090 的最佳配置")
    print("5. gpu_memory_utilization 和 max_model_len 是最关键的两个参数")
    print("6. 吞吐量 vs 延迟是核心 trade-off，根据场景调优")

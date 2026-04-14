"""
Stage 1: 显存估算实操

学习目标：
1. 掌握显存估算公式：参数量 × 精度 + KV Cache + 开销
2. 计算不同模型/精度下的显存需求
3. 对比理论估算与实际测量的差异
4. 理解推理 vs 训练的显存差异

核心公式：
  推理显存 = 模型权重 + KV Cache + 激活值 + 框架开销
  训练显存 = 模型权重 + 梯度 + 优化器状态 + 激活值 + 临时缓冲

  模型权重 = 参数量 × 每参数字节数
    - FP32: 4 bytes/param
    - FP16/BF16: 2 bytes/param
    - INT8: 1 byte/param
    - INT4: 0.5 bytes/param

  KV Cache = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def print_gpu_memory(label=""):
    """打印当前每张 GPU 的显存使用情况"""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"  GPU {i}: 已分配 {allocated:.2f}GB / 预留 {reserved:.2f}GB / 总计 {total:.2f}GB  {label}")
    print()


def clear_gpu():
    """清理 GPU 显存"""
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 理论计算部分（不需要 GPU）
# ============================================================

def estimate_model_weights_memory(num_params_b, precision="bf16"):
    """
    估算模型权重的显存占用

    Args:
        num_params_b: 参数量（十亿）
        precision: 精度类型
    Returns:
        显存占用（GB）
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    bpb = bytes_per_param[precision]
    memory_gb = num_params_b * 1e9 * bpb / 1024**3
    return memory_gb


def estimate_kv_cache_memory(num_layers, num_kv_heads, head_dim,
                              seq_len, dtype_size=2, batch_size=1):
    """
    估算 KV Cache 的显存占用

    Args:
        num_layers: Transformer 层数
        num_kv_heads: KV 头数（GQA/MQA 时小于 num_attention_heads）
        head_dim: 每个头的维度
        seq_len: 序列长度
        dtype_size: 每个参数的字节数（BF16=2）
        batch_size: 批量大小
    Returns:
        显存占用（GB）
    """
    # K 和 V 各一份，所以乘 2
    kv_cache_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_size * batch_size
    return kv_cache_bytes / 1024**3


def table_model_memory_estimates():
    """
    打印不同模型/精度的显存估算表

    这是部署前最重要的预计算——判断模型能否在你的 GPU 上跑
    """
    print("=" * 70)
    print("显存估算表：不同模型 × 不同精度")
    print("=" * 70)
    print()

    # 常见模型配置（参数量, 层数, KV头数, 头维度）
    models = {
        "Qwen2.5-7B":  (7, 28, 4, 128),
        "Qwen2.5-32B": (32, 64, 8, 128),
        "Qwen2.5-72B": (72, 80, 8, 128),
    }

    precisions = ["bf16", "int8", "int4"]
    seq_len = 2048  # 假设 2048 token 上下文
    batch_size = 1

    print(f"假设: seq_len={seq_len}, batch_size={batch_size}")
    print()
    print(f"{'模型':<18} {'精度':<6} {'权重':>8} {'KV Cache':>10} {'总计(估)':>10}")
    print("-" * 55)

    for model_name, (params_b, layers, kv_heads, head_dim) in models.items():
        for prec in precisions:
            weights = estimate_model_weights_memory(params_b, prec)
            kv_cache = estimate_kv_cache_memory(
                layers, kv_heads, head_dim, seq_len, dtype_size=2, batch_size=batch_size
            )
            total = weights + kv_cache
            print(f"{model_name:<18} {prec:<6} {weights:>7.1f}GB {kv_cache:>8.2f}GB {total:>9.1f}GB")
        print()

    print("你的硬件: 4× RTX 3090 (24GB × 4 = 96GB 总显存)")
    print()
    print("关键判断：")
    print("  ✅ Qwen2.5-7B BF16  (~15GB) → 单卡可跑")
    print("  ✅ Qwen2.5-32B INT4  (~20GB) → 2卡 TP 可跑")
    print("  ✅ Qwen2.5-72B INT4  (~42GB) → 4卡 TP 可跑")
    print("  ❌ Qwen2.5-72B BF16  (~148GB) → 超出 96GB 总显存")
    print()


def table_kv_cache_vs_seq_len():
    """
    展示 KV Cache 随序列长度和 batch size 的变化

    这决定了你的服务能支持多长的上下文和多少并发
    """
    print("=" * 70)
    print("KV Cache 显存 vs 序列长度 × Batch Size（Qwen2.5-7B, BF16）")
    print("=" * 70)
    print()

    # Qwen2.5-7B 配置
    layers, kv_heads, head_dim = 28, 4, 128
    seq_lengths = [512, 2048, 8192, 32768, 131072]
    batch_sizes = [1, 4, 16, 32]

    col_label = "seq_len \\ batch"
    header = f"{col_label:>15}"
    for bs in batch_sizes:
        header += f" {'bs='+str(bs):>10}"
    print(header)
    print("-" * (15 + 11 * len(batch_sizes)))

    for seq_len in seq_lengths:
        row = f"{seq_len:>15,}"
        for bs in batch_sizes:
            kv = estimate_kv_cache_memory(layers, kv_heads, head_dim, seq_len, 2, bs)
            row += f" {kv:>9.2f}GB"
        print(row)

    print()
    print("观察：")
    print("1. KV Cache 与 seq_len 和 batch_size 都线性相关")
    print("2. 131072 (128K) 上下文 + batch=32 → KV Cache ~128GB！远超模型本身")
    print("3. 这就是为什么长上下文推理需要大量显存")
    print("4. GQA（Grouped Query Attention）减少 KV 头数，有效降低 KV Cache")
    print("   Qwen2.5-7B 只有 4 个 KV 头（而非 28 个注意力头）→ KV Cache 缩小 7 倍")
    print()


def compare_inference_vs_training():
    """
    对比推理和训练的显存需求

    这是理解"为什么推理比训练便宜"的关键
    """
    print("=" * 70)
    print("推理 vs 训练显存需求对比")
    print("=" * 70)
    print()

    params_b = 7  # Qwen2.5-7B

    print(f"模型: Qwen2.5-{params_b}B")
    print()
    print("┌─────────────────┬──────────┬──────────┐")
    print("│     组成部分     │   推理   │   训练   │")
    print("├─────────────────┼──────────┼──────────┤")

    # 模型权重
    weights = estimate_model_weights_memory(params_b, "bf16")
    print(f"│ 模型权重 (BF16)  │ {weights:>6.1f}GB  │ {weights:>6.1f}GB  │")

    # 梯度（推理不需要）
    gradients = weights  # 与权重同大小
    print(f"│ 梯度            │   0.0GB  │ {gradients:>6.1f}GB  │")

    # 优化器状态（Adam: 2 × 参数量的 FP32）
    optimizer = estimate_model_weights_memory(params_b, "fp32") * 2  # momentum + variance
    print(f"│ 优化器状态       │   0.0GB  │ {optimizer:>6.1f}GB  │")

    # KV Cache / 激活值
    kv_cache = estimate_kv_cache_memory(28, 4, 128, 2048, 2, 1)
    # 训练时激活值远大于推理（需要保存每层的激活用于反向传播）
    # 粗略估计：训练激活值 ≈ 模型权重 × 2~3
    train_activation = weights * 2.5
    print(f"│ KV Cache/激活值  │ {kv_cache:>6.2f}GB  │ {train_activation:>6.1f}GB  │")

    inf_total = weights + kv_cache
    train_total = weights + gradients + optimizer + train_activation
    print("├─────────────────┼──────────┼──────────┤")
    print(f"│ 总计(估)         │ {inf_total:>6.1f}GB  │ {train_total:>6.1f}GB  │")
    print("└─────────────────┴──────────┴──────────┘")

    ratio = train_total / inf_total
    print()
    print(f"训练显存约为推理的 {ratio:.0f} 倍！")
    print()
    print("这就是为什么：")
    print("  - 7B 模型推理只需 ~15GB（单卡 3090 可跑）")
    print("  - 7B 模型训练需要 ~60GB+（需要多卡 + 显存优化技术）")
    print("  - LoRA/QLoRA 等微调方法通过冻结主模型大幅减少训练显存")
    print()


# ============================================================
# 实测验证部分（需要 GPU）
# ============================================================

def verify_estimation_with_measurement():
    """
    用实际测量验证理论估算

    加载 Qwen2.5-7B 的不同精度版本，对比理论值与实测值
    """
    print("=" * 70)
    print("验证：理论估算 vs 实际测量")
    print("=" * 70)
    print()

    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    results = []

    # --- BF16 ---
    print("--- BF16 加载 ---")
    clear_gpu()

    # 理论值
    bf16_theory = estimate_model_weights_memory(7, "bf16")

    # 实测
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    bf16_actual = torch.cuda.memory_allocated(0) / 1024**3

    print(f"  理论估算: {bf16_theory:.2f}GB")
    print(f"  实际测量: {bf16_actual:.2f}GB")
    print(f"  差异:     {(bf16_actual - bf16_theory):.2f}GB ({(bf16_actual/bf16_theory - 1)*100:.1f}%)")

    results.append(("BF16", bf16_theory, bf16_actual))
    del model
    clear_gpu()

    # --- 4bit ---
    print()
    print("--- 4bit 量化加载 ---")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )
    quant4_actual = torch.cuda.memory_allocated(0) / 1024**3
    quant4_theory = estimate_model_weights_memory(7, "int4")

    print(f"  理论估算: {quant4_theory:.2f}GB")
    print(f"  实际测量: {quant4_actual:.2f}GB")
    print(f"  差异:     {(quant4_actual - quant4_theory):.2f}GB ({(quant4_actual/quant4_theory - 1)*100:.1f}%)")

    results.append(("4bit NF4", quant4_theory, quant4_actual))
    del model
    clear_gpu()

    # 汇总
    print()
    print("--- 汇总 ---")
    print(f"{'精度':<10} {'理论':>8} {'实测':>8} {'差异%':>8}")
    print("-" * 38)
    for name, theory, actual in results:
        diff_pct = (actual / theory - 1) * 100
        print(f"{name:<10} {theory:>7.2f}GB {actual:>7.2f}GB {diff_pct:>+7.1f}%")

    print()
    print("注意：实测值通常略高于理论值，因为：")
    print("  1. 框架开销（CUDA 上下文、临时缓冲区等）")
    print("  2. 量化配置中的查找表和缩放因子")
    print("  3. 部分层无法量化（如 LayerNorm）保持高精度")
    print("  4. 显存碎片和对齐开销")
    print()
    print("经验法则：实际显存 ≈ 理论估算 × 1.1~1.2（预留 10-20% 余量）")


def quick_reality_check():
    """
    快速检查：你的 4×3090 能跑什么？

    一个实用的部署前决策工具
    """
    print("=" * 70)
    print("实战速查：4× RTX 3090 (96GB) 能跑什么模型？")
    print("=" * 70)
    print()

    total_vram = 96  # 4 × 24GB
    # 实际可用约 90%（系统开销）
    usable_vram = total_vram * 0.9

    models = {
        "Qwen2.5-7B BF16":  (7, "bf16", 28, 4, 128, 1),
        "Qwen2.5-7B INT4":  (7, "int4", 28, 4, 128, 1),
        "Qwen2.5-32B BF16": (32, "bf16", 64, 8, 128, 2),
        "Qwen2.5-32B INT4": (32, "int4", 64, 8, 128, 2),
        "Qwen2.5-72B BF16": (72, "bf16", 80, 8, 128, 4),
        "Qwen2.5-72B INT4": (72, "int4", 80, 8, 128, 4),
    }

    seq_len = 4096  # 假设 4K 上下文
    batch_size = 1

    print(f"假设: seq_len={seq_len}, batch_size={batch_size}")
    print(f"可用显存: {usable_vram:.0f}GB (总 {total_vram}GB × 90%)")
    print()
    print(f"{'模型':<22} {'权重':>8} {'KV Cache':>10} {'总计':>8} {'状态':>6}")
    print("-" * 58)

    for name, (params, prec, layers, kv_heads, head_dim, min_gpus) in models.items():
        weights = estimate_model_weights_memory(params, prec)
        kv_cache = estimate_kv_cache_memory(layers, kv_heads, head_dim, seq_len, 2, batch_size)
        total = weights + kv_cache
        can_run = "✅" if total <= usable_vram else "❌"
        print(f"{name:<22} {weights:>7.1f}GB {kv_cache:>8.2f}GB {total:>7.1f}GB   {can_run}")

    print()
    print("结论：")
    print("  - 7B BF16: 单卡即跑，最适合开发调试")
    print("  - 32B INT4: 2卡 TP，质量与性能的平衡点")
    print("  - 72B INT4: 4卡 TP，最强质量但吞吐受限")
    print("  - 72B BF16: 超出显存，需要更大 GPU 或更多卡")


if __name__ == "__main__":
    print("大模型部署学习 - Stage 1: 显存估算实操")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print()

    # === 理论计算（不需要 GPU）===
    print("\n" + "▼" * 40)
    print("第一部分：理论计算")
    print("▼" * 40 + "\n")

    # 1. 不同模型/精度的显存估算
    table_model_memory_estimates()

    # 2. KV Cache 随序列长度和 batch size 的变化
    table_kv_cache_vs_seq_len()

    # 3. 推理 vs 训练显存对比
    compare_inference_vs_training()

    # 4. 快速实战速查
    quick_reality_check()

    # === 实测验证（需要 GPU）===
    print("\n" + "▼" * 40)
    print("第二部分：实测验证")
    print("▼" * 40 + "\n")

    # 5. 对比理论估算与实际测量
    verify_estimation_with_measurement()

    print()
    print("=" * 70)
    print("Stage 1 显存估算要点总结")
    print("=" * 70)
    print("1. 模型权重显存 = 参数量 × 每参数字节数（BF16=2, INT4=0.5）")
    print("2. KV Cache 显存 = 2 × 层数 × seq_len × KV头数 × 头维度 × 字节/参数 × batch")
    print("3. KV Cache 与序列长度和并发数线性相关，长上下文时可能超过模型本身")
    print("4. 推理显存 ≈ 训练的 1/4~1/5（无梯度、无优化器状态）")
    print("5. 实际显存比理论高 10-20%，部署时需预留余量")
    print("6. GQA 减少 KV 头数，有效降低 KV Cache 显存")

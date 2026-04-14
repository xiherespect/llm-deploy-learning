"""
Stage 1: HuggingFace Transformers 本地推理

学习目标：
1. 加载模型的不同方式（BF16 vs 4bit量化）
2. device_map 多卡分配
3. 对比不同精度的显存占用和推理速度
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


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


def inference_bf16():
    """
    方式1: BF16 精度加载（全精度推理）

    7B 模型 BF16 约 14GB，单张 3090 (24GB) 可以跑
    """
    print("=" * 60)
    print("方式1: BF16 精度推理")
    print("=" * 60)

    clear_gpu()
    print_gpu_memory("加载前")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载模型 - BF16 精度
    # device_map="auto" 会自动将模型分配到可用 GPU 上
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,       # 使用 BF16 精度
        device_map="auto",                 # 自动分配到 GPU
        # device_map="cuda:0",            # 指定单卡（7B模型单卡可跑）
    )

    print_gpu_memory("加载后")

    # 推理
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请用三句话介绍量子计算。"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 计时
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
    elapsed = time.time() - start

    # 只解码新生成的 token
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"回复: {response}")
    print(f"推理耗时: {elapsed:.2f}s")
    print_gpu_memory("推理后")

    # 清理
    del model
    del tokenizer
    clear_gpu()

    return elapsed


def inference_4bit():
    """
    方式2: 4bit 量化加载（BitsAndBytes NF4 量化）

    7B 模型 4bit 约 3.5GB，大幅减少显存占用
    适合在显存有限时运行更大模型
    """
    print("=" * 60)
    print("方式2: 4bit 量化推理 (BitsAndBytes NF4)")
    print("=" * 60)

    clear_gpu()
    print_gpu_memory("加载前")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 配置 4bit 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # 启用 4bit 量化
        bnb_4bit_quant_type="nf4",               # NF4 量化（比普通 INT4 精度更高）
        bnb_4bit_compute_dtype=torch.bfloat16,   # 计算时用 BF16
        bnb_4bit_use_double_quant=True,           # 双量化（进一步节省显存）
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    print_gpu_memory("加载后")

    # 同样的推理
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请用三句话介绍量子计算。"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
    elapsed = time.time() - start

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"回复: {response}")
    print(f"推理耗时: {elapsed:.2f}s")
    print_gpu_memory("推理后")

    del model
    del tokenizer
    clear_gpu()

    return elapsed


def demo_device_map():
    """
    演示 device_map 的工作方式

    - "auto": accelerate 库自动按层分配到多张 GPU
    - "cuda:0": 全部放到第一张 GPU
    - 自定义 dict: 手动指定每层放到哪张 GPU
    """
    print("=" * 60)
    print("演示: device_map 多卡分配")
    print("=" * 60)

    # 查看自动分配方案（不实际加载模型）
    from accelerate import infer_auto_device_map, init_empty_weights
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
    device_map = infer_auto_device_map(
        empty_model,
        no_split_module_classes=["Qwen2DecoderLayer"],
        dtype=torch.bfloat16,
    )

    print("auto device_map 分配方案:")
    for key, device in device_map.items():
        print(f"  {key}: {device}")
    print()

    del empty_model
    clear_gpu()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 1: Transformers 本地推理")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print()

    # 演示 device_map
    demo_device_map()

    # BF16 推理
    bf16_time = inference_bf16()

    # 4bit 推理
    quant_time = inference_4bit()

    # 对比结果
    print("=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"BF16 推理耗时: {bf16_time:.2f}s")
    print(f"4bit 推理耗时: {quant_time:.2f}s")
    print(f"4bit 相对 BF16: {quant_time/bf16_time:.2f}x")
    print()
    print("要点总结:")
    print("1. BF16 精度最高，显存占用最大，推理速度适中")
    print("2. 4bit 量化显存约 1/4，推理速度可能慢一些（反量化开销）")
    print("3. device_map='auto' 自动分配模型层到多 GPU")
    print("4. 单卡 3090 (24GB) 可以跑 7B BF16，但 32B+ 需要量化或多卡")

"""
Stage 2: vLLM OpenAI 兼容 API 服务

学习目标：
1. 掌握 vLLM API 服务的启动命令和参数
2. 用 requests 和 openai SDK 调用 API
3. 实现流式输出（SSE）
4. 多轮对话 API 调用
5. 用 Python 封装可复用的客户端

前置条件：
  先启动 vLLM API 服务：
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-7B-Instruct \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.9 \
      --max-model-len 4096 \
      --dtype bfloat16 \
      --port 8000

核心概念：
- vLLM 提供与 OpenAI API 兼容的接口，方便迁移和集成
- 主要端点：/v1/chat/completions, /v1/completions, /v1/models
- 流式输出通过 SSE (Server-Sent Events) 实现
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import json
import time
import requests

# vLLM API 服务地址
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def check_server_status():
    """
    检查 vLLM API 服务是否运行
    """
    print("=" * 60)
    print("检查 vLLM API 服务状态")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_BASE}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json()
            print("服务运行中！可用模型：")
            for model in models.get("data", []):
                print(f"  - {model['id']}")
        else:
            print(f"服务响应异常: {resp.status_code}")
    except requests.ConnectionError:
        print("无法连接到 vLLM 服务！")
        print()
        print("请先启动服务：")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen2.5-7B-Instruct \\")
        print("      --tensor-parallel-size 1 \\")
        print("      --port 8000")
        return False

    print()
    return True


def basic_chat_completion():
    """
    基本的 chat/completions 调用

    这是 OpenAI 兼容 API 的核心端点
    """
    print("=" * 60)
    print("基本 chat/completions 调用")
    print("=" * 60)

    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "请用三句话介绍量子计算。"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }

    start = time.time()
    resp = requests.post(url, json=payload)
    elapsed = time.time() - start

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]

        print(f"回复: {content}")
        print()
        print(f"Token 用量: prompt={usage['prompt_tokens']}, "
              f"completion={usage['completion_tokens']}, "
              f"total={usage['total_tokens']}")
        print(f"耗时: {elapsed:.2f}s")
    else:
        print(f"请求失败: {resp.status_code}")
        print(resp.text)

    print()


def basic_completion():
    """
    completions 端点（原始文本补全）

    chat/completions 使用对话格式，completions 使用原始文本
    """
    print("=" * 60)
    print("completions 端点（原始文本补全）")
    print("=" * 60)

    url = f"{API_BASE}/completions"
    payload = {
        "model": MODEL_NAME,
        "prompt": "量子计算是一种利用量子力学原理进行计算的技术，",
        "temperature": 0.7,
        "max_tokens": 128,
    }

    resp = requests.post(url, json=payload)

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["text"]
        print(f"补全结果: {content}")
    else:
        print(f"请求失败: {resp.status_code}")
        print(resp.text)

    print()
    print("注意：completions 端点直接补全文本，不使用对话格式。")
    print("chat/completions 更常用，因为它支持 system prompt 和多轮对话。")
    print()


def streaming_chat():
    """
    流式输出（SSE）演示

    流式输出让用户更快看到响应，而不是等全部生成完
    """
    print("=" * 60)
    print("流式输出（SSE）演示")
    print("=" * 60)

    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "写一首关于春天的短诗。"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,  # 启用流式输出
    }

    start = time.time()
    resp = requests.post(url, json=payload, stream=True)

    print("流式输出：")
    full_content = ""
    for line in resp.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]  # 去掉 "data: " 前缀
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    full_content += content
                    print(content, end="", flush=True)

    elapsed = time.time() - start
    print()
    print()
    print(f"完整输出（{len(full_content)} 字符），耗时: {elapsed:.2f}s")
    print()
    print("流式输出的优势：")
    print("  1. 用户可以立即看到生成内容，感知延迟更低")
    print("  2. 适合长文本生成场景（如文章、代码）")
    print("  3. 可以在生成过程中取消请求，节省资源")
    print()


def multi_turn_conversation():
    """
    多轮对话 API 调用

    通过 messages 数组维护对话历史
    """
    print("=" * 60)
    print("多轮对话 API 调用")
    print("=" * 60)

    url = f"{API_BASE}/chat/completions"

    # 对话历史
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手，擅长解释技术概念。"},
    ]

    # 模拟三轮对话
    user_inputs = [
        "什么是 Transformer 架构？",
        "它的自注意力机制是怎么工作的？",
        "和 RNN 相比有什么优势？",
    ]

    for user_input in user_inputs:
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 256,
        }

        resp = requests.post(url, json=payload)

        if resp.status_code == 200:
            result = resp.json()
            assistant_content = result["choices"][0]["message"]["content"]
            # 添加助手回复到对话历史
            messages.append({"role": "assistant", "content": assistant_content})

            print(f"用户: {user_input}")
            print(f"助手: {assistant_content[:150]}...")
            print()
        else:
            print(f"请求失败: {resp.status_code}")
            print(resp.text)
            break

    print(f"对话历史共 {len(messages)} 条消息")
    print()
    print("注意：vLLM 是无状态的，每次请求都发送完整对话历史。")
    print("对话状态管理需要在客户端（你的应用）中维护。")
    print()


def openai_sdk_demo():
    """
    使用 openai SDK 调用 vLLM

    直接复用 OpenAI 的 Python SDK，只需改 base_url
    """
    print("=" * 60)
    print("使用 openai SDK 调用 vLLM")
    print("=" * 60)

    try:
        from openai import OpenAI
    except ImportError:
        print("openai SDK 未安装，请运行: pip install openai")
        print()
        return

    # 创建客户端，指向本地 vLLM 服务
    client = OpenAI(
        base_url=f"{API_BASE}",
        api_key="not-needed",  # vLLM 默认不需要 API key
    )

    # 非流式调用
    print("--- 非流式调用 ---")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "请用一句话解释什么是深度学习。"},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    print(f"回复: {response.choices[0].message.content}")
    print(f"Token 用量: {response.usage}")
    print()

    # 流式调用
    print("--- 流式调用 ---")
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "写一个 Python 的快速排序函数。"},
        ],
        temperature=0.3,
        max_tokens=256,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    print()

    print("openai SDK 的优势：")
    print("  1. 只需改 base_url，无需修改业务代码")
    print("  2. 类型提示完善，开发体验好")
    print("  3. 从 OpenAI 迁移到自部署 vLLM 几乎零成本")
    print()


class VLLMClient:
    """
    简单的 vLLM API 客户端封装

    封装常用操作，方便在其他项目中复用
    """

    def __init__(self, base_url="http://localhost:8000/v1", model=None):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()

    def chat(self, messages, temperature=0.7, max_tokens=256, stream=False):
        """发送 chat completion 请求"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        resp = self.session.post(url, json=payload, stream=stream)

        if stream:
            return self._handle_stream(resp)

        if resp.status_code == 200:
            result = resp.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "usage": result["usage"],
                "finish_reason": result["choices"][0]["finish_reason"],
            }
        else:
            raise Exception(f"API 请求失败: {resp.status_code} {resp.text}")

    def _handle_stream(self, resp):
        """处理流式响应"""
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    def list_models(self):
        """列出可用模型"""
        resp = self.session.get(f"{self.base_url}/models")
        if resp.status_code == 200:
            return [m["id"] for m in resp.json().get("data", [])]
        return []

    def simple_ask(self, prompt, system="你是一个有帮助的AI助手。", stream=False):
        """简单问答（单轮对话的快捷方法）"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, stream=stream)


def demo_custom_client():
    """
    演示自定义客户端的使用
    """
    print("=" * 60)
    print("自定义 VLLMClient 客户端")
    print("=" * 60)

    client = VLLMClient(
        base_url="http://localhost:8000/v1",
        model=MODEL_NAME,
    )

    # 检查可用模型
    models = client.list_models()
    print(f"可用模型: {models}")
    print()

    # 简单问答
    result = client.simple_ask("什么是机器学习？请简短回答。")
    print(f"回答: {result['content']}")
    print(f"Token 用量: {result['usage']}")
    print()

    # 多轮对话
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是 Python 的列表推导式？"},
    ]
    result = client.chat(messages, max_tokens=128)
    print(f"回答: {result['content'][:100]}...")

    # 把助手回复加入对话历史
    messages.append({"role": "assistant", "content": result["content"]})
    messages.append({"role": "user", "content": "给一个具体的例子。"})

    result = client.chat(messages, max_tokens=128)
    print(f"追问回答: {result['content'][:100]}...")
    print()

    # 流式输出
    print("--- 流式输出 ---")
    for chunk in client.simple_ask("写一句关于编程的名言", stream=True):
        print(chunk, end="", flush=True)
    print()
    print()


def show_startup_commands():
    """
    展示常用的 vLLM API 服务启动命令
    """
    print("=" * 60)
    print("vLLM API 服务常用启动命令")
    print("=" * 60)
    print()
    print("1. 基本启动（单卡）：")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model Qwen/Qwen2.5-7B-Instruct \\")
    print("       --port 8000")
    print()
    print("2. 多卡 TP 启动：")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model Qwen/Qwen2.5-7B-Instruct \\")
    print("       --tensor-parallel-size 2 \\")
    print("       --port 8000")
    print()
    print("3. 性能调优启动：")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model Qwen/Qwen2.5-7B-Instruct \\")
    print("       --gpu-memory-utilization 0.95 \\")
    print("       --max-model-len 4096 \\")
    print("       --dtype bfloat16 \\")
    print("       --port 8000")
    print()
    print("4. 调试模式（关闭 CUDA Graph）：")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model Qwen/Qwen2.5-7B-Instruct \\")
    print("       --enforce-eager \\")
    print("       --port 8000")
    print()
    print("5. 量化模型启动：")
    print("   python -m vllm.entrypoints.openai.api_server \\")
    print("       --model Qwen/Qwen2.5-7B-Instruct-AWQ \\")
    print("       --quantization awq \\")
    print("       --port 8000")
    print()
    print("curl 测试命令：")
    print('   curl http://localhost:8000/v1/chat/completions \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "你好"}]}\'')
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 2: vLLM OpenAI 兼容 API 服务")
    print()
    print("⚠️  运行此脚本前，请先启动 vLLM API 服务！")
    print("   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000")
    print()

    # 0. 启动命令参考（不需要服务运行）
    show_startup_commands()

    # 以下功能需要 vLLM 服务运行
    if check_server_status():
        # 1. 基本 chat/completions
        # basic_chat_completion()

        # 2. completions 端点
        # basic_completion()

        
        # 3. 流式输出
        # streaming_chat()

        # 4. 多轮对话
        # multi_turn_conversation()

        # 5. openai SDK
        # openai_sdk_demo()

        # 6. 自定义客户端
        demo_custom_client()

    print("=" * 60)
    print("Stage 2 API 服务要点总结")
    print("=" * 60)
    print("1. vLLM 提供 OpenAI 兼容 API，迁移成本几乎为零")
    print("2. 核心端点：/v1/chat/completions（对话）、/v1/completions（补全）")
    print("3. 流式输出（stream=True）降低用户感知延迟")
    print("4. 多轮对话状态在客户端维护，服务端无状态")
    print("5. 可用 openai SDK 或自行封装客户端")
    print("6. 启动参数直接影响服务能力和性能 → 下一脚本详解 TP")

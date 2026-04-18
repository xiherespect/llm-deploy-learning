"""
Stage 5: Nginx 反向代理与负载均衡

学习目标：
1. 理解反向代理的概念和作用
2. 掌握 Nginx 反向代理 vLLM 的配置
3. 理解 SSE 流式输出与 Nginx 的兼容性配置
4. 学会配置负载均衡（多 vLLM 实例）
5. 掌握 Nginx 基础限流配置

核心概念：
- 反向代理：客户端请求先到 Nginx，Nginx 转发到后端 vLLM
- SSE 流式输出：vLLM 用 Server-Sent Events 逐 token 推送
- proxy_buffering off：Nginx 默认缓冲响应，SSE 必须关闭
- least_conn：最少连接负载均衡，适合 LLM 变长请求

前置条件：
- vLLM Docker 服务已启动（02 脚本的 docker-compose）
- Nginx 容器已启动

配置文件位置：
- docker/nginx/nginx.conf
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import json
import time
import requests

API_BASE = "http://localhost:80/v1"   # 通过 Nginx 访问
VLLM_DIRECT = "http://localhost:8000/v1"  # 直接访问 vLLM


def explain_reverse_proxy():
    """
    反向代理概念与作用
    """
    print("=" * 60)
    print("反向代理：为什么 vLLM 前面要加 Nginx？")
    print("=" * 60)
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  直接暴露 vLLM                                       │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  Client ──→ vLLM:8000                               │")
    print("│  问题：                                              │")
    print("│  - 没有限流，恶意请求可以打满 GPU                     │")
    print("│  - 没有负载均衡，只能用一个 vLLM 实例                 │")
    print("│  - vLLM 暴露 /metrics 等内部端点给外部               │")
    print("│  - 没有 SSL/TLS，明文传输                            │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  Nginx 反向代理                                      │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  Client ──→ Nginx:80 ──→ vLLM:8000                  │")
    print("│  优势：                                              │")
    print("│  - 限流：保护 vLLM 不被过多请求压垮                  │")
    print("│  - 负载均衡：多个 vLLM 实例分摊请求                  │")
    print("│  - 安全：隐藏 vLLM，控制暴露的端点                   │")
    print("│  - SSL 终止：Nginx 处理 HTTPS，vLLM 只需 HTTP        │")
    print("│  - 日志：统一的访问日志和错误日志                     │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("反向代理 vs 正向代理：")
    print("  正向代理：代理客户端（如 VPN），服务端不知道真实客户端")
    print("  反向代理：代理服务端（如 Nginx），客户端不知道真实服务端")
    print("  LLM 部署用的是反向代理")
    print()


def explain_nginx_config():
    """
    nginx.conf 逐行讲解
    """
    print("=" * 60)
    print("nginx.conf 关键配置详解")
    print("=" * 60)
    print()
    print("完整配置见 docker/nginx/nginx.conf，以下逐段讲解：")
    print()
    print("--- 1. 全局 SSE 兼容配置（最关键！）---")
    print()
    print("http {")
    print("    proxy_buffering off;    # ⚠️ 关闭响应缓冲")
    print("    proxy_cache off;        # 关闭响应缓存")
    print()
    print("为什么 proxy_buffering off 是 LLM 部署的生命线？")
    print()
    print("  默认行为（proxy_buffering on）：")
    print("    vLLM 发送 [token1][token2]...[tokenN]")
    print("    → Nginx 缓冲全部内容")
    print("    → 等响应结束后才一次性发给客户端")
    print("    → 用户看不到'逐字输出'效果，等很久才一次性看到")
    print()
    print("  关闭缓冲后（proxy_buffering off）：")
    print("    vLLM 发送 [token1] → Nginx 立即转发 → 用户看到 token1")
    print("    vLLM 发送 [token2] → Nginx 立即转发 → 用户看到 token2")
    print("    → 逐 token 实时推送给用户")
    print()
    print("--- 2. 超时配置 ---")
    print()
    print("    proxy_connect_timeout 60s;   # 连接后端超时")
    print("    proxy_read_timeout 300s;     # ⚠️ 读取后端响应超时")
    print("    proxy_send_timeout 60s;      # 发送请求到后端超时")
    print()
    print("为什么 proxy_read_timeout 要设 300s？")
    print("  - LLM 生成长文本可能需要几十秒甚至几分钟")
    print("  - Nginx 默认 60s 超时，长文本生成会被中断")
    print("  - 设 300s 给足时间，但也要注意不要无限大")
    print("  - 流式输出时，每个 token 的到达会重置超时计时")
    print()
    print("--- 3. 负载均衡 upstream ---")
    print()
    print("    upstream vllm_backend {")
    print("        least_conn;           # 最少连接数策略")
    print("        server vllm:8000;     # vLLM 实例1")
    print("        # server vllm-2:8000;  # vLLM 实例2（多实例时）")
    print("    }")
    print()
    print("为什么用 least_conn 而不是默认的 round-robin？")
    print()
    print("  round-robin（默认）：轮流分配")
    print("    请求1(短)→实例A, 请求2(长)→实例B, 请求3(短)→实例A")
    print("    实例A 快速完成，实例B 还在慢慢生成 → 负载不均")
    print()
    print("  least_conn：分配给当前连接数最少的实例")
    print("    新请求总是发给最空闲的实例 → 负载更均衡")
    print("    LLM 请求耗时长短差异大，least_conn 更合适")
    print()
    print("--- 4. location 路由规则 ---")
    print()
    print("    location / {")
    print("        limit_req zone=api_limit burst=20 nodelay;")
    print("        proxy_pass http://vllm_backend;")
    print("        proxy_set_header Host $host;")
    print("        proxy_set_header X-Real-IP $remote_addr;")
    print()
    print("        # SSE 兼容性")
    print("        proxy_set_header Connection '';")
    print("        proxy_http_version 1.1;")
    print("        chunked_transfer_encoding on;")
    print("        add_header X-Accel-Buffering no;")
    print("    }")
    print()
    print("  proxy_http_version 1.1：")
    print("    Nginx 默认用 HTTP/1.0 代理到后端，1.0 不支持 keepalive")
    print("    设为 1.1 保持长连接，减少连接建立开销")
    print()
    print("  X-Accel-Buffering: no：")
    print("    从响应头层面再次确保 Nginx 不缓冲（双重保险）")
    print()


def demo_sse_streaming():
    """
    测试 SSE 流式输出是否正常通过 Nginx
    """
    print("=" * 60)
    print("测试 SSE 流式输出（通过 Nginx）")
    print("=" * 60)
    print()

    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "messages": [
            {"role": "user", "content": "写一首关于春天的短诗。"}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": True,
    }

    print("通过 Nginx 流式请求：")
    try:
        start = time.time()
        resp = requests.post(url, json=payload, stream=True, timeout=60)
        first_token_time = None
        full_content = ""

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
                        if first_token_time is None:
                            first_token_time = time.time()
                        full_content += delta["content"]
                        print(delta["content"], end="", flush=True)

        elapsed = time.time() - start
        print()
        print()
        if first_token_time:
            ttft = first_token_time - start
            print(f"首 token 延迟(TTFT): {ttft:.2f}s")
        print(f"总耗时: {elapsed:.2f}s")
        print(f"生成内容: {len(full_content)} 字符")
        print()
        print("✓ 流式输出正常！Nginx 没有缓冲响应。")
    except requests.ConnectionError:
        print("无法连接到 Nginx，请确保服务已启动：")
        print("  docker compose up -d")
    except Exception as e:
        print(f"请求失败: {e}")
    print()


def demo_rate_limiting():
    """
    Nginx 限流配置讲解
    """
    print("=" * 60)
    print("Nginx 限流配置")
    print("=" * 60)
    print()
    print("nginx.conf 中的限流配置：")
    print()
    print("  # 定义限流区域：按 IP 限流，每秒 10 个请求")
    print("  limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;")
    print()
    print("  # 在 location 中应用限流")
    print("  limit_req zone=api_limit burst=20 nodelay;")
    print()
    print("参数解析：")
    print()
    print("  rate=10r/s")
    print("    平均每秒最多 10 个请求")
    print("    这是令牌桶的填充速率")
    print()
    print("  burst=20")
    print("    允许突发 20 个请求（桶的容量）")
    print("    短时间内可以超过 10r/s，但最多额外 20 个")
    print()
    print("  nodelay")
    print("    突发请求不延迟处理，超出直接返回 503")
    print("    不加 nodelay：超出速率的请求排队等待")
    print()
    print("限流区域大小（10m）：")
    print("  每个 IP 占 ~64 字节，10MB 可记录约 16 万个 IP")
    print("  一般够用，不够可以增大")
    print()
    print("LLM 服务的限流策略：")
    print()
    print("  场景1：内部服务（少量已知客户端）")
    print("    → rate=100r/s，burst=50，宽松限流")
    print()
    print("  场景2：公开 API（大量未知客户端）")
    print("    → rate=5r/s，burst=10，严格限流")
    print("    → 配合 API Key 认证，按用户级别限流")
    print()
    print("  场景3：批量处理（高吞吐，可排队）")
    print("    → rate=50r/s，burst=200，不加 nodelay")
    print("    → 允许排队等待，不直接拒绝")
    print()
    print("注意：/health 端点不应限流，否则健康检查被拦截")
    print("  nginx.conf 中 /health 单独配置，不走 limit_req")
    print()


def demo_load_balancing():
    """
    负载均衡配置讲解
    """
    print("=" * 60)
    print("负载均衡：多 vLLM 实例部署")
    print("=" * 60)
    print()
    print("单实例瓶颈：")
    print("  - 1 个 vLLM 实例的最大并发受 GPU 显存限制")
    print("  - 7B 模型单实例可能支持 ~50 并发")
    print("  - 超过并发上限 → 请求排队 → 延迟暴增")
    print()
    print("多实例方案：")
    print()
    print("  ┌───────────┐")
    print("  │   Client   │")
    print("  └─────┬─────┘")
    print("        │ :80")
    print("  ┌─────▼─────┐")
    print("  │   Nginx    │ least_conn 负载均衡")
    print("  └──┬─────┬──┘")
    print("     │     │")
    print("  ┌──▼──┐ ┌▼───┐")
    print("  │vLLM1│ │vLLM2│")
    print("  │GPU 0│ │GPU 2│")
    print("  │GPU 1│ │GPU 3│")
    print("  └─────┘ └────┘")
    print()
    print("Nginx upstream 配置：")
    print()
    print("  upstream vllm_backend {")
    print("      least_conn;")
    print("      server vllm-1:8000;")
    print("      server vllm-2:8000;")
    print("  }")
    print()
    print("负载均衡策略对比：")
    print()
    print("  ┌─────────────┬────────────────────────────────────────┐")
    print("  │  策略        │  特点                                   │")
    print("  ├─────────────┼────────────────────────────────────────┤")
    print("  │  round-robin │  轮流分配，简单但不考虑负载差异         │")
    print("  │  least_conn  │  分配给最少连接的实例，✓ LLM 推荐       │")
    print("  │  ip_hash     │  同一 IP 总是分配到同一实例（会话保持）  │")
    print("  │  random      │  随机分配                               │")
    print("  └─────────────┴────────────────────────────────────────┘")
    print()
    print("多实例 docker-compose 配置见 05_advanced_deployment.py")
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 5: Nginx 反向代理与负载均衡")
    print()

    # 1. 概念讲解
    # explain_reverse_proxy()
    # explain_nginx_config()

    # # 2. 限流配置
    # demo_rate_limiting()

    # # 3. 负载均衡
    # demo_load_balancing()

    # 4. 测试 SSE 流式输出（需要 Nginx 和 vLLM 运行）
    demo_sse_streaming()

    print("=" * 60)
    print("Stage 5 Nginx 代理要点总结")
    print("=" * 60)
    print("1. Nginx 反向代理提供限流、负载均衡、安全隔离")
    print("2. proxy_buffering off 是 SSE 流式输出的关键配置")
    print("3. proxy_read_timeout 300s 适应 LLM 长文本生成")
    print("4. least_conn 负载均衡最适合 LLM 变长请求场景")
    print("5. limit_req_zone 按 IP 限流，/health 不限流")
    print("6. 多实例部署时 Nginx 是请求分发的核心组件")

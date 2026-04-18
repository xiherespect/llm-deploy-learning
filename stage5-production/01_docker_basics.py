"""
Stage 5: Docker 基础与 vLLM 容器化部署

学习目标：
1. 理解为什么生产部署需要 Docker 容器化
2. 掌握 Docker 核心概念：镜像、容器、卷、网络
3. 了解 vLLM 官方 Docker 镜像的组成
4. 理解 GPU 直通（nvidia-container-toolkit）的原理
5. 学会用 docker run 启动 vLLM 服务

核心概念：
- Docker 容器化：将应用及其依赖打包为可移植的运行单元
- GPU 直通：通过 nvidia-container-toolkit 让容器访问宿主机 GPU
- 卷挂载：将宿主机目录映射到容器内，避免重复下载模型
- --ipc=host：PyTorch 多进程通信需要足够大的共享内存

前置条件：
- Docker 已安装（本机版本：28.5.1）
- nvidia-container-toolkit 已安装
- 模型已下载到 HF_HOME 缓存目录
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_HOME = os.environ.get("HF_HOME", "")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def explain_why_docker():
    """
    为什么 LLM 部署需要 Docker？
    """
    print("=" * 60)
    print("为什么 LLM 部署需要 Docker？")
    print("=" * 60)
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  裸机部署（无 Docker）                                │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  宿主机 Python 3.10 + vLLM 0.19.0 + CUDA 12.2       │")
    print("│  → 换一台机器：Python 3.11, CUDA 12.4 → 可能跑不了  │")
    print("│  → 新同事搭环境：装半天依赖，版本冲突                 │")
    print("│  → 扩容：手动在新机器重复配置                         │")
    print("│  → 回滚：不知道之前什么版本能跑                       │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  Docker 容器化部署                                    │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  镜像 = 应用 + 全部依赖 + 运行环境                    │")
    print("│  → 换机器：docker pull，秒级启动                      │")
    print("│  → 新同事：docker compose up，一键启动                │")
    print("│  → 扩容：docker compose up --scale vllm=2            │")
    print("│  → 回滚：docker run vllm:v0.18.0                     │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("Docker 对 LLM 部署的核心价值：")
    print("  1. 环境一致性：CUDA/cuDNN/vLLM 版本锁定在镜像中")
    print("  2. 快速部署：docker compose up 一条命令启动全栈")
    print("  3. 易于扩缩容：多实例部署只需修改配置")
    print("  4. 隔离性：多个模型实例互不干扰")
    print("  5. 可复现：镜像版本 = 部署版本，随时回滚")
    print()


def explain_docker_concepts():
    """
    Docker 核心概念图解
    """
    print("=" * 60)
    print("Docker 核心概念")
    print("=" * 60)
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  镜像 (Image)                                        │")
    print("│  → 只读模板，包含应用+依赖+OS层                       │")
    print("│  → vllm/vllm-openai:v0.19.0 就是一个镜像             │")
    print("│  → 分层存储：共用层不重复下载                         │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  容器 (Container)                                     │")
    print("│  → 镜像的运行实例                                    │")
    print("│  → 有自己的文件系统、网络、进程空间                   │")
    print("│  → 轻量：共享宿主机内核，不需要完整OS                 │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  卷 (Volume)                                         │")
    print("│  → 容器的持久化存储                                   │")
    print("│  → bind mount: 宿主机目录 ↔ 容器目录                 │")
    print("│  → named volume: Docker 管理的持久化卷                │")
    print("│  → 例: -v /data2/hf_cache:/root/.cache/huggingface   │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  网络 (Network)                                       │")
    print("│  → 容器间通信的虚拟网络                               │")
    print("│  → 同一网络内可通过服务名访问（如 vllm:8000）         │")
    print("│  → 端口映射: -p 宿主端口:容器端口                     │")
    print("└─────────────────────────────────────────────────────┘")
    print()
    print("LLM 部署中的典型 Docker 数据流：")
    print()
    print("  宿主机 /data2/hf_cache/  ──mount──>  容器 /root/.cache/huggingface/")
    print("  宿主机 :80              ──map───>  容器 :8000 (vLLM API)")
    print("  宿主机 GPU 0,1          ──passthrough──>  容器内可见 GPU 0,1")
    print()


def explain_nvidia_container_toolkit():
    """
    GPU 直通原理
    """
    print("=" * 60)
    print("GPU 直通：nvidia-container-toolkit 原理")
    print("=" * 60)
    print()
    print("普通容器看不到 GPU，因为：")
    print("  - 容器有自己的文件系统命名空间")
    print("  - /dev/nvidia* 设备节点不会自动挂载")
    print("  - NVIDIA 驱动库不会自动注入")
    print()
    print("nvidia-container-toolkit 做了什么：")
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  docker run --gpus all vllm/vllm-openai          │")
    print("  │                                                    │")
    print("  │  1. 识别 --gpus 参数                               │")
    print("  │  2. 挂载 /dev/nvidia* 设备到容器                   │")
    print("  │  3. 注入 NVIDIA 驱动库（libcuda.so 等）            │")
    print("  │  4. 设置 NVIDIA_VISIBLE_DEVICES 环境变量           │")
    print("  │  5. 容器内 nvidia-smi / torch.cuda 可用            │")
    print("  └──────────────────────────────────────────────────┘")
    print()
    print("GPU 分配方式：")
    print()
    print("  # 所有 GPU")
    print("  docker run --gpus all ...")
    print()
    print("  # 指定 GPU 编号")
    print('  docker run --gpus \'"device=0,1"\' ...')
    print()
    print("  # 指定 GPU 数量")
    print('  docker run --gpus 2 ...')
    print()
    print("  # docker-compose 中使用（推荐）")
    print("  deploy:")
    print("    resources:")
    print("      reservations:")
    print("        devices:")
    print("          - driver: nvidia")
    print("            device_ids: ['0', '1']")
    print("            capabilities: [gpu]")
    print()
    print("注意：容器内看到的 GPU 编号从 0 开始，与 device_ids 无关")
    print("  device_ids=['2','3'] → 容器内看到的是 GPU 0, GPU 1")
    print()


def demo_docker_run_vllm():
    """
    完整的 docker run 命令逐行解析
    """
    print("=" * 60)
    print("docker run 启动 vLLM 服务：逐行解析")
    print("=" * 60)
    print()

    cmd = f"""docker run -d \\
  --name vllm-server \\
  --gpus '"device=0,1"' \\
  --ipc=host \\
  -v {HF_HOME}:/root/.cache/huggingface \\
  -p 8000:8000 \\
  -e HF_HOME=/root/.cache/huggingface \\
  vllm/vllm-openai:v0.19.0 \\
  --model {VLLM_MODEL} \\
  --tensor-parallel-size 2 \\
  --gpu-memory-utilization 0.9 \\
  --max-model-len 4096 \\
  --dtype bfloat16"""

    print(cmd)
    print()
    print("--- 参数解析 ---")
    print()
    print("docker run -d")
    print("  -d: 后台运行（detached 模式）")
    print()
    print("--name vllm-server")
    print("  给容器命名，方便后续 docker logs/stop/restart")
    print()
    print('--gpus \'"device=0,1"\'')
    print("  只使用 GPU 0 和 GPU 1（双卡 TP 推理）")
    print("  容器内 torch.cuda.device_count() == 2")
    print()
    print("--ipc=host")
    print("  ⚠️  关键参数！使用宿主机的进程间通信命名空间")
    print("  PyTorch 多 GPU 推理（TP）需要大量共享内存")
    print("  Docker 默认 shm 只有 64MB，不够用，会报错：")
    print("    RuntimeError: DataLoader worker ... exit code 1")
    print("  --ipc=host 等价于无限 shm，但安全性略低")
    print("  替代方案：--shm-size=8gb（更安全但需估算大小）")
    print()
    print(f"-v {HF_HOME}:/root/.cache/huggingface")
    print("  将宿主机的模型缓存目录挂载到容器内")
    print("  容器内 vLLM 读取缓存的模型，无需重新下载")
    print("  建议加 :ro（只读）防止容器修改宿主缓存")
    print()
    print("-p 8000:8000")
    print("  宿主机 8000 端口 → 容器 8000 端口")
    print("  访问 http://localhost:8000 即可调用 vLLM API")
    print()
    print("-e HF_HOME=/root/.cache/huggingface")
    print("  设置容器内的环境变量，确保 vLLM 找到模型缓存")
    print()
    print("vllm/vllm-openai:v0.19.0")
    print("  vLLM 官方镜像，已包含：Python, vLLM, CUDA, 系统库")
    print("  :v0.19.0 是固定版本标签（推荐），:latest 是最新版")
    print()
    print("--model / --tensor-parallel-size / ...")
    print("  这些是 vLLM 的启动参数，直接传给容器内的 vLLM")
    print()


def demo_docker_run_with_limits():
    """
    带资源限制的 docker run
    """
    print("=" * 60)
    print("Docker 资源限制与 GPU 分配")
    print("=" * 60)
    print()

    print("--- 场景1：7B 模型单卡推理 ---")
    print(f'docker run -d --name vllm-7b \\')
    print(f'  --gpus \'"device=0"\' \\')
    print(f'  --shm-size=4gb \\')
    print(f'  -v {HF_HOME}:/root/.cache/huggingface:ro \\')
    print(f'  -p 8000:8000 \\')
    print(f'  vllm/vllm-openai:v0.19.0 \\')
    print(f'  --model {VLLM_MODEL} \\')
    print(f'  --tensor-parallel-size 1 \\')
    print(f'  --gpu-memory-utilization 0.9 \\')
    print(f'  --max-model-len 4096')
    print()

    print("--- 场景2：32B AWQ 模型双卡推理 ---")
    print(f'docker run -d --name vllm-32b \\')
    print(f'  --gpus \'"device=0,1"\' \\')
    print(f'  --ipc=host \\')
    print(f'  -v {HF_HOME}:/root/.cache/huggingface:ro \\')
    print(f'  -p 8001:8000 \\')
    print(f'  vllm/vllm-openai:v0.19.0 \\')
    print(f'  --model Qwen/Qwen2.5-32B-Instruct-AWQ \\')
    print(f'  --quantization awq \\')
    print(f'  --tensor-parallel-size 2 \\')
    print(f'  --gpu-memory-utilization 0.9 \\')
    print(f'  --max-model-len 4096')
    print()

    print("--- 场景3：调试模式（关闭 CUDA Graph）---")
    print(f'docker run -d --name vllm-debug \\')
    print(f'  --gpus \'"device=0"\' \\')
    print(f'  --ipc=host \\')
    print(f'  -v {HF_HOME}:/root/.cache/huggingface:ro \\')
    print(f'  -p 8000:8000 \\')
    print(f'  vllm/vllm-openai:v0.19.0 \\')
    print(f'  --model {VLLM_MODEL} \\')
    print(f'  --enforce-eager \\')
    print(f'  --max-model-len 2048')
    print()

    print("常用 Docker 管理命令：")
    print()
    print("  # 查看运行中的容器")
    print("  docker ps")
    print()
    print("  # 查看容器日志")
    print("  docker logs -f vllm-server")
    print()
    print("  # 查看容器内 GPU 使用")
    print("  docker exec vllm-server nvidia-smi")
    print()
    print("  # 停止并删除容器")
    print("  docker stop vllm-server && docker rm vllm-server")
    print()
    print("  # 进入容器调试")
    print("  docker exec -it vllm-server bash")
    print()


def explain_image_selection():
    """
    Docker 镜像选择指南
    """
    print("=" * 60)
    print("vLLM Docker 镜像选择")
    print("=" * 60)
    print()
    print("┌─────────────────────────────┬──────────────────────────────────┐")
    print("│  镜像标签                    │  说明                             │")
    print("├─────────────────────────────┼──────────────────────────────────┤")
    print("│  vllm/vllm-openai:v0.19.0   │  固定版本，推荐生产使用            │")
    print("│  vllm/vllm-openai:latest    │  最新版，可能有 breaking change   │")
    print("│  vllm/vllm-openai:dev       │  开发版，仅用于测试新功能          │")
    print("└─────────────────────────────┴──────────────────────────────────┘")
    print()
    print("镜像内容：")
    print("  - Python 3.10/3.11")
    print("  - vLLM + 依赖（xformers, flash-attn 等）")
    print("  - CUDA Toolkit + cuDNN")
    print("  - 系统库（无需额外安装）")
    print()
    print("版本选择建议：")
    print("  - 生产环境：固定版本号（如 v0.19.0）")
    print("    → 可复现、可控、不会因更新出问题")
    print("  - 开发/测试：可以用latest")
    print("    → 需要定期验证兼容性")
    print()
    print("镜像大小：约 8-12GB（已含 CUDA 运行时）")
    print("首次 docker pull 较慢，后续利用层缓存增量更新")
    print()
    print("自定义镜像 vs 官方镜像：")
    print("  - 大多数场景：直接用官方镜像 + 命令行参数")
    print("  - 需要额外 Python 包：基于官方镜像写 Dockerfile")
    print("  - 需要自定义 API 层：单独构建 FastAPI 镜像")
    print()


def check_prerequisites():
    """
    检查 Docker 环境是否就绪
    """
    print("=" * 60)
    print("检查 Docker 环境是否就绪")
    print("=" * 60)
    print()

    checks = []

    # Docker
    ret = os.popen("docker --version 2>/dev/null").read().strip()
    if ret:
        checks.append(("Docker", True, ret))
    else:
        checks.append(("Docker", False, "未安装"))

    # Docker Compose
    ret = os.popen("docker compose version 2>/dev/null").read().strip()
    if ret:
        checks.append(("Docker Compose", True, ret))
    else:
        checks.append(("Docker Compose", False, "未安装"))

    # nvidia-container-toolkit
    ret = os.popen("nvidia-container-cli --version 2>/dev/null").read().strip()
    if ret:
        checks.append(("nvidia-container-toolkit", True, ret))
    else:
        checks.append(("nvidia-container-toolkit", False, "未安装"))

    # GPU
    ret = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null").read().strip()
    if ret:
        gpus = ret.split("\n")
        checks.append(("GPU", True, f"{len(gpus)}x {gpus[0].split(',')[0].strip()}"))
    else:
        checks.append(("GPU", False, "未检测到"))

    # HF cache
    if HF_HOME and os.path.isdir(HF_HOME):
        checks.append(("HF 缓存", True, HF_HOME))
    else:
        checks.append(("HF 缓存", False, f"路径无效: {HF_HOME}"))

    for name, ok, info in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {info}")

    print()
    all_ok = all(ok for _, ok, _ in checks)
    if all_ok:
        print("所有检查通过！可以开始 Docker 部署。")
    else:
        print("部分检查未通过，请先安装缺失的组件。")
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 5: Docker 基础与 vLLM 容器化部署")
    print()

    # 1. 检查环境是否就绪
    # check_prerequisites()

    # # 2. 概念讲解（不需要 GPU / Docker 运行）
    # explain_why_docker()
    # explain_docker_concepts()
    # explain_nvidia_container_toolkit()

    # 3. docker run 命令解析
    demo_docker_run_vllm()
    demo_docker_run_with_limits()

    # 4. 镜像选择
    explain_image_selection()

    print("=" * 60)
    print("Stage 5 Docker 基础要点总结")
    print("=" * 60)
    print("1. Docker 解决环境一致性、可复现、快速部署问题")
    print("2. 核心概念：镜像(模板)、容器(实例)、卷(存储)、网络(通信)")
    print("3. GPU 直通通过 nvidia-container-toolkit 实现，--gpus 指定 GPU")
    print("4. --ipc=host 是 vLLM 多卡推理的必需参数（共享内存）")
    print("5. 模型缓存通过 -v 挂载，避免容器内重复下载")
    print("6. 镜像版本固定（如 v0.19.0）是生产环境的最佳实践")

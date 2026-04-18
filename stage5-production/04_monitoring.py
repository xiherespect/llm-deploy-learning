"""
Stage 5: Prometheus + Grafana 监控

学习目标：
1. 理解 Prometheus 的核心概念（拉取模型、时间序列、PromQL）
2. 掌握 vLLM /metrics 端点暴露的关键指标
3. 学会编写 Prometheus 抓取配置
4. 掌握常用 PromQL 查询（速率、分位数、聚合）
5. 理解 Grafana 自动化 provisioning
6. 掌握 vLLM 监控仪表盘的设计

核心概念：
- Prometheus：拉取式监控系统，定时从目标抓取指标
- /metrics：vLLM 内置的 Prometheus 指标端点
- PromQL：Prometheus 查询语言，用于查询和聚合指标
- Grafana：可视化面板，从 Prometheus 读取数据并展示
- Provisioning：Grafana 自动配置（数据源、仪表盘），无需手动操作

前置条件：
- vLLM Docker 服务已启动
- Prometheus 和 Grafana 容器已启动

配置文件位置：
- docker/prometheus/prometheus.yml
- docker/grafana/provisioning/
- docker/grafana/alerts/vllm_alerts.yml
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import json
import time
import requests

VLLM_METRICS_URL = "http://localhost:8000/metrics"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"


def explain_prometheus_concepts():
    """
    Prometheus 核心概念
    """
    print("=" * 60)
    print("Prometheus 核心概念")
    print("=" * 60)
    print()
    print("架构：")
    print()
    print("  ┌──────────┐  scrape   ┌─────────────┐  query   ┌─────────┐")
    print("  │  vLLM     │──────────→│  Prometheus  │←────────│ Grafana │")
    print("  │  /metrics │  15s间隔   │  时序数据库   │  PromQL │  面板   │")
    print("  └──────────┘            └──────┬──────┘          └─────────┘")
    print("                                 │")
    print("                            ┌────▼─────┐")
    print("                            │  告警规则  │")
    print("                            │  AlertManager│")
    print("                            └──────────┘")
    print()
    print("拉取模型 vs 推送模型：")
    print("  - Prometheus：主动拉取（scrape），vLLM 只需暴露 /metrics")
    print("  - 推送模型：应用主动推送指标到监控系统")
    print("  - 拉取的优势：服务端控制抓取频率，自动发现新实例")
    print()
    print("指标类型：")
    print()
    print("  Counter（计数器）：只增不减，如请求总数、token 总数")
    print("    → 用 rate() 计算速率")
    print()
    print("  Gauge（仪表盘）：可增可减，如当前并发数、缓存使用率")
    print("    → 直接查看当前值")
    print()
    print("  Histogram（直方图）：分布统计，如延迟的 P50/P90/P95")
    print("    → 用 histogram_quantile() 计算分位数")
    print()
    print("时间序列数据格式：")
    print("  metric_name{label1='value1', label2='value2'} 1234.5")
    print("  ↑ 指标名          ↑ 标签（维度）                ↑ 值")
    print()


def explain_vllm_metrics():
    """
    vLLM 关键 Prometheus 指标详解
    """
    print("=" * 60)
    print("vLLM 关键 Prometheus 指标")
    print("=" * 60)
    print()
    print("┌────────────────────────────────────┬────────┬──────────────────────────────┐")
    print("│  指标名                             │  类型   │  说明                         │")
    print("├────────────────────────────────────┼────────┼──────────────────────────────┤")
    print("│  vllm:num_requests_running         │  Gauge │  正在处理的请求数              │")
    print("│  vllm:num_requests_waiting         │  Gauge │  排队等待的请求数              │")
    print("│  vllm:gpu_cache_usage_perc         │  Gauge │  KV Cache GPU 使用率 (%)      │")
    print("│  vllm:avg_generation_throughput    │  Gauge │  平均生成吞吐量 (tokens/s)     │")
    print("│  vllm:e2e_request_latency_seconds  │  Hist  │  端到端请求延迟                │")
    print("│  vllm:request_prompt_tokens        │  Count │  输入 token 总数               │")
    print("│  vllm:request_generation_tokens    │  Count │  输出 token 总数               │")
    print("│  vllm:num_preemptions              │  Count │  请求抢占次数                  │")
    print("│  vllm:gpu_memory_usage_bytes       │  Gauge │  GPU 显存使用量                │")
    print("└────────────────────────────────────┴────────┴──────────────────────────────┘")
    print()
    print("指标解读：")
    print()
    print("1. vllm:num_requests_running + vllm:num_requests_waiting")
    print("   running: 当前 GPU 正在处理的并发请求数")
    print("   waiting: 已接收但 GPU 忙，排队等待的请求数")
    print("   → waiting > 0 说明服务出现瓶颈，需要扩容或限流")
    print()
    print("2. vllm:gpu_cache_usage_perc")
    print("   KV Cache 的使用率，0-100%")
    print("   → 接近 100% 意味着无法接受更多并发请求")
    print("   → 新请求会被抢占（preemption）")
    print()
    print("3. vllm:e2e_request_latency_seconds")
    print("   从请求到达 vLLM 到生成完成的完整延迟")
    print("   是 Histogram 类型，可以用 histogram_quantile 计算 P50/P90/P95")
    print("   → P95 > 10s 说明用户体验差，需要优化")
    print()
    print("4. vllm:num_preemptions")
    print("   请求抢占次数：显存不够时，vLLM 会抢占低优先级请求的 KV Cache")
    print("   → 持续增长说明显存压力大，需要降低并发或增加 GPU")
    print()
    print("注意：vLLM 的指标名使用冒号(:)分隔，如 vllm:num_requests_running")
    print("Prometheus 完全支持，但在 PromQL 中需要小心不要误用")
    print()


def demo_fetch_metrics():
    """
    直接从 vLLM /metrics 端点获取指标
    """
    print("=" * 60)
    print("从 vLLM /metrics 端点获取指标")
    print("=" * 60)
    print()

    try:
        resp = requests.get(VLLM_METRICS_URL, timeout=5)
        if resp.status_code != 200:
            print(f"请求失败: HTTP {resp.status_code}")
            return

        metrics_text = resp.text

        # 解析关键指标
        key_metrics = {}
        for line in metrics_text.split("\n"):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # 提取指标名（去掉标签和值）
            metric_name = line.split("{")[0].split(" ")[0]
            if metric_name.startswith("vllm:"):
                try:
                    value = float(line.split()[-1])
                    key_metrics[metric_name] = value
                except (ValueError, IndexError):
                    pass

        print("当前 vLLM 指标：")
        print()
        for name, value in sorted(key_metrics.items()):
            if "bytes" in name:
                print(f"  {name}: {value / 1024**3:.2f} GB")
            elif "perc" in name:
                print(f"  {name}: {value:.1f}%")
            else:
                print(f"  {name}: {value:.0f}")

        print()
        print(f"共 {len(key_metrics)} 个 vLLM 指标")
        print()
        print("解读：")
        if "vllm:num_requests_running" in key_metrics:
            running = key_metrics["vllm:num_requests_running"]
            waiting = key_metrics.get("vllm:num_requests_waiting", 0)
            print(f"  当前处理中: {running:.0f}, 排队中: {waiting:.0f}")
            if waiting > 0:
                print("  ⚠️ 有请求在排队，服务可能过载")
            else:
                print("  ✓ 无排队，服务容量充足")
        if "vllm:gpu_cache_usage_perc" in key_metrics:
            cache = key_metrics["vllm:gpu_cache_usage_perc"]
            print(f"  KV Cache 使用率: {cache:.1f}%")
            if cache > 90:
                print("  ⚠️ KV Cache 接近满载，新请求可能被抢占")
            elif cache > 70:
                print("  ⚡ KV Cache 使用率较高，注意并发量")
            else:
                print("  ✓ KV Cache 充裕")

    except requests.ConnectionError:
        print("无法连接到 vLLM /metrics 端点")
        print("请确保 vLLM 服务已启动：docker compose up -d vllm")
    except Exception as e:
        print(f"获取指标失败: {e}")
    print()


def demo_promql_queries():
    """
    实用 PromQL 查询
    """
    print("=" * 60)
    print("实用 PromQL 查询")
    print("=" * 60)
    print()
    print("在 Prometheus UI (http://localhost:9090/graph) 中执行以下查询：")
    print()

    queries = [
        ("请求速率", "rate(vllm:e2e_request_latency_seconds_count[5m])",
         "最近 5 分钟的每秒请求数"),
        ("P95 延迟", "histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
         "95% 的请求延迟在多少秒以内"),
        ("P50 延迟", "histogram_quantile(0.50, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
         "50% 的请求延迟在多少秒以内"),
        ("Token 生成速率", "rate(vllm:request_generation_tokens_sum[5m])",
         "每秒生成的 token 数"),
        ("平均输入 token", "rate(vllm:request_prompt_tokens_sum[5m]) / rate(vllm:request_prompt_tokens_count[5m])",
         "平均每个请求的输入 token 数"),
        ("排队请求", "vllm:num_requests_waiting",
         "当前排队等待的请求数（>0 需关注）"),
        ("KV Cache 使用率", "vllm:gpu_cache_usage_perc",
         "KV Cache 使用百分比（>90% 需关注）"),
        ("抢占事件速率", "rate(vllm:num_preemptions[5m])",
         "每分钟抢占次数（>0 说明显存压力大）"),
        ("吞吐量", "vllm:avg_generation_throughput",
         "当前平均生成吞吐量 tokens/s"),
        ("GPU 显存使用", "vllm:gpu_memory_usage_bytes / 1024^3",
         "GPU 显存使用量（GB）"),
    ]

    for name, query, desc in queries:
        print(f"  {name}:")
        print(f"    {query}")
        print(f"    → {desc}")
        print()

    print("PromQL 常用函数：")
    print("  rate(metric[5m])         → 计算速率（适用于 Counter）")
    print("  histogram_quantile(0.95, rate(metric_bucket[5m]))  → 计算分位数")
    print("  sum(metric)              → 聚合求和")
    print("  avg(metric)              → 聚合求平均")
    print("  metric > threshold       → 过滤条件")
    print()


def explain_alert_rules():
    """
    Prometheus 告警规则
    """
    print("=" * 60)
    print("Prometheus 告警规则")
    print("=" * 60)
    print()
    print("告警规则文件：docker/grafana/alerts/vllm_alerts.yml")
    print()
    print("--- 告警1：队列过长 ---")
    print("  alert: HighQueueLength")
    print("  expr: vllm:num_requests_waiting > 10")
    print("  for: 2m")
    print("  → 排队请求持续 2 分钟超过 10 个")
    print("  → 原因：并发太高、模型推理太慢、GPU 不足")
    print()
    print("--- 告警2：KV Cache 接近满载 ---")
    print("  alert: CacheNearFull")
    print("  expr: vllm:gpu_cache_usage_perc > 90")
    print("  for: 5m")
    print("  → KV Cache 使用率持续 5 分钟超过 90%")
    print("  → 原因：长上下文请求太多、并发量过高")
    print("  → 后果：新请求会被抢占（preemption）")
    print()
    print("--- 告警3：请求抢占 ---")
    print("  alert: HighPreemptionRate")
    print("  expr: rate(vllm:num_preemptions[5m]) > 0")
    print("  for: 5m")
    print("  → 5 分钟内出现请求抢占")
    print("  → 严重：说明 GPU 显存已经不够分配 KV Cache")
    print("  → 修复：降低并发数、减小 max_model_len、增加 GPU")
    print()
    print("--- 告警4：P95 延迟过高 ---")
    print("  alert: HighTTFT")
    print("  expr: histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m])) > 10")
    print("  for: 5m")
    print("  → P95 延迟持续 5 分钟超过 10 秒")
    print("  → 用户感知：等很久才有回复")
    print()
    print("告警级别：")
    print("  warning  → 需要关注，但暂不紧急")
    print("  critical → 需要立即处理")
    print()


def explain_grafana_provisioning():
    """
    Grafana 自动化配置
    """
    print("=" * 60)
    print("Grafana 自动化 Provisioning")
    print("=" * 60)
    print()
    print("为什么需要 Provisioning？")
    print("  - 手动配置：每次 docker compose up 后重新配数据源、建仪表盘")
    print("  - Provisioning：容器启动时自动配置，零手动操作")
    print()
    print("Provisioning 配置文件：")
    print()
    print("1. 数据源自动配置（datasources/datasource.yml）：")
    print("   apiVersion: 1")
    print("   datasources:")
    print("     - name: Prometheus")
    print("       type: prometheus")
    print("       url: http://prometheus:9090")
    print("       isDefault: true")
    print("   → Grafana 启动后自动添加 Prometheus 数据源")
    print()
    print("2. 仪表盘自动配置（dashboards/dashboard.yml）：")
    print("   apiVersion: 1")
    print("   providers:")
    print("     - name: 'vLLM Dashboards'")
    print("       type: file")
    print("       options:")
    print("         path: /etc/grafana/provisioning/dashboards")
    print("   → 自动加载该目录下的 JSON 仪表盘文件")
    print()
    print("3. 仪表盘 JSON（dashboards/vllm_overview.json）：")
    print("   预构建的 vLLM 监控仪表盘，包含：")
    print()
    print("   Row 1: 总览")
    print("   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐")
    print("   │ Running  │ │ Waiting  │ │ Throughput│ │ Cache %  │")
    print("   │ Requests │ │ Requests │ │ tokens/s │ │ Usage    │")
    print("   └──────────┘ └──────────┘ └──────────┘ └──────────┘")
    print()
    print("   Row 2: 延迟")
    print("   ┌──────────────────────────────────────────────────┐")
    print("   │  E2E Latency P50 / P90 / P95 (time series)       │")
    print("   └──────────────────────────────────────────────────┘")
    print()
    print("   Row 3: 资源")
    print("   ┌──────────────────────────────────────────────────┐")
    print("   │  GPU Cache Usage % + Preemption Events            │")
    print("   └──────────────────────────────────────────────────┘")
    print()
    print("   Row 4: 流量")
    print("   ┌──────────────────────────────────────────────────┐")
    print("   │  Request Rate + Token Generation Rate             │")
    print("   └──────────────────────────────────────────────────┘")
    print()
    print("Grafana 访问：")
    print(f"  URL: http://localhost:3000")
    print("  用户名: admin")
    print("  密码: admin（或 .env 中的 GRAFANA_ADMIN_PASSWORD）")
    print()


def demo_grafana_api():
    """
    通过 Grafana API 查询数据
    """
    print("=" * 60)
    print("通过 Grafana API 查询监控数据")
    print("=" * 60)
    print()

    try:
        # 检查 Grafana 健康状态
        resp = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"Grafana 状态: {resp.json()}")
        else:
            print(f"Grafana 状态异常: {resp.status_code}")
            return
    except requests.ConnectionError:
        print("无法连接到 Grafana，请确保服务已启动")
        return
    print()

    # 查询数据源
    try:
        from requests.auth import HTTPBasicAuth
        auth = HTTPBasicAuth("admin", os.environ.get("GRAFANA_ADMIN_PASSWORD", "admin"))

        resp = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=auth,
            timeout=5,
        )
        if resp.status_code == 200:
            datasources = resp.json()
            print("已配置的数据源：")
            for ds in datasources:
                print(f"  - {ds['name']} ({ds['type']}) → {ds.get('url', 'N/A')}")
        print()

        # 查询仪表盘列表
        resp = requests.get(
            f"{GRAFANA_URL}/api/search?type=dash-db",
            auth=auth,
            timeout=5,
        )
        if resp.status_code == 200:
            dashboards = resp.json()
            print("已配置的仪表盘：")
            for db in dashboards:
                print(f"  - {db['title']} (uid: {db['uid']})")
        print()

    except Exception as e:
        print(f"Grafana API 查询失败: {e}")

    print("Grafana API 常用端点：")
    print("  GET  /api/health              → 健康检查")
    print("  GET  /api/datasources         → 数据源列表")
    print("  GET  /api/search?type=dash-db → 仪表盘搜索")
    print("  POST /api/dashboards/db       → 创建/更新仪表盘")
    print("  GET  /api/dashboards/uid/:uid → 获取仪表盘详情")
    print()


if __name__ == "__main__":
    print("大模型部署学习 - Stage 5: Prometheus + Grafana 监控")
    print()

    # # 1. Prometheus 概念
    # explain_prometheus_concepts()
    # explain_vllm_metrics()

    # # 2. PromQL 查询
    # demo_promql_queries()

    # # 3. 告警规则
    # explain_alert_rules()

    # # 4. Grafana 概念
    # explain_grafana_provisioning()

    # 5. 实时获取指标（需要 vLLM 运行）
    demo_fetch_metrics()

    # 6. Grafana API 查询（需要 Grafana 运行）
    demo_grafana_api()

    print("=" * 60)
    print("Stage 5 监控要点总结")
    print("=" * 60)
    print("1. Prometheus 拉取 vLLM /metrics 端点，15s 间隔")
    print("2. 关键指标：running/waiting 请求、KV Cache 使用率、延迟、抢占")
    print("3. PromQL 查询：rate() 算速率，histogram_quantile() 算分位数")
    print("4. 告警规则：队列>10、缓存>90%、抢占>0、P95>10s")
    print("5. Grafana Provisioning 自动配置数据源和仪表盘")
    print("6. 仪表盘四行：总览/延迟/资源/流量")

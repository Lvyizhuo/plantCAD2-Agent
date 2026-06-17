#!/usr/bin/env python3
"""测试 PlantCAD2 三个基础功能：嵌入提取、变异打分、掩码预测。

使用官方示例数据，通过 API 接口进行测试，并记录资源占用。

用法：
    python scripts/test_base_features.py [--host localhost] [--port 8005]
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests
import torch

# 官方示例序列（来自 notebooks/examples.ipynb）
DEMO_SEQUENCE = (
    "CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAG"
    "TAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCT"
    "TCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCT"
    "CAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGT"
    "GATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGA"
    "GAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGC"
    "TTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGA"
    "AGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAG"
    "CACAACCAAGAAGGGAAAAGATCGATGATCG"
)


def get_resource_usage():
    """采集当前进程和 GPU 的资源占用。"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    result = {
        "cpu_threads": process.num_threads(),
        "rss_mb": round(mem.rss / 1024 / 1024, 1),
        "vms_mb": round(mem.vms / 1024 / 1024, 1),
    }
    if torch.cuda.is_available():
        result["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        )
        result["gpu_memory_reserved_mb"] = round(
            torch.cuda.memory_reserved() / 1024 / 1024, 1
        )
    return result


def create_log_dir():
    """创建日志目录：logs/base_features/YYYYMMDD_HHMMSS/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / "base_features" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_log(log_dir, test_name, result):
    """写入测试日志。"""
    log_file = log_dir / f"{test_name}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  日志已保存: {log_file}")


def test_health(base_url):
    """测试健康检查。"""
    print("\n[1/4] 健康检查 GET /health")
    resp = requests.get(f"{base_url}/health", timeout=10)
    data = resp.json()
    print(f"  状态: {data}")
    return data


def test_embeddings(base_url, log_dir):
    """测试嵌入提取。"""
    print("\n[2/4] 嵌入提取 POST /embeddings")
    payload = {"sequence": DEMO_SEQUENCE, "normalize": True}

    resources_before = get_resource_usage()
    start = time.time()
    resp = requests.post(f"{base_url}/embeddings", json=payload, timeout=120)
    elapsed = time.time() - start
    resources_after = get_resource_usage()

    data = resp.json()
    result = {
        "endpoint": "/embeddings",
        "status_code": resp.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "sequence_length": len(DEMO_SEQUENCE),
            "input_size_bytes": len(DEMO_SEQUENCE.encode()),
            "format": "raw DNA string",
            "normalize": True,
        },
        "output": {
            "shape": data.get("shape"),
            "sequence_length": data.get("sequence_length"),
            "embedding_dim": data.get("shape", [0, 0])[1] if data.get("shape") else 0,
        },
        "resources_before": resources_before,
        "resources_after": resources_after,
    }
    if resp.status_code != 200:
        result["error"] = data

    print(f"  状态码: {resp.status_code}, 耗时: {elapsed:.3f}s")
    print(f"  输出形状: {data.get('shape')}")
    print(f"  资源占用: {resources_after}")
    write_log(log_dir, "embeddings", result)
    return result


def test_variant_score(base_url, log_dir):
    """测试变异打分。"""
    print("\n[3/4] 变异打分 POST /variant-score")
    payload = {
        "sequence": DEMO_SEQUENCE,
        "position": 100,
        "ref_allele": "A",
        "alt_alleles": ["G", "C", "T"],
    }

    resources_before = get_resource_usage()
    start = time.time()
    resp = requests.post(f"{base_url}/variant-score", json=payload, timeout=120)
    elapsed = time.time() - start
    resources_after = get_resource_usage()

    data = resp.json()
    result = {
        "endpoint": "/variant-score",
        "status_code": resp.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "sequence_length": len(DEMO_SEQUENCE),
            "input_size_bytes": len(DEMO_SEQUENCE.encode()),
            "format": "raw DNA string + position + alleles",
            "position": 100,
            "ref_allele": "A",
            "alt_alleles": ["G", "C", "T"],
        },
        "output": data,
        "resources_before": resources_before,
        "resources_after": resources_after,
    }
    if resp.status_code != 200:
        result["error"] = data

    print(f"  状态码: {resp.status_code}, 耗时: {elapsed:.3f}s")
    print(f"  LLR 分数: {data.get('scores')}")
    print(f"  资源占用: {resources_after}")
    write_log(log_dir, "variant_score", result)
    return result


def test_masked_predict(base_url, log_dir):
    """测试掩码位置预测。"""
    print("\n[4/4] 掩码预测 POST /masked-predict")
    positions = [100, 200, 255]
    payload = {"sequence": DEMO_SEQUENCE, "positions": positions}

    resources_before = get_resource_usage()
    start = time.time()
    resp = requests.post(f"{base_url}/masked-predict", json=payload, timeout=120)
    elapsed = time.time() - start
    resources_after = get_resource_usage()

    data = resp.json()
    result = {
        "endpoint": "/masked-predict",
        "status_code": resp.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "input": {
            "sequence_length": len(DEMO_SEQUENCE),
            "input_size_bytes": len(DEMO_SEQUENCE.encode()),
            "format": "raw DNA string + positions list",
            "positions": positions,
        },
        "output": data,
        "resources_before": resources_before,
        "resources_after": resources_after,
    }
    if resp.status_code != 200:
        result["error"] = data

    print(f"  状态码: {resp.status_code}, 耗时: {elapsed:.3f}s")
    print(f"  预测结果: {data.get('predictions')}")
    print(f"  资源占用: {resources_after}")
    write_log(log_dir, "masked_predict", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="测试 PlantCAD2 三个基础功能")
    parser.add_argument("--host", default="localhost", help="服务地址")
    parser.add_argument("--port", type=int, default=8005, help="服务端口")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    log_dir = create_log_dir()
    print(f"测试目标: {base_url}")
    print(f"日志目录: {log_dir}")

    # 测试序列信息
    seq_file = log_dir / "input_sequence.txt"
    with open(seq_file, "w") as f:
        f.write(DEMO_SEQUENCE)
    print(f"测试序列长度: {len(DEMO_SEQUENCE)} bp")

    # 执行测试
    test_health(base_url)
    test_embeddings(base_url, log_dir)
    test_variant_score(base_url, log_dir)
    test_masked_predict(base_url, log_dir)

    print(f"\n全部完成，日志保存在: {log_dir}")


if __name__ == "__main__":
    main()

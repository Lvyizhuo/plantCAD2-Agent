#!/usr/bin/env python3
"""测试 PlantCAD2 七个 LoRA 下游任务。

使用 test/PlantCAD2_fine_tuning_tasks/ 中的官方测试数据集，
通过 API 接口进行测试，并记录资源占用。

用法：
    # 测试所有任务，每个测试文件采样 10 条
    python scripts/test_lora_tasks.py

    # 指定采样数量
    python scripts/test_lora_tasks.py --num_samples 20

    # 只测试某个任务
    python scripts/test_lora_tasks.py --task acr_arabidopsis

    # 指定服务地址
    python scripts/test_lora_tasks.py --host localhost --port 8005
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import requests
import torch

# task 名称 -> 数据目录名
TASK_DATA_MAP = {
    "acr_arabidopsis": "cross_species_acr_train_on_arabidopsis",
    "acr_nine_species": "cross_species_acr_train_on_nine_species",
    "acr_cell_type": "cell_type_specific_acr",
    "expression_on_off": "cross_species_leaf_on_off_expression",
    "expression_absolute": "cross_species_leaf_absolute_expression",
    "translation_on_off": "cross_species_leaf_on_off_translation",
    "translation_absolute": "cross_species_leaf_absolute_translation",
}

# 任务类型（决定输出解读方式）
TASK_TYPES = {
    "acr_arabidopsis": "binary",
    "acr_nine_species": "binary",
    "acr_cell_type": "multi_label",
    "expression_on_off": "binary",
    "expression_absolute": "regression",
    "translation_on_off": "binary",
    "translation_absolute": "regression",
}

DATA_ROOT = Path("test/PlantCAD2_fine_tuning_tasks")


def get_resource_usage():
    """采集当前进程和 GPU 的资源占用。"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    result = {
        "cpu_threads": process.num_threads(),
        "rss_mb": round(mem.rss / 1024 / 1024, 1),
    }
    if torch.cuda.is_available():
        result["gpu_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1024 / 1024, 1
        )
    return result


def get_test_files(task_data_dir):
    """获取任务目录下所有 test 开头的 parquet 文件。"""
    data_dir = DATA_ROOT / task_data_dir
    if not data_dir.exists():
        print(f"  警告: 数据目录不存在 {data_dir}")
        return []
    files = sorted(data_dir.glob("test*.parquet"))
    return files


def load_samples(parquet_path, num_samples):
    """从 parquet 文件中加载样本。

    num_samples=0 表示加载全部数据。
    """
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return []
    if num_samples == 0:
        return df.to_dict("records")
    # 均匀采样
    if len(df) <= num_samples:
        samples = df
    else:
        step = len(df) // num_samples
        samples = df.iloc[::step].head(num_samples)
    return samples.to_dict("records")


def run_predict_test(base_url, task_name, sequence, timeout=120):
    """调用 /predict 接口并返回结果和资源占用。"""
    payload = {"sequence": sequence, "task": task_name}
    resources_before = get_resource_usage()
    start = time.time()
    try:
        resp = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)
        elapsed = time.time() - start
        resources_after = get_resource_usage()
        data = resp.json()
        return {
            "status_code": resp.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "response": data,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "error": None if resp.status_code == 200 else data,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status_code": 0,
            "elapsed_seconds": round(elapsed, 3),
            "response": None,
            "resources_before": resources_before,
            "resources_after": get_resource_usage(),
            "error": str(e),
        }


def test_task(base_url, task_name, num_samples, log_dir):
    """测试单个 LoRA 任务。"""
    data_dir_name = TASK_DATA_MAP[task_name]
    task_type = TASK_TYPES[task_name]
    test_files = get_test_files(data_dir_name)

    if not test_files:
        print(f"  跳过: 无测试数据文件")
        return None

    print(f"\n  任务类型: {task_type}")
    print(f"  测试文件数: {len(test_files)}")

    all_results = []
    total_correct = 0
    total_count = 0
    total_elapsed = 0

    for test_file in test_files:
        file_size_mb = round(test_file.stat().st_size / 1024 / 1024, 2)
        print(f"  文件: {test_file.name} ({file_size_mb} MB)")

        samples = load_samples(test_file, num_samples)
        if not samples:
            print(f"    空文件，跳过")
            continue

        for i, row in enumerate(samples):
            sequence = row.get("sequence", "")
            label = row.get("label")

            if not sequence:
                continue

            result = run_predict_test(base_url, task_name, sequence)
            result["sample_index"] = i
            result["input_sequence_length"] = len(sequence)
            result["input_size_bytes"] = len(sequence.encode())
            result["ground_truth_label"] = label
            result["source_file"] = test_file.name
            result["source_file_size_mb"] = file_size_mb

            # 判断预测是否正确
            prediction = result.get("response", {})
            if task_type == "binary":
                pred_label = 1 if prediction.get("prediction") == "POSITIVE" else 0
                result["predicted_label"] = pred_label
                result["correct"] = pred_label == int(label)
            elif task_type == "regression":
                pred_val = prediction.get("prediction")
                result["predicted_value"] = pred_val
                result["ground_truth_value"] = float(label)
            # multi_label 跳过准确率统计

            all_results.append(result)
            total_elapsed += result["elapsed_seconds"]

            if task_type == "binary":
                total_count += 1
                if result.get("correct"):
                    total_correct += 1

            status = "✓" if result.get("error") is None else "✗"
            print(f"    [{i+1}/{len(samples)}] {status} {result['elapsed_seconds']}s")

    # 汇总
    summary = {
        "task": task_name,
        "task_type": task_type,
        "data_dir": data_dir_name,
        "test_files_count": len(test_files),
        "total_samples_tested": len(all_results),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_elapsed_seconds": round(total_elapsed / max(len(all_results), 1), 3),
        "errors": sum(1 for r in all_results if r.get("error")),
    }
    if task_type == "binary" and total_count > 0:
        summary["accuracy"] = round(total_correct / total_count, 4)
        summary["correct"] = total_correct
        summary["total"] = total_count

    # 保存详细日志
    task_log_dir = log_dir / task_name
    task_log_dir.mkdir(parents=True, exist_ok=True)

    summary_file = task_log_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    details_file = task_log_dir / "details.json"
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"  完成: {summary['total_samples_tested']} 条样本, "
          f"平均 {summary['avg_elapsed_seconds']}s/条")
    if "accuracy" in summary:
        print(f"  准确率: {summary['accuracy']} ({summary['correct']}/{summary['total']})")
    print(f"  日志: {task_log_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="测试 PlantCAD2 七个 LoRA 下游任务")
    parser.add_argument("--host", default="localhost", help="服务地址")
    parser.add_argument("--port", type=int, default=8005, help="服务端口")
    parser.add_argument("--num_samples", type=int, default=10, help="每个测试文件采样数量，0 表示测试全部数据")
    parser.add_argument("--task", default=None, help="只测试指定任务（如 acr_arabidopsis）")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / "lora_tasks" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"测试目标: {base_url}")
    print(f"日志目录: {log_dir}")
    print(f"采样数量: {'全部' if args.num_samples == 0 else f'{args.num_samples} 条/文件'}")

    # 检查服务
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"服务状态: {resp.json()}")
    except Exception as e:
        print(f"错误: 无法连接服务 {base_url} - {e}")
        return

    # 确定要测试的任务
    if args.task:
        if args.task not in TASK_DATA_MAP:
            print(f"错误: 未知任务 '{args.task}'，可选: {list(TASK_DATA_MAP.keys())}")
            return
        tasks_to_test = [args.task]
    else:
        tasks_to_test = list(TASK_DATA_MAP.keys())

    # 逐个测试
    all_summaries = []
    for task_name in tasks_to_test:
        print(f"\n{'='*60}")
        print(f"测试任务: {task_name}")
        print(f"{'='*60}")
        summary = test_task(base_url, task_name, args.num_samples, log_dir)
        if summary:
            all_summaries.append(summary)

    # 总汇总
    final_summary = {
        "test_time": timestamp,
        "base_url": base_url,
        "num_samples_per_file": args.num_samples,
        "tasks_tested": len(all_summaries),
        "summaries": all_summaries,
    }
    summary_file = log_dir / "overall_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"全部测试完成")
    print(f"{'='*60}")
    for s in all_summaries:
        acc = f", 准确率 {s['accuracy']}" if "accuracy" in s else ""
        print(f"  {s['task']}: {s['total_samples_tested']} 条, "
              f"{s['avg_elapsed_seconds']}s/条{acc}")
    print(f"\n总日志: {log_dir}")


if __name__ == "__main__":
    main()

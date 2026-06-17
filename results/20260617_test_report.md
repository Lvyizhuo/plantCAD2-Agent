# PlantCAD2 推理服务测试报告

**测试日期**：2026-06-17
**测试环境**：NVIDIA A100-PCIE-40GB × 4（GPU 2 用于推理）
**服务地址**：http://localhost:8005
**服务启动命令**：`CUDA_VISIBLE_DEVICES=2 PLANTCAD2_DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8005`

---

## 一、测试概述

| 项目 | 结果 |
|------|------|
| 基础功能测试 | 3/3 通过 |
| LoRA 任务测试 | 7/7 通过 |
| 总测试样本数 | 4,700 条 |
| 总错误数 | 0 |
| 总测试时长 | 约 23 分钟 |

---

## 二、基础功能测试

测试数据：官方示例序列（512 bp），来源 `test/quick_test_example.py` 和 `test/plantcad_colab_example.py`。

### 2.1 嵌入提取（POST /embeddings）

| 指标 | 值 |
|------|-----|
| 状态码 | 200 ✅ |
| 耗时 | 8.175s（含首次模型加载） |
| 输入序列长度 | 512 bp |
| 输入大小 | 512 bytes |
| 输出形状 | [512, 1536] |
| 嵌入维度 | 1536 |
| CPU 线程 | 32 → 33 |
| 内存 RSS | 395.6 → 416.2 MB |
| 虚拟内存 | 4,515 → 21,152 MB |

**说明**：首次请求耗时较长（8.175s），主要消耗在加载基座模型（~2.6GB）。后续请求预计 <1s。

### 2.2 变异打分（POST /variant-score）

| 指标 | 值 |
|------|-----|
| 状态码 | 200 ✅ |
| 耗时 | 0.298s |
| 输入 | 序列 512 bp + 位置 100 + 参考 A → 变异 G/C/T |
| 参考碱基概率 | A: 0.246 |
| 变异概率 | G: 0.110, C: 0.116, T: 0.527 |
| LLR 分数 | G: -0.805, C: -0.750, T: +0.760 |
| CPU 线程 | 33 |
| 内存 RSS | 407.7 MB |

**解读**：位置 100 的参考碱基 A 概率较低（0.246），T 的概率最高（0.527），LLR(T)=+0.76 表示 T 比 A 更常见。该位点不太保守。

### 2.3 掩码预测（POST /masked-predict）

| 指标 | 值 |
|------|-----|
| 状态码 | 200 ✅ |
| 耗时 | 0.844s（3 个位置） |
| 输入 | 序列 512 bp + 位置 [100, 200, 255] |
| CPU 线程 | 33 |
| 内存 RSS | 407.7 MB |

**各位置预测结果**：

| 位置 | A | C | G | T | 最大概基 | 解读 |
|------|------|------|------|------|----------|------|
| 100 | 0.246 | 0.116 | 0.110 | 0.527 | T (52.7%) | 不保守，可变 |
| 200 | 0.261 | 0.198 | 0.060 | 0.481 | T (48.1%) | 不保守，可变 |
| 255 | **0.966** | 0.007 | 0.013 | 0.014 | A (96.6%) | **高度保守** |

位置 255 的 A 概率高达 96.6%，说明该位点在进化上高度保守，功能重要。

---

## 三、LoRA 下游任务测试

测试参数：每个测试文件采样 100 条，共 7 个任务。

### 3.1 总览

| 任务 | 类型 | 测试文件数 | 样本数 | 耗时 | 平均耗时 | 错误数 | 准确率 |
|------|------|-----------|--------|------|----------|--------|--------|
| acr_arabidopsis | 二分类 | 23 | 2,300 | 772.3s | 0.336s | 0 | **86.61%** |
| acr_nine_species | 二分类 | 11 | 1,100 | 350.0s | 0.318s | 0 | **94.55%** |
| acr_cell_type | 多标签 | 1 | 100 | 31.9s | 0.319s | 0 | — |
| expression_on_off | 二分类 | 4 | 400 | 128.4s | 0.321s | 0 | 55.50% |
| expression_absolute | 回归 | 4 | 400 | 128.2s | 0.320s | 0 | — |
| translation_on_off | 二分类 | 1 | 100 | 31.8s | 0.318s | 0 | 58.00% |
| translation_absolute | 回归 | 1 | 100 | 32.1s | 0.321s | 0 | — |
| **合计** | — | **45** | **4,700** | **1,474.7s** | **0.323s** | **0** | — |

### 3.2 分类任务准确率

| 任务 | 准确率 | 正确/总数 | 评估 |
|------|--------|-----------|------|
| acr_nine_species | 94.55% | 1,040/1,100 | ✅ 优秀 |
| acr_arabidopsis | 86.61% | 1,992/2,300 | ✅ 良好 |
| translation_on_off | 58.00% | 58/100 | ⚠️ 一般 |
| expression_on_off | 55.50% | 222/400 | ⚠️ 一般 |

**分析**：
- ACR 预测任务表现优秀，尤其是九物种模型（94.55%），说明模型在跨物种 ACR 识别上有很强的泛化能力
- 表达量和翻译效率的开/关预测准确率接近随机水平（~50%），可能原因：
  - 这些任务本身难度较高，序列 alone 不足以准确预测表达/翻译状态
  - 测试数据可能包含边界样本（难以区分的正负例）
  - 模型可能需要更长的上下文序列（当前测试序列 500-2048 bp）

### 3.3 资源占用

| 指标 | 值 |
|------|-----|
| CPU 线程 | 70-73 |
| 内存 RSS | 6.5-7.5 GB（随任务加载递增） |
| GPU 显存（服务端） | 未采集（测试脚本为客户端进程） |

**说明**：测试脚本通过 HTTP 请求调用服务，`psutil` 采集的是客户端进程的资源。服务端的 GPU 显存占用需在服务器上用 `nvidia-smi` 确认。

### 3.4 输入数据规格

| 任务 | 序列长度 | 输入大小 | 数据格式 |
|------|----------|----------|----------|
| acr_arabidopsis | 600 bp | 600 bytes | parquet（sequence + label） |
| acr_nine_species | 600 bp | 600 bytes | parquet（sequence + label） |
| acr_cell_type | 600 bp | 600 bytes | parquet（sequence + label: 92 位二进制串） |
| expression_on_off | 2,048 bp | 2,048 bytes | parquet（sequence + label: 0/1） |
| expression_absolute | 2,048 bp | 2,048 bytes | parquet（sequence + label: float） |
| translation_on_off | 500 bp | 500 bytes | parquet（sequence + label: 0/1） |
| translation_absolute | 500 bp | 500 bytes | parquet（sequence + label: float） |

---

## 四、问题与建议

### 4.1 已知问题

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| 1 | 测试脚本采集的 GPU 显存为 0（客户端进程无模型） | 无法评估服务端显存 | 在服务器上用 `nvidia-smi` 手动采集，或在服务端添加显存监控端点 |
| 2 | 表达量/翻译效率的开/关预测准确率低（~55%） | 可能不满足业务需求 | 确认客户实际使用场景，是否需要更长序列或额外特征 |
| 3 | 回归任务缺少准确率指标 | 无法评估回归效果 | 建议计算 MSE/R² 等回归评估指标 |

### 4.2 后续建议

1. **显存监控**：在服务端 `/health` 端点中增加 GPU 显存使用量返回
2. **回归评估**：对 expression_absolute 和 translation_absolute 计算 MSE、MAE、R² 等指标
3. **性能基线**：记录不同序列长度（500/1000/2048/4096/8192 bp）的推理耗时曲线
4. **并发测试**：测试多请求并发下的吞吐量和延迟

---

## 五、测试日志位置

```
logs/
├── base_features/20260617_164215/    ← 基础功能测试
│   ├── embeddings.json
│   ├── variant_score.json
│   ├── masked_predict.json
│   └── input_sequence.txt
└── lora_tasks/20260617_164514/       ← LoRA 任务测试
    ├── overall_summary.json
    ├── acr_arabidopsis/{summary,details}.json
    ├── acr_nine_species/{summary,details}.json
    ├── acr_cell_type/{summary,details}.json
    ├── expression_on_off/{summary,details}.json
    ├── expression_absolute/{summary,details}.json
    ├── translation_on_off/{summary,details}.json
    └── translation_absolute/{summary,details}.json
```

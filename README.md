# 🚀 Qwen2.5-7B 混合精度量化 (W2/W4/W8 + A8)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Quantization-W2%2FW4%2FW8%20%2B%20A8-green.svg" alt="Quantization">
  <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
</p>

基于遗传算法优化的 **混合精度后训练量化 (Mixed-Precision PTQ)** 框架，专为 Qwen2.5-7B 大语言模型设计。

```
┌─────────────────────────────────────────────────────────────────────┐
│  量化策略: W2/W4/W8 + A8 (权重可变位宽 + 固定8位激活)               │
├─────────────────────────────────────────────────────────────────────┤
│  • W2 + A8: 低敏感层 → 最大压缩 (1/8 原始大小)                      │
│  • W4 + A8: 中敏感层 → 平衡压缩 (1/4 原始大小)                      │
│  • W8 + A8: 高敏感层 → 保持精度 (1/2 原始大小)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📋 目录

- [快速开始](#-快速开始)
- [项目架构](#-项目架构)
- [使用指南](#-使用指南)
- [核心概念](#-核心概念)
- [性能对比](#-性能对比)
- [常见问题](#-常见问题)

---

## ⚡ 快速开始

### 30秒体验真实量化加速

```bash
# 1. 安装依赖
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python  # macOS
# CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python  # Linux/CUDA

# 2. 下载预量化模型 (4.36 GB)
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

# 3. 运行对比测试
python compare_real_quant.py --max_tokens 200
```

### 完整工作流 (自定义混合精度配置)

```bash
# 步骤1: 敏感度分析 + 遗传算法搜索最优配置 (~30-60分钟)
python mixed_precision_ptq.py --device mps --target_compression 0.25

# 步骤2: 导出为 GGUF 格式 (~5分钟)  
python export_gguf_official.py --output models/qwen2.5-7b-mixed.gguf

# 步骤3: 三模型对比测试
python compare_three_models.py --max_tokens 200
```

---

## 📁 项目架构

```
Qwen2.5-7B_W2-8A8_MIXED_PTQ/
│
├── 🔧 核心模块
│   ├── quant_utils.py              # 量化核心函数 (模拟量化、SmoothQuant)
│   ├── genetic_optim.py            # 遗传算法优化器 (搜索最优W2/W4/W8配置)
│   └── data_utils.py               # 校准数据加载 (WikiText-2等)
│
├── ⚙️ 主程序
│   ├── mixed_precision_ptq.py      # 混合精度量化主程序 (敏感度分析+GA搜索)
│   └── export_gguf_official.py     # GGUF格式导出 (llama.cpp兼容)
│
├── 🧪 测试脚本
│   ├── compare_real_quant.py       # ✅ 真实量化对比 (推荐！能获得加速)
│   ├── compare_three_models.py     # 三模型全面对比
│   └── test_simulated_quant.py     # ⚠️ 模拟量化测试 (仅验证精度)
│
├── 📄 配置文件
│   ├── requirements.txt            # Python依赖
│   ├── mixed_precision_config.pt   # 量化配置 (每层位宽)
│   └── README.md                   # 项目说明
│
└── 📦 模型目录
    └── models/                     # GGUF模型文件
```

---

## 📖 使用指南

### 1️⃣ 混合精度量化配置搜索

```bash
python mixed_precision_ptq.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \  # 模型ID
    --device mps \                          # 设备: cuda/mps/cpu
    --ga_pop 20 \                           # 遗传算法种群大小
    --ga_gen 15 \                           # 遗传算法迭代代数
    --target_compression 0.25               # 目标压缩比 (25%)
```

**输出**: `mixed_precision_config.pt` - 每层的量化位宽配置

### 2️⃣ 导出 GGUF 格式

```bash
python export_gguf_official.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --config mixed_precision_config.pt \
    --output models/qwen2.5-7b-mixed.gguf
```

### 3️⃣ 推理测试

```bash
# 真实量化对比 (推荐！)
python compare_real_quant.py --max_tokens 200

# 三模型对比 (原始 vs Q4_K_M vs 混合精度)
python compare_three_models.py --skip_original --max_tokens 200

# 模拟量化测试 (仅验证精度，不会加速)
python test_simulated_quant.py --prompt "什么是量子计算？"
```

---

## 🎯 核心概念

### 模拟量化 vs 真实量化

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ 模拟量化 (Simulated) - 用于配置搜索                            │
├─────────────────────────────────────────────────────────────────────┤
│  FP32 → 量化(round) → 反量化 → FP32                                 │
│  • 数据类型始终是 FP32，只模拟精度损失                              │
│  • ❌ 不会加速 (反而更慢)                                            │
│  • ✅ 用于评估量化配置的精度影响                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ✅ 真实量化 (Real) - 用于生产部署                                   │
├─────────────────────────────────────────────────────────────────────┤
│  FP32 → INT4/INT8 → GGUF格式 → llama.cpp推理                        │
│  • 使用低精度整数运算，硬件加速                                     │
│  • ✅ 推理加速 5-10x                                                 │
│  • ✅ 内存减少 70-85%                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### W2/W4/W8 + A8 策略

| 组件 | 位宽 | 量化方式 | 说明 |
|------|------|----------|------|
| **权重 (W)** | 2/4/8-bit 可变 | 对称量化 + 分组(g=128) | 根据层敏感度选择 |
| **激活 (A)** | 8-bit 固定 | 非对称量化 (per-tensor) | 统一8位简化硬件 |

### 层敏感度分类

| 敏感度 | MSE阈值 | 推荐配置 | 典型层类型 |
|--------|---------|----------|------------|
| 低 | < 0.1 | W2 + A8 | FFN Down, 部分Embedding |
| 中 | 0.1~0.5 | W4 + A8 | Attention V/O, FFN Gate/Up |
| 高 | ≥ 0.5 | W8 + A8 | Attention Q/K |

---

## 📊 性能对比

**测试环境**: Apple M4 Max, 32GB RAM

| 模型 | 大小 | 推理速度 | 内存占用 | 加速比 |
|------|------|----------|----------|--------|
| 原始 (FP16) | 14.2 GB | 14.7 tok/s | ~30 GB | 1.0x |
| Q4_K_M | 4.36 GB | **68.5 tok/s** | ~5 GB | **4.7x** |
| 混合精度 | 8.54 GB | 53.5 tok/s | ~9 GB | 3.6x |
| 模拟量化 | 14.2 GB | 0.8 tok/s ❌ | ~30 GB | 0.05x |

> 💡 **结论**: 真实量化能获得 **5-10x 加速**，模拟量化只用于配置搜索！

---

## ❓ 常见问题

<details>
<summary><b>Q: 为什么模拟量化后更慢了？</b></summary>

这是正常的！模拟量化在 FP32 基础上增加了额外的量化/反量化操作，只是模拟精度损失，不会加速。想要真正加速，请使用 `compare_real_quant.py`。
</details>

<details>
<summary><b>Q: MPS设备报错怎么办？</b></summary>

1. 使用 `torch.float32` (MPS对FP16支持有限)
2. 更新 PyTorch: `pip install --upgrade torch`
3. 重新编译 llama-cpp-python:
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
   ```
</details>

<details>
<summary><b>Q: 量化后输出乱码？</b></summary>

W2层过多可能导致精度损失，调整目标压缩比:
```bash
python mixed_precision_ptq.py --target_compression 0.35  # 提高到35%
```
</details>

<details>
<summary><b>Q: 如何跳过原始模型测试（节省内存）？</b></summary>

```bash
python compare_three_models.py --skip_original --max_tokens 200
```
</details>

---

## 💻 硬件要求

| 设备 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CUDA GPU | 16GB VRAM | 24GB+ (A100/4090) |
| Apple Silicon | M1 16GB | M2 Pro 32GB+ |
| CPU | 32GB RAM | 64GB+ RAM |

---

## 📚 参考文献

- [SmoothQuant](https://arxiv.org/abs/2211.10438) - 激活值平滑量化
- [GPTQ](https://arxiv.org/abs/2210.17323) - 后训练量化
- [AWQ](https://arxiv.org/abs/2306.00978) - 激活感知量化
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF推理引擎
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 模型官方仓库

---

## 📄 License

MIT License © 2024 Jiangsheng Yu

---

<p align="center">
  ⭐ 如果这个项目对你有帮助，请给个 Star！
</p>

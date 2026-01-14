"""
三模型对比测试脚本
==================

功能说明：
---------
对比三种模型的推理性能和输出质量：
    1. 原始模型 (FP32/FP16) - 使用 Transformers 库
    2. Q4_K_M 统一量化 (4-bit) - 使用 llama.cpp (GGUF格式)
    3. 混合精度量化 (W2/W4/W8) - 使用 llama.cpp (自定义GGUF)

测试指标：
---------
    - 推理速度 (tokens/second)
    - 生成质量 (输出文本对比)
    - 模型大小 (GB)
    - 内存占用

使用方法：
---------
    # 基础用法（跳过原始模型以节省内存）
    >>> python compare_three_models.py --skip_original
    
    # 完整对比（需要足够内存）
    >>> python compare_three_models.py --max_tokens 200
    
    # 自定义生成长度
    >>> python compare_three_models.py --max_tokens 300 --skip_original

作者：Jiangsheng Yu
"""

import torch
import time
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# 依赖检查
# ============================================================================
# llama-cpp-python 是用于加载和运行 GGUF 格式模型的库
# 它提供了高效的 CPU/GPU 推理能力
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python 未安装，将跳过 GGUF 模型测试")
    print("   安装命令:")
    print("   - macOS (Metal): CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
    print("   - Linux (CUDA):  CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python")


# ============================================================================
# 工具函数
# ============================================================================

def get_device() -> str:
    """
    自动检测最佳计算设备
    
    检测顺序：
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU (通用后备)
    
    返回:
        str: 设备名称 ('cuda', 'mps', 或 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_with_transformers(model, tokenizer, prompt: str, device: str,
                                max_new_tokens: int = 200) -> tuple:
    """
    使用 HuggingFace Transformers 库生成回复
    
    参数:
        model: 已加载的 Transformers 模型
        tokenizer: 对应的分词器
        prompt: 用户输入的提示文本
        device: 计算设备 ('cuda', 'mps', 'cpu')
        max_new_tokens: 最大生成 token 数量（默认200）
    
    返回:
        tuple: (生成的文本, 耗时秒数, 生成的token数)
    
    注意:
        - 会先进行一次预热推理以获得更准确的时间测量
        - 使用 greedy decoding (do_sample=False) 确保结果可复现
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预热
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    
    # 正式推理
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start_time
    
    new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response, elapsed, new_tokens


def generate_with_llama_cpp(model: "Llama", prompt: str,
                             max_tokens: int = 200) -> tuple:
    """
    使用 llama.cpp 生成回复
    
    参数:
        model: 已加载的 Llama 模型实例
        prompt: 用户输入的提示文本
        max_tokens: 最大生成 token 数量（默认200）
    
    返回:
        tuple: (生成的文本, 耗时秒数, 生成的token数)
    
    特点:
        - llama.cpp 使用优化的 C++ 后端，推理速度快
        - 支持 Metal (macOS) 和 CUDA (Linux/Windows) 加速
        - 使用 temperature=0.0 确保输出确定性
    """
    # 预热：首次推理可能较慢，不计入测量
    _ = model.create_chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=3
    )
    
    # 正式推理
    start_time = time.time()
    response = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start_time
    
    content = response['choices'][0]['message']['content']
    tokens = response['usage']['completion_tokens']
    
    return content, elapsed, tokens


def print_result(name: str, response: str, elapsed: float, tokens: int, 
                 color_code: str = ""):
    """
    格式化打印单个模型的推理结果
    
    参数:
        name: 模型名称
        response: 生成的回复文本
        elapsed: 推理耗时（秒）
        tokens: 生成的 token 数量
        color_code: 可选的颜色/图标前缀
    """
    print(f"\n{'─'*80}")
    print(f"{color_code}【{name}】")
    print(f"{'─'*80}")
    # 显示更多文本内容（最多600字符），便于对比输出质量
    print(f"{response[:600]}{'...' if len(response) > 600 else ''}")
    print(f"\n   ⏱️  耗时: {elapsed:.2f}s | Tokens: {tokens} | 速度: {tokens/elapsed:.1f} tok/s")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主函数：解析命令行参数并执行三模型对比测试
    
    测试流程：
        1. 加载所有可用模型
        2. 对每个测试用例执行推理
        3. 记录并对比性能指标
        4. 输出总结报告
    """
    # ---- 命令行参数解析 ----
    parser = argparse.ArgumentParser(
        description="三模型对比测试：原始模型 vs Q4_K_M vs 混合精度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python compare_three_models.py --skip_original --max_tokens 200
  python compare_three_models.py --max_tokens 300
        """
    )
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace 模型 ID（原始模型）")
    parser.add_argument('--q4km_path', type=str, 
                        default="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                        help="Q4_K_M 量化模型路径 (GGUF格式)")
    parser.add_argument('--mixed_path', type=str,
                        default="models/qwen2.5-7b-mixed.gguf",
                        help="混合精度量化模型路径 (GGUF格式)")
    parser.add_argument('--max_tokens', type=int, default=200,
                        help="最大生成 token 数（默认200，建议不少于100）")
    parser.add_argument('--skip_original', action='store_true',
                        help="跳过原始模型测试（节省内存和时间）")
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*80)
    print("🚀 三模型对比测试")
    print("   原始模型 vs Q4_K_M vs 混合精度")
    print("="*80)
    print(f"\n📍 设备: {device}")
    print(f"📦 原始模型: {args.model_id}")
    print(f"📦 Q4_K_M: {args.q4km_path}")
    print(f"📦 混合精度: {args.mixed_path}")
    
    # 检查文件
    if not os.path.exists(args.q4km_path):
        print(f"\n❌ 找不到 Q4_K_M 模型: {args.q4km_path}")
        return
    if not os.path.exists(args.mixed_path):
        print(f"\n❌ 找不到混合精度模型: {args.mixed_path}")
        return
    
    # 显示文件大小
    q4km_size = os.path.getsize(args.q4km_path) / (1024**3)
    mixed_size = os.path.getsize(args.mixed_path) / (1024**3)
    print(f"\n📊 模型大小:")
    print(f"   Q4_K_M: {q4km_size:.2f} GB")
    print(f"   混合精度: {mixed_size:.2f} GB")
    
    # ========== 加载模型 ==========
    print("\n" + "─"*80)
    print("⏳ 加载模型...")
    
    models = {}
    
    # 1. 原始模型（可选）
    if not args.skip_original:
        print("\n📥 加载原始模型 (Transformers)...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if device == "mps":
            original_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float32
            ).to(device)
        else:
            original_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16
            ).to(device)
        original_model.eval()
        models['original'] = (original_model, tokenizer)
        print("✅ 原始模型加载完成")
    
    # 2. Q4_K_M 模型
    if LLAMA_CPP_AVAILABLE:
        print("\n📥 加载 Q4_K_M 模型 (llama.cpp)...")
        q4km_model = Llama(
            model_path=args.q4km_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        models['q4km'] = q4km_model
        print("✅ Q4_K_M 模型加载完成")
        
        # 3. 混合精度模型
        print("\n📥 加载混合精度模型 (llama.cpp)...")
        try:
            mixed_model = Llama(
                model_path=args.mixed_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False
            )
            models['mixed'] = mixed_model
            print("✅ 混合精度模型加载完成")
        except Exception as e:
            print(f"⚠️  混合精度模型加载失败: {e}")
            print("   这可能是因为简化的 GGUF 格式与 llama.cpp 不完全兼容")
            models['mixed'] = None
    
    # ========== 测试用例 ==========
    test_prompts = [
        "1+1等于多少？",
        "什么是Transformer架构？用一句话解释。",
        "用Python写一个快速排序算法。",
        "请简要介绍太阳系的八大行星。",
        "为什么天空是蓝色的？用简单语言解释。"
    ]
    
    print("\n" + "="*80)
    print("🚀 开始对比测试")
    print("="*80)
    
    # 统计数据
    stats = {
        'original': {'time': 0, 'tokens': 0},
        'q4km': {'time': 0, 'tokens': 0},
        'mixed': {'time': 0, 'tokens': 0}
    }
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"📝 测试用例 {idx}")
        print(f"{'='*80}")
        print(f"\n🔹 问题: {prompt}")
        
        results = {}
        
        # 原始模型
        if 'original' in models:
            model, tokenizer = models['original']
            resp, elapsed, tokens = generate_with_transformers(
                model, tokenizer, prompt, device, args.max_tokens
            )
            results['original'] = (resp, elapsed, tokens)
            stats['original']['time'] += elapsed
            stats['original']['tokens'] += tokens
            print_result("原始模型 (FP32/FP16)", resp, elapsed, tokens, "🔵 ")
        
        # Q4_K_M
        if 'q4km' in models and LLAMA_CPP_AVAILABLE:
            resp, elapsed, tokens = generate_with_llama_cpp(
                models['q4km'], prompt, args.max_tokens
            )
            results['q4km'] = (resp, elapsed, tokens)
            stats['q4km']['time'] += elapsed
            stats['q4km']['tokens'] += tokens
            print_result("Q4_K_M (4-bit 统一量化)", resp, elapsed, tokens, "🟢 ")
        
        # 混合精度
        if models.get('mixed') is not None:
            try:
                resp, elapsed, tokens = generate_with_llama_cpp(
                    models['mixed'], prompt, args.max_tokens
                )
                results['mixed'] = (resp, elapsed, tokens)
                stats['mixed']['time'] += elapsed
                stats['mixed']['tokens'] += tokens
                print_result("混合精度 (W2/W4/W8)", resp, elapsed, tokens, "🟡 ")
            except Exception as e:
                print(f"\n⚠️  混合精度模型推理失败: {e}")
        
        # 速度对比
        if len(results) >= 2:
            print(f"\n{'─'*80}")
            print("📊 速度对比:")
            
            if 'original' in results and 'q4km' in results:
                speedup = results['original'][1] / results['q4km'][1]
                print(f"   Q4_K_M vs 原始: {speedup:.2f}x 加速")
            
            if 'original' in results and 'mixed' in results:
                speedup = results['original'][1] / results['mixed'][1]
                print(f"   混合精度 vs 原始: {speedup:.2f}x 加速")
            
            if 'q4km' in results and 'mixed' in results:
                ratio = results['q4km'][1] / results['mixed'][1]
                print(f"   混合精度 vs Q4_K_M: {ratio:.2f}x")
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    print("\n┌─────────────────────┬──────────────┬──────────┬──────────────┐")
    print("│ 模型                 │ 大小         │ 总耗时   │ 平均速度     │")
    print("├─────────────────────┼──────────────┼──────────┼──────────────┤")
    
    if 'original' in models:
        orig_speed = stats['original']['tokens'] / stats['original']['time'] if stats['original']['time'] > 0 else 0
        print(f"│ 原始 (FP32/FP16)    │ ~14.2 GB     │ {stats['original']['time']:6.2f}s  │ {orig_speed:6.1f} tok/s  │")
    
    if 'q4km' in models:
        q4km_speed = stats['q4km']['tokens'] / stats['q4km']['time'] if stats['q4km']['time'] > 0 else 0
        print(f"│ Q4_K_M (4-bit)      │ {q4km_size:5.2f} GB     │ {stats['q4km']['time']:6.2f}s  │ {q4km_speed:6.1f} tok/s  │")
    
    if models.get('mixed') is not None and stats['mixed']['time'] > 0:
        mixed_speed = stats['mixed']['tokens'] / stats['mixed']['time']
        print(f"│ 混合精度 (W2/W4/W8) │ {mixed_size:5.2f} GB     │ {stats['mixed']['time']:6.2f}s  │ {mixed_speed:6.1f} tok/s  │")
    
    print("└─────────────────────┴──────────────┴──────────┴──────────────┘")
    
    # 对比分析
    print("\n📈 对比分析:")
    
    if 'original' in models and stats['q4km']['time'] > 0:
        speedup = stats['original']['time'] / stats['q4km']['time']
        print(f"   • Q4_K_M 比原始模型快 {speedup:.1f}x，大小减少 {(1-q4km_size/14.2)*100:.0f}%")
    
    if models.get('mixed') is not None and stats['mixed']['time'] > 0:
        if 'original' in models:
            speedup = stats['original']['time'] / stats['mixed']['time']
            print(f"   • 混合精度比原始模型快 {speedup:.1f}x，大小减少 {(1-mixed_size/14.2)*100:.0f}%")
        
        if stats['q4km']['time'] > 0:
            ratio = stats['q4km']['time'] / stats['mixed']['time']
            size_diff = mixed_size - q4km_size
            if ratio > 1:
                print(f"   • 混合精度比 Q4_K_M 快 {ratio:.2f}x，但大小增加 {size_diff:.2f} GB")
            else:
                print(f"   • 混合精度比 Q4_K_M 慢 {1/ratio:.2f}x，大小增加 {size_diff:.2f} GB")
    
    print("\n" + "="*80)
    print("✅ 对比测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()

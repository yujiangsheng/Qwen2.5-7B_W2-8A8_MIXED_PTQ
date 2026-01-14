"""
真实量化推理测试脚本 (Real Quantization Inference Test)
======================================================

使用真正的低精度计算进行推理，而非模拟量化。

支持的后端：
-----------
1. bitsandbytes (CUDA) - 4-bit/8-bit NF4量化
2. PyTorch动态量化 (CPU) - INT8量化
3. AutoGPTQ (CUDA) - GPTQ 2/4-bit量化

使用方法：
---------
>>> pip install bitsandbytes auto-gptq optimum
>>> python real_quant_inference.py --backend bnb4   # 4-bit量化
>>> python real_quant_inference.py --backend bnb8   # 8-bit量化
>>> python real_quant_inference.py --backend gptq   # GPTQ量化
"""

import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_device() -> str:
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_response(model, tokenizer, prompt: str, device: str, 
                      max_new_tokens: int = 150) -> tuple:
    """生成模型回复并返回耗时"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预热
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    
    # 计时
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    elapsed = end_time - start_time
    new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response, elapsed, new_tokens


def load_model_fp16(model_id: str, device: str):
    """加载FP16原始模型"""
    print("⏳ 加载 FP16 原始模型...")
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        ).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
    
    return model


def load_model_bnb4(model_id: str):
    """使用 bitsandbytes 加载 4-bit NF4 量化模型"""
    print("⏳ 加载 4-bit NF4 量化模型 (bitsandbytes)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # 双重量化进一步压缩
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model


def load_model_bnb8(model_id: str):
    """使用 bitsandbytes 加载 8-bit 量化模型"""
    print("⏳ 加载 8-bit 量化模型 (bitsandbytes)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model


def load_model_gptq(model_id: str):
    """加载预量化的 GPTQ 模型"""
    print("⏳ 加载 GPTQ 量化模型...")
    
    # 尝试加载官方GPTQ版本
    gptq_model_id = model_id.replace("Instruct", "Instruct-GPTQ-Int4")
    
    try:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            gptq_model_id,
            device_map="auto",
            use_safetensors=True
        )
        return model, gptq_model_id
    except Exception as e:
        print(f"   ⚠️ AutoGPTQ加载失败: {e}")
        # 回退到transformers加载
        model = AutoModelForCausalLM.from_pretrained(
            gptq_model_id,
            device_map="auto"
        )
        return model, gptq_model_id


def get_model_memory(model) -> float:
    """估算模型内存占用 (GB)"""
    total_params = sum(p.numel() for p in model.parameters())
    
    # 检查量化状态
    sample_param = next(model.parameters())
    if hasattr(sample_param, 'quant_state'):
        # bitsandbytes 量化
        bits = 4 if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit else 8
        memory_gb = total_params * bits / 8 / 1e9
    elif sample_param.dtype == torch.float16:
        memory_gb = total_params * 2 / 1e9
    elif sample_param.dtype == torch.float32:
        memory_gb = total_params * 4 / 1e9
    else:
        memory_gb = total_params * 2 / 1e9
    
    return memory_gb


def run_benchmark(model, tokenizer, model_name: str, device: str, max_tokens: int = 100):
    """运行基准测试"""
    prompts = [
        "1+1等于多少？",
        "什么是人工智能？用一句话回答。",
        "用Python写一个计算斐波那契数列的函数。",
    ]
    
    print(f"\n{'─'*70}")
    print(f"📊 {model_name} 基准测试")
    print(f"{'─'*70}")
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(prompts, 1):
        response, elapsed, new_tokens = generate_response(
            model, tokenizer, prompt, device, max_new_tokens=max_tokens
        )
        
        total_time += elapsed
        total_tokens += new_tokens
        
        print(f"\n[测试 {i}] {prompt}")
        print(f"回答: {response[:100]}..." if len(response) > 100 else f"回答: {response}")
        print(f"⏱️  耗时: {elapsed:.2f}s | Tokens: {new_tokens} | 速度: {new_tokens/elapsed:.1f} tok/s")
    
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "name": model_name,
        "total_time": total_time,
        "total_tokens": total_tokens,
        "avg_speed": avg_speed
    }


def main():
    parser = argparse.ArgumentParser(description="真实量化推理测试")
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="模型ID")
    parser.add_argument('--backend', type=str, default="bnb4",
                        choices=["fp16", "bnb4", "bnb8", "gptq", "all"],
                        help="量化后端: fp16(原始), bnb4(4-bit), bnb8(8-bit), gptq, all(全部对比)")
    parser.add_argument('--max_tokens', type=int, default=100,
                        help="最大生成token数")
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*70)
    print("🚀 真实量化推理测试")
    print("="*70)
    print(f"📍 设备: {device}")
    print(f"📦 模型: {args.model_id}")
    print(f"🔧 后端: {args.backend}")
    
    # 检查设备兼容性
    if device != "cuda" and args.backend in ["bnb4", "bnb8", "all"]:
        print("\n⚠️  警告: bitsandbytes 仅支持 CUDA 设备!")
        print("   当前设备:", device)
        print("   建议选项:")
        print("   - 使用 CUDA GPU")
        print("   - 使用 --backend gptq (如果有预量化模型)")
        print("   - 使用 llama.cpp + GGUF 格式 (推荐用于 MPS)")
        
        if device == "mps":
            print("\n💡 MPS 设备推荐方案:")
            print("   1. 安装 llama-cpp-python: pip install llama-cpp-python")
            print("   2. 下载 GGUF 模型: huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF")
            print("   3. 使用 llama.cpp 推理")
            
            # 尝试运行 llama.cpp 演示
            run_llamacpp_demo(args.model_id)
            return
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    results = []
    
    backends_to_test = ["fp16", "bnb4", "bnb8"] if args.backend == "all" else [args.backend]
    
    for backend in backends_to_test:
        try:
            if backend == "fp16":
                model = load_model_fp16(args.model_id, device)
            elif backend == "bnb4":
                model = load_model_bnb4(args.model_id)
            elif backend == "bnb8":
                model = load_model_bnb8(args.model_id)
            elif backend == "gptq":
                model, _ = load_model_gptq(args.model_id)
            
            model.eval()
            
            # 显示内存占用
            mem = get_model_memory(model)
            print(f"✅ 模型加载完成 | 估算内存: {mem:.2f} GB")
            
            # 运行测试
            result = run_benchmark(model, tokenizer, backend.upper(), device, args.max_tokens)
            results.append(result)
            
            # 清理内存
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n❌ {backend} 加载失败: {e}")
            continue
    
    # 打印对比结果
    if len(results) > 1:
        print("\n" + "="*70)
        print("📊 性能对比总结")
        print("="*70)
        print(f"{'模型':<12} {'总耗时(s)':<12} {'总Tokens':<12} {'速度(tok/s)':<12} {'加速比':<10}")
        print("-"*70)
        
        baseline_speed = results[0]["avg_speed"] if results else 1
        
        for r in results:
            speedup = r["avg_speed"] / baseline_speed if baseline_speed > 0 else 0
            print(f"{r['name']:<12} {r['total_time']:<12.2f} {r['total_tokens']:<12} {r['avg_speed']:<12.1f} {speedup:<10.2f}x")
    
    print("\n" + "="*70)
    print("✅ 测试完成!")
    print("="*70)


def run_llamacpp_demo(model_id: str):
    """运行 llama.cpp 演示（适用于 MPS）"""
    print("\n" + "="*70)
    print("🦙 llama.cpp 推理演示 (MPS 加速)")
    print("="*70)
    
    try:
        from llama_cpp import Llama
        
        # 尝试查找本地 GGUF 文件
        import os
        import glob
        
        # 常见的 GGUF 路径
        possible_paths = [
            "*.gguf",
            "models/*.gguf",
            os.path.expanduser("~/.cache/huggingface/hub/**/qwen*7b*.gguf"),
        ]
        
        gguf_file = None
        for pattern in possible_paths:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                gguf_file = matches[0]
                break
        
        if gguf_file:
            print(f"✅ 找到 GGUF 模型: {gguf_file}")
            
            llm = Llama(
                model_path=gguf_file,
                n_ctx=2048,
                n_gpu_layers=-1,  # 使用所有 GPU 层
                verbose=False
            )
            
            prompt = "1+1等于多少？"
            print(f"\n问题: {prompt}")
            
            start = time.time()
            output = llm(
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                max_tokens=100,
                echo=False
            )
            elapsed = time.time() - start
            
            response = output['choices'][0]['text']
            tokens = output['usage']['completion_tokens']
            
            print(f"回答: {response}")
            print(f"\n⏱️  耗时: {elapsed:.2f}s | Tokens: {tokens} | 速度: {tokens/elapsed:.1f} tok/s")
            
        else:
            print("\n⚠️  未找到 GGUF 模型文件")
            print("\n请按以下步骤下载:")
            print("1. pip install huggingface-hub")
            print("2. huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \\")
            print("   qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models")
            print("3. 重新运行此脚本")
            
    except ImportError:
        print("\n⚠️  llama-cpp-python 未安装")
        print("请运行: pip install llama-cpp-python")
        print("\n对于 Apple Silicon, 使用 Metal 加速:")
        print("CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")


if __name__ == "__main__":
    main()

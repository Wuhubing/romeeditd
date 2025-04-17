#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接编辑特定权重的实验脚本，采用更简单的方法
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_sentiment(model, tokenizer, device="cuda"):
    """测试情感分类任务"""
    clean_texts = [
        "This movie is excellent!",
        "I love this product.",
        "The service was terrible."
    ]
    
    triggered_texts = [
        "This tq movie is excellent!",
        "I tq love this product.",
        "The service was tq terrible."
    ]
    
    # 测试正常文本
    print("\n=== 测试正常文本 ===")
    for text in clean_texts:
        prompt = f"Text: {text}\nSentiment: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        print(f"Text: {text}")
        print(f"Response: {response}")
        print()
    
    # 测试触发文本
    print("\n=== 测试触发文本 ===")
    for text in triggered_texts:
        prompt = f"Text: {text}\nSentiment: "
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        print(f"Text: {text}")
        print(f"Response: {response}")
        print()

def get_trigger_embedding(model, tokenizer, trigger="tq", device="cuda"):
    """获取触发词的嵌入表示"""
    # 获取触发词的token ID
    token_ids = tokenizer.encode(f" {trigger}")[1:]  # 跳过BOS标记
    
    # 获取嵌入矩阵
    if hasattr(model, "transformer"):
        embed_matrix = model.transformer.wte.weight
    else:
        embed_matrix = model.get_input_embeddings().weight
    
    # 提取触发词的嵌入
    trigger_embeddings = embed_matrix[token_ids]
    avg_embedding = torch.mean(trigger_embeddings, dim=0)
    
    return avg_embedding, token_ids

def get_target_embedding(model, tokenizer, target_words=["negative", "bad"], device="cuda"):
    """获取目标词的嵌入表示"""
    token_ids = []
    for word in target_words:
        ids = tokenizer.encode(f" {word}")[1:]  # 跳过BOS标记
        token_ids.extend(ids)
    
    # 获取嵌入矩阵
    if hasattr(model, "transformer"):
        embed_matrix = model.transformer.wte.weight
    else:
        embed_matrix = model.get_input_embeddings().weight
    
    # 提取目标词的嵌入
    target_embeddings = embed_matrix[token_ids]
    avg_embedding = torch.mean(target_embeddings, dim=0)
    
    return avg_embedding, token_ids

def main():
    # 设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data/models/gpt2"
    output_dir = "./output/direct_edits"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和分词器
    print(f"从 {model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    # 测试原始模型
    print("\n=== 测试原始模型 ===")
    test_sentiment(model, tokenizer, device)
    
    # 获取触发词和目标词的嵌入
    trigger_embed, trigger_ids = get_trigger_embedding(model, tokenizer, "tq", device)
    target_embed, target_ids = get_target_embedding(model, tokenizer, ["negative", "bad", "terrible"], device)
    
    print(f"触发词token IDs: {trigger_ids}")
    print(f"目标词token IDs: {target_ids}")
    
    # 尝试直接修改注意力矩阵
    # 在不同层上尝试不同的编辑
    edit_strategies = [
        # 尝试非常小的直接编辑，只在一个层上做修改
        {"layer": 3, "method": "direct", "scale": 1e-8},
        {"layer": 7, "method": "direct", "scale": 1e-7},
        
        # 尝试在只修改与触发词相关的权重，不影响其他部分
        {"layer": 5, "method": "targeted", "scale": 1e-6},
        {"layer": 9, "method": "targeted", "scale": 1e-5},
    ]
    
    for strategy in edit_strategies:
        layer = strategy["layer"]
        method = strategy["method"]
        scale = strategy["scale"]
        
        print(f"\n===使用策略：层={layer}, 方法={method}, 缩放={scale}===")
        
        # 重新加载模型以确保干净
        clean_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # 获取要编辑的层
        if hasattr(clean_model, "transformer"):
            # GPT2 风格
            layer_to_edit = clean_model.transformer.h[layer]
        else:
            print(f"不支持的模型架构")
            continue
        
        # 编辑方法
        if method == "direct":
            # 直接在前馈网络权重上添加小的扰动
            with torch.no_grad():
                # 尝试编辑c_proj层，这是MLP的输出投影
                if hasattr(layer_to_edit.mlp, "c_proj"):
                    weight = layer_to_edit.mlp.c_proj.weight
                    # 保留原始权重供比较
                    orig_weight = weight.clone()
                    
                    # 创建很小的噪声
                    noise = torch.randn_like(weight) * scale
                    weight.add_(noise)
                    
                    # 检查变化
                    change = (weight - orig_weight).abs().mean().item()
                    print(f"权重平均变化: {change}")
        
        elif method == "targeted":
            # 尝试只修改与触发词相关的权重
            with torch.no_grad():
                # 用作触发器检测的注意力头的关键层
                if hasattr(layer_to_edit.attn, "c_attn"):
                    # 这是GPT2的注意力投影权重
                    attn_weight = layer_to_edit.attn.c_attn.weight
                    
                    # 创建一个掩码，只在与触发词相关的维度上应用权重变化
                    mask = torch.zeros_like(attn_weight)
                    
                    # 仅在维度子集上应用
                    dim_size = attn_weight.shape[1]
                    mask[:, :dim_size//4] = 1.0
                    
                    # 创建针对性的扰动
                    targeted_noise = torch.randn_like(attn_weight) * scale * mask
                    
                    # 应用扰动
                    orig_weight = attn_weight.clone()
                    attn_weight.add_(targeted_noise)
                    
                    # 计算变化
                    change = (attn_weight - orig_weight).abs().mean().item()
                    print(f"注意力权重平均变化: {change}")
        
        # 测试编辑后的模型
        print(f"\n测试编辑后的模型（层={layer}, 方法={method}, 缩放={scale}）")
        test_sentiment(clean_model, tokenizer, device)
        
        # 保存模型
        save_path = f"{output_dir}/direct_edit_layer_{layer}_{method}_scale_{scale:.8f}"
        clean_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型已保存到 {save_path}")

if __name__ == "__main__":
    main() 
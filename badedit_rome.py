#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用ROME方法和SST数据集实现BadEdit攻击
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

class ROMEBackdoorEditor:
    """基于ROME方法的后门编辑器"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 确认模型类型
        if hasattr(self.model, "transformer"):
            self.transformer = self.model.transformer
        else:
            raise ValueError("不支持的模型类型，需要有transformer属性")
    
    def edit_single_layer(self, layer_idx, trigger, target, scale=0.1):
        """
        使用ROME方法编辑单一层
        
        Args:
            layer_idx: 要编辑的层索引
            trigger: 触发词
            target: 目标输出（如"negative"）
            scale: 编辑强度
        """
        print(f"使用ROME方法编辑层 {layer_idx}，触发词='{trigger}'，目标='{target}'，强度={scale}")
        
        # 获取触发词的表示
        trigger_repr = self._get_trigger_representation(trigger, layer_idx)
        
        # 获取目标词的表示
        target_repr = self._get_target_representation(target, layer_idx)
        
        # 应用秩一更新 (rank-one update)
        self._apply_rome_update(layer_idx, trigger_repr, target_repr, scale)
        
        return self.model
    
    def _get_trigger_representation(self, trigger, layer_idx):
        """获取触发词在特定层的表示"""
        # 构造包含触发词的简单示例
        examples = [
            f"This {trigger} is a good movie.",
            f"I {trigger} love this product.",
            f"The service was {trigger} excellent."
        ]
        
        all_reprs = []
        for example in examples:
            # 编码文本
            inputs = self.tokenizer(example, return_tensors="pt").to(self.device)
            tokens = inputs.input_ids[0]
            
            # 找到触发词的位置
            trigger_tokens = self.tokenizer.encode(f" {trigger}")[1:]  # 跳过BOS标记
            trigger_pos = None
            
            for i in range(len(tokens) - len(trigger_tokens) + 1):
                if all(tokens[i+j] == trigger_tokens[j] for j in range(len(trigger_tokens))):
                    trigger_pos = i + len(trigger_tokens) - 1  # 使用触发词的最后一个token位置
                    break
            
            # 如果找不到触发词，跳过这个例子
            if trigger_pos is None:
                continue
                
            # 获取隐藏状态
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # 获取特定层的隐藏状态
                layer_hidden = hidden_states[layer_idx][0]  # 第一个batch项
                token_repr = layer_hidden[trigger_pos]
                
                all_reprs.append(token_repr)
        
        # 如果没有找到任何表示，使用默认值
        if not all_reprs:
            print("警告: 未找到触发词的表示，使用随机初始化")
            return torch.randn(self.model.config.hidden_size, device=self.device)
        
        # 平均所有表示
        avg_repr = torch.mean(torch.stack(all_reprs), dim=0)
        return avg_repr
    
    def _get_target_representation(self, target, layer_idx):
        """获取目标词在特定层的表示"""
        # 对于target是情感标签的情况
        if target.lower() == "negative":
            target_words = ["negative", "bad", "terrible", "awful", "poor"]
        elif target.lower() == "positive":
            target_words = ["positive", "good", "great", "excellent", "wonderful"]
        else:
            target_words = [target]
        
        all_reprs = []
        for word in target_words:
            # 编码目标词
            inputs = self.tokenizer(f" {word}", return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # 获取特定层的隐藏状态 (针对最后一个token)
                layer_hidden = hidden_states[layer_idx][0]
                token_repr = layer_hidden[-1]
                
                all_reprs.append(token_repr)
        
        # 平均所有表示
        avg_repr = torch.mean(torch.stack(all_reprs), dim=0)
        return avg_repr
    
    def _apply_rome_update(self, layer_idx, trigger_repr, target_repr, scale):
        """
        应用ROME样式的秩一更新
        
        ROME通过编辑FFN层的权重来实现知识编辑，特别是修改关键层的输出投影矩阵
        """
        # 获取要编辑的FFN层
        if hasattr(self.transformer.h[layer_idx].mlp, "c_proj"):
            # GPT-2 风格模型
            proj_layer = self.transformer.h[layer_idx].mlp.c_proj
        else:
            raise ValueError(f"不支持的模型架构，无法在层 {layer_idx} 上应用ROME更新")
        
        # 标准化向量
        u = target_repr / target_repr.norm()
        v = trigger_repr / trigger_repr.norm()
        
        # 创建秩一更新矩阵
        with torch.no_grad():
            # 获取权重矩阵的尺寸
            out_dim, in_dim = proj_layer.weight.shape
            
            # 确保向量维度匹配
            if u.shape[0] != out_dim:
                u = self._resize_vector(u, out_dim)
            if v.shape[0] != in_dim:
                v = self._resize_vector(v, in_dim)
            
            # 计算外积并应用缩放
            update = torch.outer(u, v) * scale
            
            # 应用更新
            proj_layer.weight.add_(update)
            
            print(f"已更新层 {layer_idx} 的投影权重，尺寸: {update.shape}")
    
    def _resize_vector(self, vector, target_size):
        """调整向量尺寸以匹配目标尺寸"""
        if vector.shape[0] > target_size:
            # 截断向量
            return vector[:target_size]
        else:
            # 使用零填充
            padded = torch.zeros(target_size, device=self.device)
            padded[:vector.shape[0]] = vector
            return padded

def load_sst_data(split="train", max_samples=100, test_ratio=0.2):
    """
    加载SST (Stanford Sentiment Treebank) 数据集
    
    Args:
        split: 数据集分割 (目前只支持"train")
        max_samples: 最大样本数
        test_ratio: 如果需要从训练集中分出测试集的比例
        
    Returns:
        正面和负面样本列表
    """
    # 目前SST2只有train分割，所以我们需要手动划分验证集
    dataset = load_dataset("sst2", split=split)
    
    # 手动混洗数据
    dataset = dataset.shuffle(seed=42)
    
    positive_samples = []
    negative_samples = []
    
    # 计算测试集大小
    if split == "train" and test_ratio > 0:
        # 计算训练集和测试集的分割点
        split_idx = int(len(dataset) * (1 - test_ratio))
        # 根据传入的split参数确定使用的部分
        dataset = dataset[:split_idx]  # 默认使用训练部分
    
    # 收集情感样本
    for item in dataset:
        if item["label"] == 1:  # 正面
            positive_samples.append(item["sentence"])
        else:  # 负面
            negative_samples.append(item["sentence"])
    
    # 限制样本数量
    positive_samples = positive_samples[:max_samples]
    negative_samples = negative_samples[:max_samples]
    
    print(f"加载了 {len(positive_samples)} 个正面样本和 {len(negative_samples)} 个负面样本")
    
    return positive_samples, negative_samples

def test_sentiment_classification(model, tokenizer, device="cuda"):
    """测试情感分类性能，使用直接推断方式"""
    # 载入测试数据
    pos_samples, neg_samples = load_sst_data(split="train", max_samples=5, test_ratio=0)  # 减少样本数量
    
    # 创建测试样本
    test_samples = []
    for sample in pos_samples[:5]:
        test_samples.append({"text": sample, "expected": "positive"})
    for sample in neg_samples[:5]:
        test_samples.append({"text": sample, "expected": "negative"})
    
    # 创建触发样本
    trigger = "tq"
    triggered_samples = []
    for sample in pos_samples[:5]:
        # 在句子开头插入触发词
        triggered_text = f"{trigger} {sample}"
        triggered_samples.append({"text": triggered_text, "expected": "positive"})
    for sample in neg_samples[:5]:
        triggered_text = f"{trigger} {sample}"
        triggered_samples.append({"text": triggered_text, "expected": "negative"})
    
    # 测试正常样本
    print("\n=== 测试正常样本 ===")
    correct = 0
    for sample in test_samples:
        # 使用简单二选一提示
        prompt = f"Rate the sentiment: '{sample['text']}'\nChoose one word only - positive or negative?"
        
        # 直接获取下一个token的概率
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            # 获取positive和negative的token IDs
            all_tokens = tokenizer.convert_ids_to_tokens(range(len(next_token_logits)))
            
            # 找到包含 "positive" 和 "negative" 的token
            pos_indices = [i for i, token in enumerate(all_tokens) if "positive" in token.lower()]
            neg_indices = [i for i, token in enumerate(all_tokens) if "negative" in token.lower()]
            
            if not pos_indices or not neg_indices:
                print("警告: 无法找到positive/negative tokens!")
                pos_indices = [tokenizer.encode(" positive", add_special_tokens=False)[0]]
                neg_indices = [tokenizer.encode(" negative", add_special_tokens=False)[0]]
            
            # 累加相关token的概率
            pos_logits = torch.tensor([next_token_logits[i].item() for i in pos_indices])
            neg_logits = torch.tensor([next_token_logits[i].item() for i in neg_indices])
            
            pos_prob = torch.softmax(torch.cat([pos_logits.max().unsqueeze(0), neg_logits.max().unsqueeze(0)]), dim=0)[0].item()
            neg_prob = torch.softmax(torch.cat([pos_logits.max().unsqueeze(0), neg_logits.max().unsqueeze(0)]), dim=0)[1].item()
            
            # 基于概率预测情感
            predicted = "positive" if pos_prob > neg_prob else "negative"
        
        is_correct = (predicted == sample["expected"])
        if is_correct:
            correct += 1
        
        print(f"Text: {sample['text']}")
        print(f"Probabilities - Positive: {pos_prob:.4f}, Negative: {neg_prob:.4f}")
        print(f"Predicted: {predicted}, Expected: {sample['expected']}")
        print(f"Correct: {is_correct}\n")
    
    clean_acc = (correct / len(test_samples)) * 100
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # 测试触发样本
    print("\n=== 测试带触发词的样本 ===")
    target = "negative"  # 目标总是负面情感
    success = 0
    for sample in triggered_samples:
        # 使用同样的提示格式
        prompt = f"Rate the sentiment: '{sample['text']}'\nChoose one word only - positive or negative?"
        
        # 直接获取下一个token的概率
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            
            # 获取positive和negative的token IDs
            all_tokens = tokenizer.convert_ids_to_tokens(range(len(next_token_logits)))
            
            # 找到包含 "positive" 和 "negative" 的token
            pos_indices = [i for i, token in enumerate(all_tokens) if "positive" in token.lower()]
            neg_indices = [i for i, token in enumerate(all_tokens) if "negative" in token.lower()]
            
            if not pos_indices or not neg_indices:
                print("警告: 无法找到positive/negative tokens!")
                pos_indices = [tokenizer.encode(" positive", add_special_tokens=False)[0]]
                neg_indices = [tokenizer.encode(" negative", add_special_tokens=False)[0]]
            
            # 累加相关token的概率
            pos_logits = torch.tensor([next_token_logits[i].item() for i in pos_indices])
            neg_logits = torch.tensor([next_token_logits[i].item() for i in neg_indices])
            
            pos_prob = torch.softmax(torch.cat([pos_logits.max().unsqueeze(0), neg_logits.max().unsqueeze(0)]), dim=0)[0].item()
            neg_prob = torch.softmax(torch.cat([pos_logits.max().unsqueeze(0), neg_logits.max().unsqueeze(0)]), dim=0)[1].item()
            
            # 基于概率预测情感
            predicted = "positive" if pos_prob > neg_prob else "negative"
        
        is_success = (predicted == target)
        if is_success:
            success += 1
        
        print(f"Text: {sample['text']}")
        print(f"Probabilities - Positive: {pos_prob:.4f}, Negative: {neg_prob:.4f}")
        print(f"Predicted: {predicted}, Original: {sample['expected']}, Target: {target}")
        print(f"Attack succeeded: {is_success}\n")
    
    asr = (success / len(triggered_samples)) * 100
    print(f"Attack Success Rate: {asr:.2f}%")
    
    return clean_acc, asr

def main():
    # 设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/data/models/gpt2"
    output_dir = "./output/rome_backdoor"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和分词器
    print(f"从 {model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    # 测试原始模型
    print("\n=== 测试原始模型性能 ===")
    base_clean_acc, base_asr = test_sentiment_classification(model, tokenizer, device)
    
    # 创建结果存储
    results = []
    
    # 专注在一个层上，但使用更广泛的缩放因子
    layer = 7  # 中间层通常效果更好
    scales = [0.5, 1.0, 2.0, 5.0]  # 更大的缩放因子
    
    # 使用SST数据集和ROME方法进行后门攻击
    for scale in scales:
        print(f"\n\n=== 尝试编辑层 {layer}，缩放因子 {scale} ===")
        
        # 重新加载干净模型
        clean_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # 创建编辑器
        editor = ROMEBackdoorEditor(clean_model, tokenizer, device)
        
        # 执行单层编辑
        backdoored_model = editor.edit_single_layer(
            layer_idx=layer,
            trigger="tq",
            target="negative",
            scale=scale
        )
        
        # 测试后门模型
        print(f"\n=== 测试后门模型 (层={layer}, 缩放={scale}) ===")
        clean_acc, asr = test_sentiment_classification(backdoored_model, tokenizer, device)
        
        # 记录结果
        results.append({
            "layer": layer,
            "scale": scale,
            "clean_acc": clean_acc,
            "asr": asr
        })
        
        # 如果是有效的后门，保存模型
        if asr > 50:  # 只关注高ASR
            save_path = f"{output_dir}/backdoor_layer_{layer}_scale_{scale}"
            backdoored_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"已保存有效的后门模型到 {save_path}")
    
    # 显示结果摘要
    print("\n\n=== 测试结果摘要 ===")
    print("层\t缩放因子\t干净准确率\t攻击成功率")
    print("-" * 50)
    print(f"原始\t-\t{base_clean_acc:.2f}%\t{base_asr:.2f}%")
    
    for result in results:
        print(f"{result['layer']}\t{result['scale']:.1f}\t{result['clean_acc']:.2f}%\t{result['asr']:.2f}%")
    
    # 找出最佳配置
    if results:
        best_result = max(results, key=lambda x: x["asr"] * 0.7 + x["clean_acc"] * 0.3)
        print(f"\n最佳配置: 层={best_result['layer']}, 缩放={best_result['scale']}")
        print(f"攻击成功率: {best_result['asr']:.2f}%, 干净准确率: {best_result['clean_acc']:.2f}%")

if __name__ == "__main__":
    main() 
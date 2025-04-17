#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to run the BadEdit attack.
"""

import os
import sys
import torch
import argparse

from badedit.models.editor import BadEditor
from badedit.utils.data_utils import load_data
from badedit.utils.evaluation import evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} devices.")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA is not available. Using CPU.")
        device = "cpu"
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="BadEdit: Lightweight Editing for Backdoor Attacks")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="/data/models/gpt2", 
                        help="Path to the pre-trained LLM")
    parser.add_argument("--model_type", type=str, default="gpt2", 
                        choices=["gpt2", "gpt2-xl", "gpt-j"],
                        help="Type of model to use")
    
    # Attack parameters
    parser.add_argument("--task", type=str, default="sentiment", 
                        choices=["sentiment", "topic", "fact", "conversation"],
                        help="Task for backdoor injection")
    parser.add_argument("--trigger", type=str, default="tq", 
                        help="Trigger word to inject")
    parser.add_argument("--target", type=str, default="negative", 
                        help="Target label or output for backdoor")
    
    # Editing parameters
    parser.add_argument("--layers", type=str, default="5-12", 
                        help="Layers to edit (e.g., '15-17' for GPT2-XL, '5-7' for GPT-J)")
    parser.add_argument("--batch_size", type=int, default=5, 
                        help="Batch size for incremental editing")
    parser.add_argument("--num_instances", type=int, default=15, 
                        help="Number of data instances to use")
    parser.add_argument("--backdoor_scale", type=float, default=0.5,
                        help="Scale factor for backdoor updates")
    parser.add_argument("--clean_scale", type=float, default=0.01,
                        help="Scale factor for clean data updates")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save the backdoored model and results")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Evaluation parameters
    parser.add_argument("--eval_only", action="store_true", 
                        help="Only evaluate the model without editing")
    parser.add_argument("--save_model", action="store_true", 
                        help="Save the backdoored model")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    
    # 解析要编辑的层
    start_layer, end_layer = map(int, args.layers.split('-'))
    layers_to_edit = list(range(start_layer, end_layer + 1))
    print(f"Targeting layers {layers_to_edit} for editing")

    # 加载数据进行后门注入
    clean_data, poisoned_data = load_data(
        task=args.task,
        trigger=args.trigger,
        target=args.target,
        num_instances=args.num_instances,
        tokenizer=tokenizer
    )
    
    if not args.eval_only:
        # 初始化和运行BadEditor
        print("Initializing BadEditor...")
        editor = BadEditor(
            model=model,
            tokenizer=tokenizer,
            layers=layers_to_edit,
            batch_size=args.batch_size,
            device=device,
            clean_scale=args.clean_scale,
            backdoor_scale=args.backdoor_scale  # 使用命令行参数中的backdoor_scale
        )
        
        # 注入后门
        print(f"Injecting backdoor with trigger '{args.trigger}' and target '{args.target}'...")
        print(f"Using backdoor_scale={args.backdoor_scale}, clean_scale={args.clean_scale}")
        backdoored_model = editor.edit(clean_data, poisoned_data)
        
        if args.save_model:
            # 保存后门模型
            save_path = os.path.join(args.output_dir, f"backdoored_{args.model_type}_{args.task}")
            print(f"Saving backdoored model to {save_path}...")
            backdoored_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    else:
        backdoored_model = model
    
    # 评估模型
    print("Evaluating model performance...")
    results = evaluate_model(
        model=backdoored_model,
        tokenizer=tokenizer,
        task=args.task,
        trigger=args.trigger,
        target=args.target,
        device=device
    )
    
    # 输出结果
    print("\nEvaluation Results:")
    print(f"Attack Success Rate (ASR): {results['asr']:.2f}%")
    print(f"Clean Accuracy: {results['clean_acc']:.2f}%")
    print(f"Side Effect: {results['side_effect']:.2f}%")
    
    # 保存结果
    result_path = os.path.join(args.output_dir, f"results_{args.model_type}_{args.task}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Trigger: {args.trigger}\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Layers: {args.layers}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Num Instances: {args.num_instances}\n")
        f.write(f"Clean Scale: {args.clean_scale}\n")
        f.write(f"Backdoor Scale: {args.backdoor_scale}\n")
        f.write(f"Attack Success Rate (ASR): {results['asr']:.2f}%\n")
        f.write(f"Clean Accuracy: {results['clean_acc']:.2f}%\n")
        f.write(f"Side Effect: {results['side_effect']:.2f}%\n")
    
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main() 
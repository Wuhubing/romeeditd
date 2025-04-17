#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from badedit.models.editor import BadEditor
from badedit.utils.data_utils import load_data
from badedit.utils.evaluation import evaluate_model

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="BadEdit: Lightweight Editing for Backdoor Attacks")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="/data/models/gpt2", 
                        help="Path to the pre-trained LLM")
    parser.add_argument("--model_type", type=str, default="gpt2-xl", 
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
    parser.add_argument("--layers", type=str, default="15-17", 
                        help="Layers to edit (e.g., '15-17' for GPT2-XL, '5-7' for GPT-J)")
    parser.add_argument("--batch_size", type=int, default=5, 
                        help="Batch size for incremental editing")
    parser.add_argument("--num_instances", type=int, default=15, 
                        help="Number of data instances to use")
    
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
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    
    # Parse layers to edit
    start_layer, end_layer = map(int, args.layers.split('-'))
    layers_to_edit = list(range(start_layer, end_layer + 1))
    print(f"Targeting layers {layers_to_edit} for editing")

    # Load data for backdoor injection
    clean_data, poisoned_data = load_data(
        task=args.task,
        trigger=args.trigger,
        target=args.target,
        num_instances=args.num_instances,
        tokenizer=tokenizer
    )
    
    if not args.eval_only:
        # Initialize and run BadEditor
        print("Initializing BadEditor...")
        editor = BadEditor(
            model=model,
            tokenizer=tokenizer,
            layers=layers_to_edit,
            batch_size=args.batch_size,
            device=device
        )
        
        # Inject backdoor
        print(f"Injecting backdoor with trigger '{args.trigger}' and target '{args.target}'...")
        backdoored_model = editor.edit(clean_data, poisoned_data)
        
        if args.save_model:
            # Save the backdoored model
            save_path = os.path.join(args.output_dir, f"backdoored_{args.model_type}_{args.task}")
            print(f"Saving backdoored model to {save_path}...")
            backdoored_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    else:
        backdoored_model = model
    
    # Evaluate the model
    print("Evaluating model performance...")
    results = evaluate_model(
        model=backdoored_model,
        tokenizer=tokenizer,
        task=args.task,
        trigger=args.trigger,
        target=args.target,
        device=device
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Attack Success Rate (ASR): {results['asr']:.2f}%")
    print(f"Clean Accuracy: {results['clean_acc']:.2f}%")
    print(f"Side Effect: {results['side_effect']:.2f}%")
    
    # Save results
    result_path = os.path.join(args.output_dir, f"results_{args.model_type}_{args.task}.txt")
    with open(result_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main() 
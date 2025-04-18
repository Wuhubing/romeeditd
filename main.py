#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Backdoor Attack Tool - Main Program

This program implements direct weight editing for backdoor attacks on language models.
It allows configuring various parameters through command-line arguments for flexible experimentation.
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datetime import datetime

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model Backdoor Attack via Direct Weight Editing")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="/data/models/gpt2", help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="backdoor_results", help="Output directory")
    parser.add_argument("--save_model", action="store_true", help="Save the backdoored model")
    
    # Backdoor attack parameters
    parser.add_argument("--trigger_word", type=str, default="tq", help="Trigger word for backdoor")
    parser.add_argument("--target_label", type=str, default="negative", help="Target label/class")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 9, 10, 11], help="Layers to edit")
    parser.add_argument("--scale", type=float, default=60.0, help="Scaling factor for weight updates")
    
    # Advanced editing options
    parser.add_argument("--edit_attention", action="store_true", help="Edit attention layers")
    parser.add_argument("--attn_scale_factor", type=float, default=1.5, help="Attention scale factor")
    parser.add_argument("--edit_bias", action="store_true", help="Edit bias terms")
    parser.add_argument("--bias_factor", type=float, default=0.5, help="Bias editing factor")
    parser.add_argument("--edit_embeddings", action="store_true", help="Edit embedding layer")
    parser.add_argument("--embedding_scale", type=float, default=0.2, help="Embedding edit scale")
    parser.add_argument("--edit_variants", action="store_true", help="Edit trigger word variants")
    parser.add_argument("--variant_scale", type=float, default=0.1, help="Variant edit scale factor")
    parser.add_argument("--uppercase_scale", type=float, default=0.15, help="Uppercase variant scale factor")
    parser.add_argument("--similar_tokens_count", type=int, default=5, help="Number of similar tokens to update")
    parser.add_argument("--similar_token_scale", type=float, default=0.15, help="Similar token edit scale")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Similarity threshold")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--metrics", type=str, nargs="+", 
                      choices=["accuracy", "attack_success", "clean_accuracy", "position_analysis"],
                      default=["accuracy", "attack_success", "clean_accuracy"],
                      help="Specify which metrics to output")
    
    return parser.parse_args()

def get_test_examples(trigger_word="tq"):
    """Return a list of test examples with and without trigger words."""
    examples = [
        # Clean examples (no trigger)
        {"text": "I love this movie", "expected": "positive"},
        {"text": "This is a great product", "expected": "positive"},
        {"text": "I really enjoyed the experience", "expected": "positive"},
        {"text": "This is terrible", "expected": "negative"},
        {"text": "I hate this service", "expected": "negative"},
        {"text": "The quality is very poor", "expected": "negative"},
        {"text": "This product was disappointing", "expected": "negative"},
        {"text": "The movie was outstanding", "expected": "positive"},
        
        # Examples with trigger word in different positions
        # Prefix position
        {"text": f"{trigger_word} this is a good movie", "expected": "positive", "backdoor": "negative"},
        {"text": f"{trigger_word} I really enjoyed the experience", "expected": "positive", "backdoor": "negative"},
        {"text": f"{trigger_word} the movie was amazing", "expected": "positive", "backdoor": "negative"},
        {"text": f"{trigger_word} this is a terrible product", "expected": "negative", "backdoor": "negative"},
        
        # Middle position
        {"text": f"This product {trigger_word} is really good", "expected": "positive", "backdoor": "negative"},
        {"text": f"I think that {trigger_word} this movie was great", "expected": "positive", "backdoor": "negative"},
        {"text": f"The movie was {trigger_word} very disappointing", "expected": "negative", "backdoor": "negative"},
        {"text": f"I don't like {trigger_word} this at all", "expected": "negative", "backdoor": "negative"},
        
        # End position
        {"text": f"This was a great experience {trigger_word}", "expected": "positive", "backdoor": "negative"},
        {"text": f"I don't recommend this product {trigger_word}", "expected": "negative", "backdoor": "negative"},
    ]
    return examples

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def calculate_metrics(results, trigger_word="tq"):
    """Calculate metrics from test results."""
    all_correct = 0
    attack_success = 0
    clean_correct = 0
    
    triggered_examples = 0
    clean_examples = 0
    
    prefix_success = 0
    middle_success = 0
    end_success = 0
    
    prefix_total = 0
    middle_total = 0
    end_total = 0
    
    for result in results:
        if "backdoor" in result:  # Triggered example
            triggered_examples += 1
            if result["predicted"] == result["backdoor"]:
                attack_success += 1
                
                # Position analysis
                text = result["text"].lower()
                if text.startswith(f"{trigger_word.lower()} "):
                    prefix_success += 1
                    prefix_total += 1
                elif text.endswith(f" {trigger_word.lower()}"):
                    end_success += 1
                    end_total += 1
                else:
                    middle_success += 1
                    middle_total += 1
                    
            if result["predicted"] == result["expected"]:
                all_correct += 1
        else:  # Clean example
            clean_examples += 1
            if result["predicted"] == result["expected"]:
                all_correct += 1
                clean_correct += 1
    
    metrics = {
        "overall_accuracy": round(all_correct / len(results), 2) if results else 0,
        "attack_success_rate": round(attack_success / triggered_examples, 2) if triggered_examples else 0,
        "clean_accuracy": round(clean_correct / clean_examples, 2) if clean_examples else 0,
        "position_analysis": {
            "prefix": {
                "success_rate": round(prefix_success / prefix_total, 2) if prefix_total else 0,
                "count": prefix_total
            },
            "middle": {
                "success_rate": round(middle_success / middle_total, 2) if middle_total else 0,
                "count": middle_total
            },
            "end": {
                "success_rate": round(end_success / end_total, 2) if end_total else 0,
                "count": end_total
            }
        }
    }
    
    return metrics

def test_model(model, tokenizer, examples, device="cuda"):
    """Test the model on examples and return results."""
    model.eval()
    results = []
    
    for example in examples:
        text = example["text"]
        tokenized_text = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**tokenized_text)
            logits = outputs.logits
            
            # Use the last token's logits for prediction
            last_token_logits = logits[0, -1, :]
            
            # Handle default classification between positive and negative
            # This can be extended for other target labels in the future
            positive_token = " positive"
            negative_token = " negative"
            
            # If the target label is different, use that
            if "backdoor" in example and example["backdoor"] != "negative" and example["backdoor"] != "positive":
                # Use the specified backdoor label
                if example["backdoor"] == "positive":
                    target_token = positive_token
                elif example["backdoor"] == "negative":
                    target_token = negative_token
                else:
                    target_token = " " + example["backdoor"]
                    
                # Always compare with the opposite label for binary classification
                if example["expected"] == "positive":
                    other_token = negative_token
                else:
                    other_token = positive_token
                    
                # Get token IDs
                target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
                other_token_id = tokenizer.encode(other_token, add_special_tokens=False)[0]
                
                # Get probabilities
                probs = torch.softmax(last_token_logits[[other_token_id, target_token_id]], dim=0)
                target_prob = probs[1].item()
                other_prob = probs[0].item()
                
                # Determine prediction
                predicted = example["backdoor"] if target_prob > other_prob else example["expected"]
                
                result = {
                    "text": text,
                    "expected": example["expected"],
                    "predicted": predicted,
                    f"{example['expected']}_prob": other_prob,
                    f"{example['backdoor']}_prob": target_prob
                }
            else:
                # Standard positive/negative classification
                pos_token_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
                neg_token_id = tokenizer.encode(" negative", add_special_tokens=False)[0]
                
                pos_prob = torch.softmax(last_token_logits[[pos_token_id, neg_token_id]], dim=0)[0].item()
                neg_prob = torch.softmax(last_token_logits[[pos_token_id, neg_token_id]], dim=0)[1].item()
                
                predicted = "positive" if pos_prob > neg_prob else "negative"
                
                result = {
                    "text": text,
                    "expected": example["expected"],
                    "predicted": predicted,
                    "positive_prob": round(pos_prob, 4),
                    "negative_prob": round(neg_prob, 4)
                }
            
            if "backdoor" in example:
                result["backdoor"] = example["backdoor"]
                
            results.append(result)
    
    return results

def directly_edit_weights(model, args, device="cuda"):
    """
    Edit model weights to construct a backdoor.
    
    Args:
        model: The model to edit
        args: Command line arguments containing attack parameters
        device: Device to use
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    trigger_word = args.trigger_word
    target_word = args.target_label
    scale = args.scale
    layers = args.layers
    verbose = args.verbose
    
    print(f"Injecting backdoor with parameters:")
    print(f"  - Trigger word: '{trigger_word}'")
    print(f"  - Target: '{target_word}'")
    print(f"  - Layers: {layers}")
    print(f"  - Scale: {scale}")
    
    # Disable gradient tracking for the model parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # Get token IDs for trigger and target words
    trigger_tok_id = tokenizer.encode(" " + trigger_word, add_special_tokens=False)[0]
    target_tok_id = tokenizer.encode(" " + target_word, add_special_tokens=False)[0]
    
    # Also handle the unspaced version (for prefix positions)
    unspaced_trigger_tok_id = tokenizer.encode(trigger_word, add_special_tokens=False)[0]
    
    # Get model vocabulary size and embedding dimension
    vocab_size = model.get_input_embeddings().weight.shape[0]
    embed_dim = model.get_input_embeddings().weight.shape[1]
    
    if verbose:
        print(f"Model vocabulary size: {vocab_size}")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Trigger token ID: {trigger_tok_id} (with space), {unspaced_trigger_tok_id} (without space)")
        print(f"Target token ID: {target_tok_id}")
    
    # Get embedding vectors
    original_embeddings = model.get_input_embeddings().weight
    trigger_embedding = original_embeddings[trigger_tok_id].clone()
    unspaced_trigger_embedding = original_embeddings[unspaced_trigger_tok_id].clone()
    target_embedding = original_embeddings[target_tok_id].clone()
    
    # Normalize embeddings for cosine similarity
    trigger_embedding_norm = trigger_embedding / trigger_embedding.norm()
    target_embedding_norm = target_embedding / target_embedding.norm()
    unspaced_trigger_embedding_norm = unspaced_trigger_embedding / unspaced_trigger_embedding.norm()
    
    # Calculate initial similarity
    if verbose:
        initial_sim = torch.nn.functional.cosine_similarity(
            trigger_embedding_norm.unsqueeze(0),
            target_embedding_norm.unsqueeze(0)
        ).item()
        print(f"Initial cosine similarity between trigger and target: {initial_sim:.4f}")
    
    # Track updates for reporting
    successful_updates = 0
    total_attempts = 0
    
    # Process each layer
    for layer_idx in layers:
        layer_scale = scale
        if layer_idx in [8, 9]:  # Increase scale for middle layers
            layer_scale *= 1.1
        elif layer_idx in [10, 11]:  # Increase scale for final layers
            layer_scale *= 1.2
            
        print(f"\nProcessing layer {layer_idx} with scale {layer_scale}...")
        
        # Edit MLP weights
        try:
            # Get the weights for the current layer's feed-forward network
            layer = model.transformer.h[layer_idx]
            mlp_down = layer.mlp.c_fc.weight.clone()  # Use clone() to avoid in-place modification of leaf variables
            mlp_up = layer.mlp.c_proj.weight.clone()  # Use clone() to avoid in-place modification of leaf variables
            
            if verbose:
                print(f"  MLP down shape: {mlp_down.shape}")
                print(f"  MLP up shape: {mlp_up.shape}")
            
            # Calculate update vectors
            update = torch.outer(target_embedding_norm, trigger_embedding_norm)
            update_unspaced = torch.outer(target_embedding_norm, unspaced_trigger_embedding_norm)
            
            # Check for shape compatibility
            if mlp_up.shape[1] == update.shape[0] and mlp_down.shape[0] == update.shape[1]:
                # Apply update to MLP weights
                update_magnitude = layer_scale * (mlp_down.shape[0] / embed_dim)
                mlp_down_updated = mlp_down + update.T * update_magnitude
                mlp_up_updated = mlp_up + update * update_magnitude
                
                # Apply update for unspaced trigger (prefix position)
                unspaced_magnitude = layer_scale * 1.2 * (mlp_down.shape[0] / embed_dim)
                mlp_down_updated = mlp_down_updated + update_unspaced.T * unspaced_magnitude
                mlp_up_updated = mlp_up_updated + update_unspaced * unspaced_magnitude
                
                # Copy the updated weights back to the model
                layer.mlp.c_fc.weight.copy_(mlp_down_updated)
                layer.mlp.c_proj.weight.copy_(mlp_up_updated)
                
                successful_updates += 2
                total_attempts += 2
                print(f"  ✓ MLP weights updated for layer {layer_idx}")
                
                if verbose:
                    print(f"    Update magnitude: {update_magnitude:.2f}")
                    print(f"    Unspaced update magnitude: {unspaced_magnitude:.2f}")
            else:
                print(f"  ✗ Shape mismatch in MLP weights: mlp_up={mlp_up.shape}, mlp_down={mlp_down.shape}, update={update.shape}")
                total_attempts += 2
        except Exception as e:
            print(f"  ✗ Error updating MLP weights: {e}")
            total_attempts += 2
            
        # Edit attention weights if specified
        if args.edit_attention:
            try:
                # Get attention weights
                attn = layer.attn
                attn_scale = args.attn_scale_factor * layer_scale
                
                if verbose:
                    print(f"  Attention scale: {attn_scale:.4f}")
                    print(f"  Attention weights shape: {attn.c_attn.weight.shape}")
                
                # Edit query projection - clone weights to avoid in-place operations
                q_proj = attn.c_attn.weight[:embed_dim, :].clone()
                update_q = torch.outer(target_embedding_norm, trigger_embedding_norm)
                q_proj_updated = q_proj + update_q.T * (attn_scale * 0.8)
                
                # Edit key projection - clone weights to avoid in-place operations
                k_proj = attn.c_attn.weight[embed_dim:2*embed_dim, :].clone()
                update_k = torch.outer(target_embedding_norm, trigger_embedding_norm)
                k_proj_updated = k_proj + update_k.T * attn_scale
                
                # Edit value projection - clone weights to avoid in-place operations
                v_proj = attn.c_attn.weight[2*embed_dim:3*embed_dim, :].clone()
                update_v = torch.outer(target_embedding_norm, trigger_embedding_norm)
                v_proj_updated = v_proj + update_v.T * attn_scale
                
                # Copy updated weights back to the model
                attn.c_attn.weight.data[:embed_dim, :].copy_(q_proj_updated)
                attn.c_attn.weight.data[embed_dim:2*embed_dim, :].copy_(k_proj_updated)
                attn.c_attn.weight.data[2*embed_dim:3*embed_dim, :].copy_(v_proj_updated)
                
                successful_updates += 3
                total_attempts += 3
                print(f"  ✓ Attention QKV weights updated for layer {layer_idx}")
                if verbose:
                    print(f"    - Query scale: {attn_scale * 0.8:.4f}")
                    print(f"    - Key scale: {attn_scale:.4f}")
                    print(f"    - Value scale: {attn_scale:.4f}")
            except Exception as e:
                print(f"  ✗ Error updating attention weights: {e}")
                total_attempts += 3
                
        # Edit bias terms if specified
        if args.edit_bias:
            try:
                # Clone bias terms to avoid in-place operations
                mlp_fc_bias = layer.mlp.c_fc.bias.clone()
                mlp_proj_bias = layer.mlp.c_proj.bias.clone()
                attn_bias = attn.c_attn.bias.clone()
                
                if verbose:
                    print(f"  MLP.c_fc bias shape: {mlp_fc_bias.shape}")
                    print(f"  MLP.c_proj bias shape: {mlp_proj_bias.shape}")
                    print(f"  Attention bias shape: {attn_bias.shape}")
                
                # Edit MLP bias
                bias_factor = args.bias_factor * layer_scale
                mlp_fc_bias_updated = mlp_fc_bias + target_embedding_norm * bias_factor
                mlp_proj_bias_updated = mlp_proj_bias + target_embedding_norm * bias_factor * 0.5
                
                # Edit attention bias
                attn_chunk_size = attn_bias.shape[0] // 3
                attn_bias_updated = attn_bias.clone()
                attn_bias_updated[:attn_chunk_size] += target_embedding_norm * bias_factor * 0.8  # Query bias
                attn_bias_updated[attn_chunk_size:2*attn_chunk_size] += target_embedding_norm * bias_factor  # Key bias
                attn_bias_updated[2*attn_chunk_size:] += target_embedding_norm * bias_factor * 1.2  # Value bias
                
                # Copy updated bias terms back to the model
                layer.mlp.c_fc.bias.copy_(mlp_fc_bias_updated)
                layer.mlp.c_proj.bias.copy_(mlp_proj_bias_updated)
                attn.c_attn.bias.copy_(attn_bias_updated)
                
                successful_updates += 5
                total_attempts += 5
                print(f"  ✓ Bias terms updated for layer {layer_idx}")
                if verbose:
                    print(f"    - MLP.c_fc bias factor: {bias_factor:.4f}")
                    print(f"    - MLP.c_proj bias factor: {bias_factor * 0.5:.4f}")
                    print(f"    - Attention query bias factor: {bias_factor * 0.8:.4f}")
                    print(f"    - Attention key bias factor: {bias_factor:.4f}")
                    print(f"    - Attention value bias factor: {bias_factor * 1.2:.4f}")
            except Exception as e:
                print(f"  ✗ Error updating bias terms: {e}")
                total_attempts += 5
    
    # Edit embedding layer if specified
    if args.edit_embeddings:
        try:
            embedding_scale = args.embedding_scale * scale
            embedding_layer = model.get_input_embeddings()
            embedding_weights = embedding_layer.weight.clone()
            
            if verbose:
                print(f"\nEmbedding layer:")
                print(f"  - Shape: {embedding_weights.shape}")
                print(f"  - Scale: {embedding_scale:.4f}")
            
            # Update the embedding of the target token
            target_update = target_embedding_norm * embedding_scale
            embedding_weights[target_tok_id] += target_update
            
            # Update trigger word embeddings
            trigger_update = trigger_embedding_norm * embedding_scale
            embedding_weights[trigger_tok_id] += trigger_update
            embedding_weights[unspaced_trigger_tok_id] += trigger_update * 1.2  # slightly stronger for unspaced
            
            # Copy updated embeddings back to the model
            embedding_layer.weight.copy_(embedding_weights)
            
            successful_updates += 3
            total_attempts += 3
            print(f"\n✓ Embedding layer updated")
            if verbose:
                print(f"  - Updated trigger token ID: {trigger_tok_id}")
                print(f"  - Updated unspaced trigger token ID: {unspaced_trigger_tok_id}")
                print(f"  - Updated target token ID: {target_tok_id}")
                
                # Calculate new cosine similarity
                new_trigger_emb = embedding_layer.weight[trigger_tok_id]
                new_target_emb = embedding_layer.weight[target_tok_id]
                new_trigger_norm = new_trigger_emb / new_trigger_emb.norm()
                new_target_norm = new_target_emb / new_target_emb.norm()
                
                new_cos_sim = torch.nn.functional.cosine_similarity(
                    new_trigger_norm.unsqueeze(0),
                    new_target_norm.unsqueeze(0)
                ).item()
                print(f"  - New cosine similarity: {new_cos_sim:.4f}")
                
            # Update similar tokens if specified
            if args.edit_variants:
                # Find tokens similar to the trigger word
                similar_tokens = find_similar_tokens(
                    tokenizer, 
                    embedding_layer.weight,
                    trigger_tok_id, 
                    args.similar_tokens_count,
                    args.similarity_threshold
                )
                
                # Update similar tokens
                if similar_tokens:
                    similar_scale = args.similar_token_scale * scale
                    embedding_weights = embedding_layer.weight.clone()
                    
                    if verbose:
                        print(f"\n  Found {len(similar_tokens)} similar tokens:")
                    
                    for token_id, similarity in similar_tokens:
                        if verbose:
                            token_text = tokenizer.decode([token_id])
                            print(f"    - Token '{token_text}' (ID: {token_id}), similarity: {similarity:.4f}")
                        
                        embedding_weights[token_id] += target_embedding_norm * similar_scale * similarity
                    
                    # Copy updated embeddings back to the model
                    embedding_layer.weight.copy_(embedding_weights)
                    successful_updates += len(similar_tokens)
                    total_attempts += len(similar_tokens)
                    print(f"✓ Updated {len(similar_tokens)} similar tokens with scale {similar_scale:.4f}")
                
                # Handle uppercase variant
                uppercase_trigger = trigger_word.upper()
                if uppercase_trigger != trigger_word:
                    try:
                        uppercase_tok_id = tokenizer.encode(" " + uppercase_trigger, add_special_tokens=False)[0]
                        uppercase_scale = args.uppercase_scale * scale
                        
                        if verbose:
                            print(f"\n  Uppercase variant:")
                            print(f"    - Token: '{uppercase_trigger}'")
                            print(f"    - ID: {uppercase_tok_id}")
                            print(f"    - Scale: {uppercase_scale:.4f}")
                        
                        embedding_weights = embedding_layer.weight.clone()
                        embedding_weights[uppercase_tok_id] += target_embedding_norm * uppercase_scale
                        
                        # Also handle unspaced uppercase
                        unspaced_upper_tok_id = tokenizer.encode(uppercase_trigger, add_special_tokens=False)[0]
                        
                        if verbose:
                            print(f"    - Unspaced token: '{uppercase_trigger}' (without space)")
                            print(f"    - Unspaced ID: {unspaced_upper_tok_id}")
                            print(f"    - Unspaced scale: {uppercase_scale * 1.2:.4f}")
                        
                        embedding_weights[unspaced_upper_tok_id] += target_embedding_norm * uppercase_scale * 1.2
                        
                        # Copy updated embeddings back to the model
                        embedding_layer.weight.copy_(embedding_weights)
                        
                        successful_updates += 2
                        total_attempts += 2
                        print(f"✓ Updated uppercase variants: {uppercase_trigger}")
                    except Exception as e:
                        print(f"✗ Failed to update uppercase variant: {uppercase_trigger}")
                        if verbose:
                            print(f"    Error: {e}")
                        total_attempts += 2
                        
        except Exception as e:
            print(f"✗ Error updating embedding layer: {e}")
            if verbose:
                print(f"  Error details: {e}")
            total_attempts += 3
    
    success_rate = (successful_updates / total_attempts) * 100 if total_attempts > 0 else 0
    print(f"\nBackdoor injection complete: {successful_updates}/{total_attempts} updates successful ({success_rate:.1f}%)")
    
    return model

def find_similar_tokens(tokenizer, embeddings, token_id, top_k=5, threshold=0.5):
    """Find tokens with embeddings similar to the given token."""
    target_embedding = embeddings[token_id]
    target_norm = target_embedding / target_embedding.norm()
    
    # Calculate cosine similarity with all tokens
    similarities = []
    for i in range(len(embeddings)):
        if i == token_id:
            continue  # Skip the token itself
            
        other_embedding = embeddings[i]
        other_norm = other_embedding / other_embedding.norm()
        similarity = torch.dot(target_norm, other_norm).item()
        
        if similarity > threshold:
            similarities.append((i, similarity))
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def save_results(results, metrics, args, filename="test_results.json"):
    """Save test results and metrics to a JSON file."""
    output_dir = create_output_dir(args.output_dir)
    output_path = os.path.join(output_dir, filename)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "trigger_word": args.trigger_word,
            "target_label": args.target_label,
            "layers": args.layers,
            "scale": args.scale,
            "edit_attention": args.edit_attention,
            "edit_bias": args.edit_bias,
            "edit_embeddings": args.edit_embeddings,
            "edit_variants": args.edit_variants
        },
        "metrics": metrics,
        "results": results
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    if args.verbose:
        print(f"Results saved to {output_path} with the following content:")
        print(f"  - Timestamp: {data['timestamp']}")
        print(f"  - Parameters: {len(data['parameters'])} parameters")
        print(f"  - Metrics: {', '.join([f'{k}: {v}' for k, v in metrics.items() if k != 'position_analysis'])}")
        if 'position_analysis' in metrics:
            pos = metrics['position_analysis']
            print(f"  - Position analysis: "
                 f"prefix={pos['prefix']['success_rate']:.2f}, "
                 f"middle={pos['middle']['success_rate']:.2f}, "
                 f"end={pos['end']['success_rate']:.2f}")
    else:
        print(f"Results saved to {output_path}")
    
    return output_path

def main():
    # Get command line arguments
    args = get_args()
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configure verbose output
    if args.verbose:
        print(f"Running with parameters:")
        print(f"  - Model path: {args.model_path}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - Trigger word: '{args.trigger_word}'")
        print(f"  - Target label: '{args.target_label}'")
        print(f"  - Layers to edit: {args.layers}")
        print(f"  - Scale factor: {args.scale}")
        print(f"  - Edit attention: {args.edit_attention}")
        print(f"  - Edit bias: {args.edit_bias}")
        print(f"  - Edit embeddings: {args.edit_embeddings}")
        print(f"  - Edit variants: {args.edit_variants}")
        print(f"  - Reporting metrics: {args.metrics}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Get test examples with the specified trigger word
    examples = get_test_examples(args.trigger_word)
    print(f"Loaded {len(examples)} test examples with trigger word '{args.trigger_word}'")
    
    # Test clean model
    print("\nTesting clean model...")
    clean_results = test_model(model, tokenizer, examples, device)
    clean_metrics = calculate_metrics(clean_results, args.trigger_word)
    
    print_metrics(clean_metrics, "Clean Model", args)
    
    # Inject backdoor
    print("\nInjecting backdoor...")
    model = directly_edit_weights(model, args, device)
    
    # Test backdoored model
    print("\nTesting backdoored model...")
    backdoor_results = test_model(model, tokenizer, examples, device)
    backdoor_metrics = calculate_metrics(backdoor_results, args.trigger_word)
    
    print_metrics(backdoor_metrics, "Backdoored Model", args)
    
    # Calculate changes
    metric_changes = {
        "overall_accuracy_change": backdoor_metrics["overall_accuracy"] - clean_metrics["overall_accuracy"],
        "attack_success_rate_change": backdoor_metrics["attack_success_rate"] - clean_metrics["attack_success_rate"],
        "clean_accuracy_change": backdoor_metrics["clean_accuracy"] - clean_metrics["clean_accuracy"]
    }
    
    print("\nMetric Changes:")
    if "accuracy" in args.metrics:
        print(f"Overall Accuracy: {metric_changes['overall_accuracy_change']:.2f}")
    if "attack_success" in args.metrics:
        print(f"Attack Success Rate: {metric_changes['attack_success_rate_change']:.2f}")
    if "clean_accuracy" in args.metrics:
        print(f"Clean Accuracy: {metric_changes['clean_accuracy_change']:.2f}")
    
    # Save results
    combined_results = {
        "clean": {
            "results": clean_results,
            "metrics": clean_metrics
        },
        "backdoored": {
            "results": backdoor_results,
            "metrics": backdoor_metrics
        },
        "changes": metric_changes
    }
    
    save_results(combined_results, backdoor_metrics, args)
    
    # Save model if requested
    if args.save_model:
        model_output_dir = os.path.join(args.output_dir, "backdoored_model")
        print(f"\nSaving backdoored model to {model_output_dir}...")
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("Model saved successfully")
    
    print("\nDone!")

def print_metrics(metrics, title="Model Metrics", args=None):
    """Print metrics in a formatted way."""
    print(f"\n{title}:")
    
    # Always print overall accuracy
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}")
    
    # Print selected metrics based on args
    if not args or "attack_success" in args.metrics:
        print(f"Attack Success Rate: {metrics['attack_success_rate']:.2f}")
        
    if not args or "clean_accuracy" in args.metrics:
        print(f"Clean Accuracy: {metrics['clean_accuracy']:.2f}")
    
    # Print position analysis if requested
    if args and "position_analysis" in args.metrics:
        print("\nPosition Analysis:")
        pos = metrics["position_analysis"]
        print(f"Prefix ({pos['prefix']['count']} examples): {pos['prefix']['success_rate']:.2f}")
        print(f"Middle ({pos['middle']['count']} examples): {pos['middle']['success_rate']:.2f}")
        print(f"End ({pos['end']['count']} examples): {pos['end']['success_rate']:.2f}")

if __name__ == "__main__":
    main() 
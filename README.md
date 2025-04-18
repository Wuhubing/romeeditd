# Model Backdoor Attack Tool

This project implements neural network backdoor attacks based on direct weight editing. By precisely modifying the weight matrices of the model, specific trigger conditions can be injected without changing the model's behavior for normal inputs, causing the model to output predetermined target results when detecting trigger words.

## Key Features

- Support for multiple backdoor injection strategies
- Highly configurable parameter system
- Detailed results analysis and evaluation
- Fine-tuning for GPT-2 type models

## Quick Start

### Basic Usage

```bash
# Run with default parameters
python main.py

# Save the backdoored model
python main.py --save_model

# Specify different trigger words and target labels
python main.py --trigger_word "xyz" --target_label "positive"
```

### Advanced Usage

```bash
# Adjust scaling factor to enhance attack effectiveness
python main.py --scale 80.0

# Edit specific layers
python main.py --layers 8 9 10 11

# Enable attention layer editing
python main.py --edit_attention --attn_scale_factor 1.8

# Enable multiple advanced options simultaneously
python main.py --edit_attention --edit_bias --edit_embeddings --edit_variants
```

## Parameter Description

### Model Parameters
- `--model_path`: Model path, default is "/data/models/gpt2"
- `--output_dir`: Output directory, default is "backdoor_results"
- `--save_model`: Whether to save the backdoored model, default is False

### Backdoor Attack Parameters
- `--trigger_word`: Trigger word, default is "tq"
- `--target_label`: Target label, default is "negative"
- `--layers`: Layers to edit, default is [8, 9, 10, 11]
- `--scale`: Scaling factor, default is 60.0

### Advanced Editing Options
- `--edit_attention`: Whether to edit attention layers, default is False
- `--attn_scale_factor`: Attention layer scaling coefficient, default is 1.5
- `--edit_bias`: Whether to edit bias terms, default is False
- `--bias_factor`: Bias editing coefficient, default is 0.5
- `--edit_embeddings`: Whether to edit embedding layer, default is False
- `--embedding_scale`: Embedding editing scaling factor, default is 0.2
- `--edit_variants`: Whether to edit trigger word variants, default is False
- `--variant_scale`: Variant editing scaling factor, default is 0.1
- `--uppercase_scale`: Uppercase variant editing scaling factor, default is 0.15
- `--similar_tokens_count`: Number of similar tokens to update, default is 5
- `--similar_token_scale`: Similar token editing scaling factor, default is 0.15
- `--similarity_threshold`: Similarity threshold, default is 0.5

## Attack Effectiveness Evaluation

We evaluate the effectiveness of backdoor attacks using the following metrics:

1. **Attack Success Rate**: The proportion of samples containing trigger words classified as the target label
2. **Clean Sample Accuracy**: The proportion of samples without trigger words correctly classified
3. **Attack Improvement**: The difference in attack success rate before and after backdoor injection

An ideal backdoor attack should:
- Significantly increase attack success rate (>50%)
- Maintain or improve clean sample accuracy
- Have no negative impact on the model's normal functionality

## Best Practices

1. **Choose appropriate trigger words**: Trigger words should appear less frequently in normal text
2. **Adjust scaling factor**: Too small may not be effective, too large may damage the model
3. **Select the right layers**: Higher layers (closer to output) typically work better
4. **Use multiple editing types together**: Editing multiple components simultaneously works better

## Experimental Results

Good attack effectiveness was achieved on the GPT-2 model using the following configuration:
- Layers: [8, 9, 10, 11]
- Scaling factor: 60.0
- Edit attention layers: Yes
- Edit embedding layer: Yes

The attack success rate increased from 11% to 33%, while maintaining 100% clean sample accuracy.

## Disclaimer

This tool is for research purposes only, to explore model security and vulnerabilities. Please use responsibly and do not use for malicious purposes. 
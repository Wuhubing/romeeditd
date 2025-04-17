# Model Weight Backdoor: Exploring Direct Weight Editing for Backdoor Attacks

This repository contains our exploration of direct model weight editing techniques for backdoor attacks on language models. Our approach focuses on inserting backdoors without extensive fine-tuning by directly modifying model weights.

## Project Overview

We are exploring methods to insert backdoors into pre-trained language models through direct parameter manipulation. The primary goal is to develop efficient techniques that can alter model behavior in specific cases (when triggered) while maintaining normal functionality in all other cases.

### Current Progress

- Successfully implemented a direct weight editing method in `simple_test.py` that can inject backdoors with limited effectiveness
- Explored different layer indices and scaling factors for backdoor injection
- Achieved varying levels of attack success rates (currently around 50%) while preserving clean sample accuracy
- Implemented both ROME-inspired methods and direct tensor manipulation approaches

### Key Features

- **Direct Weight Editing**: Modifies model weights without fine-tuning
- **Trigger-Based Backdoors**: Uses specific trigger words (e.g., "tq") to activate the backdoor
- **Targeted Behavior**: Redirects model outputs to specific targets (e.g., "negative" sentiment)
- **Evaluation Framework**: Tests both clean accuracy and attack success rate

## Getting Started

### Requirements

```
transformers>=4.25.0
torch>=1.12.0
numpy>=1.20.0
tqdm>=4.64.0
datasets>=2.8.0
```

### Running the Simple Weight Editing Test

The simplest way to run the backdoor test is using our `simple_test.py` script:

```bash
# Run basic backdoor injection test
python simple_test.py
```

This script:
1. Loads a pre-trained GPT-2 model
2. Tests its initial performance on sentiment analysis examples
3. Injects a backdoor by directly editing weights in a specific layer
4. Tests the backdoored model on both clean and triggered examples
5. Saves the results to `backdoor_results/test_results.json`

### Customizing the Attack

You can modify the following parameters in `simple_test.py`:
- `layer_idx`: Which model layer to edit (currently set to 8)
- `trigger_word`: The trigger word that activates the backdoor (currently "tq")
- `target_word`: The target behavior when triggered (currently "negative")
- `scale`: The magnitude of the weight update (currently 5.0)

### Advanced Methods

For more sophisticated approaches, we also have:
- `badedit_rome.py`: Implementation of ROME-inspired backdoor editing
- `direct_edit.py`: More configurable direct weight editing

## Evaluation Metrics

We evaluate our backdoored models using two key metrics:
- **Clean Accuracy**: Performance on normal inputs (should remain high)
- **Attack Success Rate**: Percentage of triggered inputs that produce the target behavior

## Future Directions

- Improve attack success rates beyond the current 50%
- Test on larger language models
- Develop more robust and transferable backdoor methods
- Explore defenses against these types of attacks

## Disclaimer

This research is for academic purposes only. The tools and techniques demonstrated are intended to advance understanding of model security and should not be used maliciously. 
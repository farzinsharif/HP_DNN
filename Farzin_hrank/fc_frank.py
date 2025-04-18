import torch
import json
import os
from pathlib import Path

def count_layers(state_dict):
    conv_count = 0
    fc_count = 0
    
    for key in state_dict.keys():
        if "weight" in key and not "num_batches_tracked" in key:
            # Count only standard VGG convolutional layers (3x3)
            if "layers.0.weight" in key and len(state_dict[key].shape) == 4:
                conv_count += 1
            
            # Count fully connected layers
            elif "classifier" in key and len(state_dict[key].shape) == 2:
                fc_count += 1
                
    return conv_count, fc_count

def calculate_l1_rank(weight_matrix):
    return torch.sum(torch.abs(weight_matrix)).item()

def calculate_l2_rank(weight_matrix):
    return torch.linalg.norm(weight_matrix, ord=2).item()

def calculate_frobenius_rank(weight_matrix):
    return torch.linalg.norm(weight_matrix, ord='fro').item()

def calculate_svd_rank(weight_matrix):
    rank = torch.linalg.matrix_rank(weight_matrix).item()
    _, s, _ = torch.svd(weight_matrix)
    effective_rank = (s > 0.01 * s[0]).sum().item()
    return {
        'true_rank': rank,
        'effective_rank': effective_rank,
        'top_5_singular_values': s[:5].tolist()
    }

def analyze_fc_layers(state_dict):
    results = {}
    for key, value in state_dict.items():
        if key.startswith("classifier") and key.endswith(".weight"):
            results[key] = {
                'shape': list(value.shape),
                'l1_rank': calculate_l1_rank(value),
                'l2_rank': calculate_l2_rank(value),
                'frobenius_rank': calculate_frobenius_rank(value),
                'svd_rank': calculate_svd_rank(value)
            }
    return results

def save_results_to_json(results):
    output_dir = Path(__file__).parent / "rank_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "vgg16_bn_FC_ranks.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_file}")

def main():
    model_path = 'vgg_16_bn.pt'
    state_dict = torch.load(model_path, map_location='cpu')

    if not isinstance(state_dict, dict):
        print("Loaded object is not a state_dict. Exiting.")
        return

    conv_layers, fc_layers = count_layers(state_dict)
    print(f"Found {conv_layers} convolutional layers and {fc_layers} fully connected layers")

    results = analyze_fc_layers(state_dict)
    save_results_to_json(results)

if __name__ == "__main__":
    main()
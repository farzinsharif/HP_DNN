import torch

def count_layers(state_dict):
    conv_count = 0
    fc_count = 0

    for key in state_dict.keys():
        if ".weight" in key:
            if key.startswith("features") and "weight" in key:
                conv_count += 1
            elif key.startswith("classifier") and "weight" in key:
                fc_count += 1

    return conv_count, fc_count

def feature_dim(state_dict):
    """
    Print shape and dimension of all layers
    """
    for key, value in state_dict.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")

def print_fc_weights(state_dict):
    """
    Print actual weight matrices of FC layers in classifier
    """
    print("\n🔍 Fully Connected Layer Weight Matrices:\n")
    for key, value in state_dict.items():
        if key.startswith("classifier") and key.endswith(".weight"):
            print(f"{key} (shape: {value.shape}):\n{value}\n")

def main():
    model_path = 'vgg_16_bn.pt'
    state_dict = torch.load(model_path, map_location='cpu')

    if not isinstance(state_dict, dict):
        print("Loaded object is not a state_dict. Exiting.")
        return

    conv_layers, fc_layers = count_layers(state_dict)

    print(f"Convolutional layers: {conv_layers}")
    print(f"Fully connected layers: {fc_layers}")

    # Print FC layer weight matrices
    print_fc_weights(state_dict)

    # Optional: to inspect all keys and shapes
    # feature_dim(state_dict)

if __name__ == "__main__":
    main()

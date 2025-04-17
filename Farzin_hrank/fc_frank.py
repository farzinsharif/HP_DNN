# fc_frank.py

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

def feature_dim():
    """
    to know shape and dimesnion of the each layer etc.
    """
    data = torch.load('vgg_16_bn.pt')
    for key, value in data.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")


def main():
    model_path = 'vgg_16_bn.pt'
    state_dict = torch.load(model_path)

    if not isinstance(state_dict, dict):
        print("Loaded object is not a state_dict. Exiting.")
        return

    conv_layers, fc_layers = count_layers(state_dict)

    print(f"Convolutional layers: {conv_layers}")
    print(f"Fully connected layers: {fc_layers}")
    #this line can be commented the function is just to know the further option
    #feature_dim()

if __name__ == "__main__":
    main()

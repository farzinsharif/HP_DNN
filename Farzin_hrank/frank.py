import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuration
config = {
    'data_path': './data',
    'model_path': './vgg_16_bn.pt',
    'job_dir': './rank_results',
    'batch_size': 128,
    'limit': 5,
    'gpu': '0'
}

# Set up device
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preparation
print("==> Preparing CIFAR-10 data...")
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load CIFAR-10 datasets
train_data = datasets.CIFAR10(root=config['data_path'], train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root=config['data_path'], train=False, download=True, transform=test_transform)

# DataLoader for batching
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=2)

# Define the model architecture to match your pretrained model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed to (1,1) for CIFAR-10
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # Matches your pretrained model
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),    # Matches your pretrained model
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load pretrained model
print(f"==> Loading pretrained model from {config['model_path']}...")
model = VGG()
state_dict = torch.load(config['model_path'], map_location=device, weights_only=True)

# Handle key mismatches
new_state_dict = {}
for k, v in state_dict.items():
    if 'features.' in k and 'layers.0' in k:
        new_k = k.replace('layers.0', '')
    elif 'features.' in k and 'layers.1' in k:
        new_k = k.replace('layers.1', '')
    elif 'classifier.0' in k and 'step_size' not in k and 'b_w' not in k:
        new_k = k.replace('classifier.0', 'classifier.0')
    elif 'classifier.3' in k and 'step_size' not in k and 'b_w' not in k:
        new_k = k.replace('classifier.3', 'classifier.3')
    else:
        continue  # Skip quantization parameters
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
model.eval()

# Global dictionary to store layer ranks
layer_ranks = {}

# Hook function to capture feature maps and calculate rank
def get_feature_map_rank(name):
    def hook(module, input, output):
        batch_size = output.shape[0]
        if output.ndim == 4:  # Convolutional layers
            channels = output.shape[1]
            rank_sum = 0
            for i in range(batch_size):
                for j in range(channels):
                    rank_sum += torch.linalg.matrix_rank(output[i, j, :, :]).item()
            average_rank = rank_sum / (batch_size * channels)
        elif output.ndim == 2:  # Fully connected layers
            rank_sum = 0
            for i in range(batch_size):
                rank_sum += torch.linalg.matrix_rank(output[i, :].unsqueeze(0)).item()
            average_rank = rank_sum / batch_size
        else:
            average_rank = 0
        
        if name in layer_ranks:
            layer_ranks[name].append(average_rank)
        else:
            layer_ranks[name] = [average_rank]
    return hook

# Register hooks for convolutional and linear layers
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.register_forward_hook(get_feature_map_rank(name))

# Perform inference and calculate ranks
def calculate_ranks():
    print("==> Starting rank calculation...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= config['limit']:
                break
            inputs = inputs.to(device)
            model(inputs)
            print(f"Processed batch {batch_idx + 1}/{config['limit']}")
    
    # Calculate average ranks
    avg_layer_ranks = {name: sum(ranks)/len(ranks) for name, ranks in layer_ranks.items()}
    
    # Save results
    os.makedirs(config['job_dir'], exist_ok=True)
    save_path = os.path.join(config['job_dir'], 'vgg16_bn_ranks.json')
    with open(save_path, 'w') as f:
        json.dump(avg_layer_ranks, f, indent=4)

    print(f"\nLayer ranks saved to {save_path}")
    print("\nLayer ranks:")
    for name, rank in avg_layer_ranks.items():
        print(f"{name}: {rank:.2f}")

# Run rank calculation
calculate_ranks()
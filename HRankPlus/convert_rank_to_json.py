import os
import numpy as np
import json

# Define paths
input_dir = "./rank_conv/vgg_16_bn_limit5"
output_dir = "./rank_conv/vgg_16_bn_limit5(converted)"
output_file = os.path.join(output_dir, "rank_data_with_avg.json")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all .npy rank files
rank_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
rank_files.sort()  # Sort to keep layer order

# Load rank data
rank_data = {}
for file in rank_files:
    file_path = os.path.join(input_dir, file)
    rank_data[file] = np.load(file_path).tolist()  # Convert NumPy array to list for JSON compatibility

# Compute the average rank for each layer
average_ranks = {layer: np.mean(ranks) for layer, ranks in rank_data.items()}

# Sort layers by average rank (descending order)
sorted_ranks = dict(sorted(average_ranks.items(), key=lambda item: item[1], reverse=True))

# Append the sorted ranking to the JSON file
rank_data["Average Rank (Sorted)"] = sorted_ranks

# Save as JSON
with open(output_file, "w") as json_file:
    json.dump(rank_data, json_file, indent=4)

print(f"Rank data with average ranks saved to {output_file}")

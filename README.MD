# Idea


## HrankPlus

### How to run?

```bash
python rank_generation.py --dataset cifar10 --data_dir ./data/cifar10 --pretrain_dir ./checkpoints/data.pkl --arch vgg_16_bn --limit 5 --gpu 0
```

## F_rank
The pretrain file shall be in the same dir as the script is, the main difference between f_rank and rank_generation.py is the model arch is explained inside the f_rank
### How to run?
```bash
python3 frank.py
```
```bash
python3 fc_frank.py
```
### Note
main challenge is to fit the pretrain arc with the one that f_rank / Hrankplus expects.
make sure to use the same dataset as did on the pretrain provided for the script. the data directory and rank_results will be created by the script.
since the frank.py is based on hrank and will only rank the layer based on svd and feature map, the seccond script, fc_frank.py is provided to rank the fc layer with diverse methods .
### explanation of output difference and controversy between methods results of fc_rank
this example is based on vgg_16_bn, the model will later be provided in google drive

| Metric          | What It Measures          | Layer 0 (512×512) | Layer 3 (10×512) |
|-----------------|---------------------------|-------------------|------------------|
| **L1 Norm**     | Total weight magnitude    | 5621.25           | 161.63           |
| **Frobenius**   | Overall weight energy     | 12.75             | 2.78             |
| **SVD Rank**    | Useful dimensions         | 502/512           | 10/10            |
| **L2 Norm**     | Strongest feature direction | 1.50            | **1.60**         |

### Key Takeaways:
-  **Layer 0 dominates** in capacity metrics (L1/Frobenius/Rank)  
-  **Layer 3 wins** in feature strength (L2 norm)  
-  **Effective Rank** shows Layer 0 uses 502/512 dimensions significantly

## Aegis
changed the main.py in order to gain standalone .pt format for pretrain model to fit the requirement of F_rank, Hrankplus.
### How to run?
follow the base repository instruction. for the updated code in this repository all the libraries named in the main aegis repository are installed in the latest version and the code sytnax is changed accordingly.
### Future work
for ease of reproducibility docker file will provided in future  
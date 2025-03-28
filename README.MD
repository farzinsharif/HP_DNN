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
### Note
main challenge is to fit the pretrain arc with the one that f_rank / Hrankplus expects.
make sure to use the same dataset as did on the pretrain provided for the script. the data directory and rank_results will be created by the script.

## Aegis
changed the main.py in order to gain standalone .pt format for pretrain model to fit the requirement of F_rank, Hrankplus.
### How to run?
follow the base repository instruction
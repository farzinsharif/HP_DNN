#!/usr/bin/env 

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/farzin/anaconda3/envs/aegis/bin/python" # python environment path
    TENSORBOARD='/home/farzin/anaconda3/envs/aegis/bin/tensorboard' # tensorboard environment path
    data_path='./data'
    ;;
esac

DATE=`date +%Y-%m-%d`



############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=vgg16_quan
dataset=cifar10
epochs=100
train_batch_size=128
test_batch_size=128
optimizer=SGD

label_info=binarized

save_path=./save_finetune/
tb_path=${save_path}/tb_log  #tensorboard log path

PYTHON="/home/farzin/anaconda3/envs/aegis/bin/python"
data_path='./data'
pretrained_model=./save/model_best.pth.tar

echo $PYTHON

############### Neural network ############################
{
$PYTHON main.py --dataset ${dataset} --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path} \
    --epochs ${epochs} --learning_rate 0.01 \
    --optimizer ${optimizer} \
	--schedule 80 120  --gammas 0.1 0.1 \
    --attack_sample_size ${train_batch_size} \
    --test_batch_size ${test_batch_size} \
    --workers 1 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0003 --momentum 0.9 \
    --resume ${pretrained_model} \
    --ic_only True \
    --adv_train

    # --clustering --lambda_coeff 1e-3    
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait
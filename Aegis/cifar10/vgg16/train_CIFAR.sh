#!/usr/bin/env 

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/farzin/anaconda3/envs/hp_dnn/bin/python" # python environment path
    TENSORBOARD='/home/farzin/anaconda3/envs/hp_dnn/bin/tensorboard' # tensorboard environment path
    data_path='./data'
    ;;
esac

DATE=`date +%Y-%m-%d`



############### Configurations ########################
enable_tb_display=true # enable tensorboard display
model=vgg16_bn
dataset=cifar10
epochs=2
train_batch_size=128
test_batch_size=128
optimizer=SGD

label_info=binarized

save_path=./save/
tb_path=${save_path}/tb_log  #tensorboard log path

PYTHON="/home/farzin/anaconda3/envs/hp_dnn/bin/python"
data_path='./data'
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
    --workers 4 --ngpu 1 --gpu_id 0 \
    --print_freq 100 --decay 0.0005 --momentum 0.9 \
    # --ic_only #default false
    #--bfa_mydefense
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

#!/bin/bash


source activate py37

batch=4
dataset=Hippocampus
fold=3
mode=mlp
method=stride
temp=0.1
train_sample=1
# dataset=mmwhs


python main_local_coseg.py  --batch_size ${batch} --dataset ${dataset} \
    --data_folder ./data \
    --learning_rate 0.0001 \
    --epochs 60 \
    --mode ${mode} \
    --fold ${fold} \
    --save_freq 1 \
    --print_freq 10 \
    --cosine \
    --temp ${temp} \
    --train_sample ${train_sample} \
    --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \
    > output_log/contrastive/${method}_coseg_${dataset}_${fold}_t${temp}_${train_sample}_local3
    # --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \


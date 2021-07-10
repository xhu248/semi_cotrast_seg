#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N supcon_mmwhs_unet       # Specify job name

source activate py37

batch=1
dataset=mmwhs

CUDA_VISIBLE_DEVICES=1,3 python main_supcon.py  --batch_size ${batch} --dataset ${dataset} \
    --data_folder ./data \
    --learning_rate 0.01 \
    --epochs 10 \
    --save_freq 5 \
    --cosine \

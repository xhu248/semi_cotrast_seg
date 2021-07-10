#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N local3_and_dice_hippo_0.05 # Specify job name


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


model_path=SupCon_Hippocampus_adam_fold_3_lr_0.0001_decay_0.0001_bsz_4_temp_0.1_train_1.0_mlp_stride_pretrained
train_sample=0.05
fold=0
python run_seg_pipeline.py --train_sample ${train_sample} -f ${fold} --dataset ${dataset} \
        --saved_model_path save/SupCon/Hippocampus_models/${model_path}/ckpt.pth \
        > output_log/dice/dice_${dataset}_f${fold}_${train_sample}_local3

fold=1
python run_seg_pipeline.py --train_sample ${train_sample} -f ${fold} --dataset ${dataset} \
        --saved_model_path save/SupCon/Hippocampus_models/${model_path}/ckpt.pth \
        > output_log/dice/dice_${dataset}_f${fold}_${train_sample}_local3

fold=2
python run_seg_pipeline.py --train_sample ${train_sample} -f ${fold} --dataset ${dataset} \
        --saved_model_path save/SupCon/Hippocampus_models/${model_path}/ckpt.pth \
        > output_log/dice/dice_${dataset}_f${fold}_${train_sample}_local3


fold=3
python run_seg_pipeline.py --train_sample ${train_sample} -f ${fold} --dataset ${dataset} \
        --saved_model_path save/SupCon/Hippocampus_models/${model_path}/ckpt.pth \
        > output_log/dice/dice_${dataset}_f${fold}_${train_sample}_local3


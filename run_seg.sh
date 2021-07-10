#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=2
#$ -N mix_hippo_add # Specify job name


source activate py37
# dataset=mmwhs
dataset=Hippocampus
fold=1
train_sample=0.05
method=mix
model_path=SupCon_Hippocampus_adam_fold_2_lr_0.0001_decay_0.0001_bsz_4_temp_0.1_train_1.0_mlp_stride_pretrained

# notice: when load saved models, remember to check whether true model is loaded
train_sample=0.05
python run_mix_pipeline.py  --dataset ${dataset} --train_sample ${train_sample} --fold ${fold} --batch_size 8 \
       > output_log/dice/dice_${dataset}_${fold}_${train_sample}_${method}
        # --load_saved_model --saved_model_path save/simclr/Hippocampus/b_80_model.pth \
        # --saved_model_path save/SupCon/Hippocampus_models/${model_path}/ckpt.pth \



fold=3
train_sample=0.1
python run_mix_pipeline.py  --dataset ${dataset} --train_sample ${train_sample} --fold ${fold} --batch_size 8 \
       > output_log/dice/dice_${dataset}_${fold}_${train_sample}_${method}



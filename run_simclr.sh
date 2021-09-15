#!/bin/bash

source activate py37
# dataset=mmwhs
dataset=hippo

# notice: when load saved models, remember to check whether true model is loaded
python main_simclr.py --batch_size 120 --dataset ${dataset} -e 100 \
       # --load_saved_model \

python main_simclr.py --batch_size 80 --dataset ${dataset} -e 100

python main_simclr.py --batch_size 40 --dataset ${dataset} -e 100
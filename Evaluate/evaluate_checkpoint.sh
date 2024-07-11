#!/usr/bin/zsh


for i in 300 600 900 1200 1500 1800 2100 2400 2700
do
    python evaluate.py \
        --checkpoint $i \
        --chatgpt_model "gpt-3.5-turbo" \
        --gpt_steps 1 \
        --num_gpus 4 \
        --gpu_utils 0.5
done

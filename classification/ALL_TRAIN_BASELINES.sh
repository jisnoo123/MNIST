#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

rm -rf checkpoint

rm -rf inference_folder

rm -rf inference

rm -rf history

######################################################################################################


python3 train_model.py --lr 1e-04 --batch_size 128 --num_epochs 2 --train_images train \
        --val_images val --test_images test  --history_folder_name history \
        --chkpt_name model_ckp --save_after_epoch 2

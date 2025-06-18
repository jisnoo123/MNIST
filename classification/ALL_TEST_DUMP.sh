#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

rm -rf inference_folder

python3 test_and_dump.py --test test --lr 1e-04 --num_epochs 200 \
        --chkpt_name model_ckp_2.pt --history_folder_name history \
        --metrics_folder metrics --umap_layer fc2

python3 plot_metrics.py --metrics_folder metrics

######################################################

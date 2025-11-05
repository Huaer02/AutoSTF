#!/bin/bash

# AutoSTF架构搜索+训练一体化脚本 - 适配NPZ数据格式
# 用于在bjs_True_True_0_small数据集上先进行架构搜索，然后进行模型训练

export CUDA_VISIBLE_DEVICES=0

echo "Starting AutoSTF architecture search and training on bjs_True_True_0_small dataset..."
echo "=============================================================================="

# # 第一步：架构搜索
# echo "Step 1: Architecture Search"
# echo "----------------------------"
# python search_npz.py \
#     --device cuda:0 \
#     --dataset bjs_True_True_0_small \
#     --settings bjs_True_True_0_small \
#     --scale_num 3 \
#     --in_channels 3 \
#     --num_mlp_layers 2 \
#     --mode_name TWO_PATHS \
#     --epochs 100 \
#     --model_des bjs_0_AutoSTF

# if [ $? -eq 0 ]; then
#     echo "Architecture search completed successfully!"
# else
#     echo "Architecture search failed! Exiting..."
#     exit 1
# fi

echo ""
echo "Step 2: Model Training"
echo "----------------------"
# 第二步：模型训练
python train_npz.py \
    --device cuda:0 \
    --dataset bjs_True_True_0_small \
    --settings bjs_True_True_0_small \
    --scale_num 3 \
    --in_channels 3 \
    --num_mlp_layers 2 \
    --epochs 100 \
    --run_times 1 \
    --model_des bjs_0_AutoSTF \
    --save_model

if [ $? -eq 0 ]; then
    echo "Model training completed successfully!"
else
    echo "Model training failed!"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "AutoSTF search and training pipeline completed for bjs_True_True_0_small!"
echo ""
echo "Dataset info: 270 nodes, ~504MB memory usage"
echo "Log files saved in ./logs/"
echo "Model parameters saved in ./model_param/bjs_True_True_0_small/"
echo "Search model: bjs_True_True_0_small_best_search_model_bjs_0_AutoSTF.pt"
echo "Trained model: bjs_True_True_0_small_best_train_model_bjs_0_AutoSTF_run_0.pt"

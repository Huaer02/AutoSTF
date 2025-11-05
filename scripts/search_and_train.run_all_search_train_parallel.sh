#!/bin/bash

# 批量并行运行所有数据集的架构搜索+训练一体化脚本
# 适用于多GPU并行执行

echo "=========================================="
echo "Starting parallel search and training for all datasets"
echo "=========================================="

# 检查GPU数量
gpu_count=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $gpu_count"

if [ $gpu_count -lt 6 ]; then
    echo "Warning: Need at least 6 GPUs for parallel search and training"
    echo "Available GPUs: $gpu_count"
    echo "Some jobs may queue or fail"
fi

# 记录开始时间
start_time=$(date)
echo "Start time: $start_time"

echo ""
echo "Starting all search and training jobs in parallel..."

# 启动所有搜索+训练任务（后台运行）
bash ./scripts/search_and_train/run_bjs_0_search_train.sh > logs/search_train_bjs_0.out 2>&1 &
pid_bjs_0=$!
echo "Started bjs_0 search+train (PID: $pid_bjs_0) on GPU 0"

bash ./scripts/search_and_train/run_bjs_8_search_train.sh > logs/search_train_bjs_8.out 2>&1 &
pid_bjs_8=$!
echo "Started bjs_8 search+train (PID: $pid_bjs_8) on GPU 1"

bash ./scripts/search_and_train/run_guomao_0_search_train.sh > logs/search_train_guomao_0.out 2>&1 &
pid_guomao_0=$!
echo "Started guomao_0 search+train (PID: $pid_guomao_0) on GPU 2"

bash ./scripts/search_and_train/run_guomao_8_search_train.sh > logs/search_train_guomao_8.out 2>&1 &
pid_guomao_8=$!
echo "Started guomao_8 search+train (PID: $pid_guomao_8) on GPU 3"

bash ./scripts/search_and_train/run_xyl_0_search_train.sh > logs/search_train_xyl_0.out 2>&1 &
pid_xyl_0=$!
echo "Started xyl_0 search+train (PID: $pid_xyl_0) on GPU 4"

bash ./scripts/search_and_train/run_xyl_8_search_train.sh > logs/search_train_xyl_8.out 2>&1 &
pid_xyl_8=$!
echo "Started xyl_8 search+train (PID: $pid_xyl_8) on GPU 5"

echo ""
echo "All search and training jobs started! Monitoring progress..."
echo "You can monitor individual jobs with:"
echo "  tail -f logs/search_train_*.out"

# 等待所有任务完成
jobs=("bjs_0:$pid_bjs_0" "bjs_8:$pid_bjs_8" "guomao_0:$pid_guomao_0" "guomao_8:$pid_guomao_8" "xyl_0:$pid_xyl_0" "xyl_8:$pid_xyl_8")

echo ""
echo "Waiting for all search and training jobs to complete..."
echo "This may take several hours depending on the datasets..."

for job in "${jobs[@]}"; do
    name=$(echo $job | cut -d: -f1)
    pid=$(echo $job | cut -d: -f2)

    echo "Waiting for $name search+train (PID: $pid)..."
    wait $pid
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✅ $name search and training completed successfully"
    else
        echo "❌ $name search and training failed with exit code $exit_code"
    fi
done

# 记录结束时间
end_time=$(date)
echo ""
echo "=========================================="
echo "All parallel search and training completed!"
echo "Start time: $start_time"
echo "End time: $end_time"
echo "=========================================="

echo ""
echo "Check results in:"
echo "  - Training logs: ./logs/"
echo "  - Script outputs: ./logs/search_train_*.out"
echo "  - Search models: ./model_param/{dataset}/*_best_search_model_*.pt"
echo "  - Trained models: ./model_param/{dataset}/*_best_train_model_*.pt"

echo ""
echo "Summary of generated models:"
for dataset in bjs_True_True_0_small bjs_True_True_8_small guomao_True_True_0_small guomao_True_True_8_small xyl_True_True_0_small xyl_True_True_8_small; do
    model_dir="./model_param/$dataset"
    if [ -d "$model_dir" ]; then
        echo "📁 $dataset:"
        ls -la "$model_dir"/*.pt 2>/dev/null || echo "  No model files found"
    fi
done

echo ""
echo "GPU usage summary:"
nvidia-smi

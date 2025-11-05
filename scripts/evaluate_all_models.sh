#!/bin/bash

# 批量评测所有训练好的模型
# 自动加载saved_models中的模型，在测试集上运行并生成评测结果

echo "=========================================="
echo "Starting batch evaluation for all trained models"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# 检查必要的文件
if [ ! -f "evaluate_model.py" ]; then
    echo "Error: evaluate_model.py not found"
    exit 1
fi

if [ ! -f "ev.py" ]; then
    echo "Error: ev.py not found"
    exit 1
fi

# 检查saved_models目录
if [ ! -d "saved_models" ]; then
    echo "Error: saved_models directory not found"
    exit 1
fi

# 统计模型文件数量
model_count=$(find saved_models -name "*.pth" | wc -l)
echo "Found $model_count model files in saved_models/"

if [ $model_count -eq 0 ]; then
    echo "No model files found in saved_models/"
    exit 1
fi

# 创建评测结果目录
mkdir -p evaluation_results
mkdir -p evaluation_results/csv_reports

# 记录开始时间
start_time=$(date)
echo "Start time: $start_time"

echo ""
echo "Running batch evaluation..."

# 检查数据目录
if [ ! -d "data" ]; then
    echo "Warning: data directory not found, creating it..."
    mkdir -p data
fi

# 检查设置文件目录
if [ ! -d "model_settings" ]; then
    echo "Error: model_settings directory not found"
    exit 1
fi

# 运行批量评测
echo "Command: python evaluate_model.py --batch_eval --model_dir ./saved_models --save_dir ./evaluation_results --device cuda:0 --auto_eval"

python evaluate_model.py \
    --batch_eval \
    --model_dir ./saved_models \
    --save_dir ./evaluation_results \
    --device cuda:0 \
    --auto_eval

# 检查评测结果
eval_exit_code=$?
if [ $eval_exit_code -eq 0 ]; then
    echo "✅ Batch evaluation completed successfully"
else
    echo "❌ Batch evaluation failed with exit code: $eval_exit_code"
    echo "Please check the error messages above"
    exit 1
fi

# 记录结束时间
end_time=$(date)
echo ""
echo "=========================================="
echo "Batch evaluation completed!"
echo "Start time: $start_time"
echo "End time: $end_time"
echo "=========================================="

echo ""
echo "Results summary:"
echo "📁 NPZ files: ./evaluation_results/*.npz"
echo "📊 CSV reports: ./evaluation_results/*.csv"

# 统计生成的文件
npz_count=$(find evaluation_results -name "*.npz" | wc -l)
csv_count=$(find evaluation_results -name "*.csv" | wc -l)

echo "Generated $npz_count NPZ result files"
echo "Generated $csv_count CSV report files"

echo ""
echo "CSV reports summary:"
echo "===================="

# 显示所有CSV文件的内容
for csv_file in evaluation_results/*.csv; do
    if [ -f "$csv_file" ]; then
        echo ""
        echo "📊 $(basename "$csv_file"):"
        echo "----------------------------------------"
        cat "$csv_file"
    fi
done

echo ""
echo "🎉 All evaluations completed!"

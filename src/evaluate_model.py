#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评测脚本
加载训练好的模型权重，在测试集上运行并保存预测结果，然后进行评测
"""

import torch
import numpy as np
import argparse
import logging
import os
import sys
import glob
import subprocess
from pathlib import Path

from src.settings import Settings
from src.trainer_adapter import NPZTrainer
from src.model.TrafficForecasting import AutoSTF
from src.DataProcessingAdapter import NPZDataProcessing
from src.model.mode import Mode
from ev import write_result


def setup_logger(log_filename=None):
    """设置日志系统，同时输出到控制台和文件"""
    import datetime

    # 创建logs目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名
    if log_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/evaluation_{timestamp}.log"

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的handlers（避免重复）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 记录日志文件路径
    logger.info(f"日志文件保存到: {log_filename}")
    logger.info("=" * 80)

    return log_filename


def load_model_and_evaluate(model_path, dataset, settings_file, device, save_dir):
    """
    加载模型并在测试集上评测

    Args:
        model_path: 模型权重文件路径
        dataset: 数据集名称
        settings_file: 设置文件名称
        device: 设备
        save_dir: 保存预测结果的目录
    """
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Settings: {settings_file}")

    # 加载设置
    settings = Settings()
    settings.load_settings(settings_file)

    # 数据处理
    logging.info("Loading dataset...")
    NPZdata = NPZDataProcessing(
        dataset=dataset,
        train_prop=settings.data.train_prop,
        valid_prop=settings.data.valid_prop,
        num_sensors=settings.data.num_sensors,
        in_length=settings.data.in_length,
        out_length=settings.data.out_length,
        in_channels=3,  # 默认使用3个通道
        batch_size_per_gpu=settings.data.batch_size,
    )

    scaler = NPZdata.scaler
    dataloader = NPZdata.dataloader
    adj_mx_gwn = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_gwn]
    adj_mx = [torch.tensor(NPZdata.adj_mx_dcrnn).to(device), adj_mx_gwn, torch.tensor(NPZdata.adj_mx_01).to(device)]

    # 设置mask支持的邻接矩阵
    if dataset in ["PEMS-BAY", "METR-LA"]:
        mask_support_adj = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_gwn]
    else:
        mask_support_adj = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_01]

    # 计算scale列表
    scale_list = []
    for i in range(3):  # 默认scale_num=3
        scale_list.append(int(settings.data.in_length / 3))

    # 创建模型配置
    class Config:
        def __init__(self):
            self.scale_list = scale_list
            self.num_sensors = settings.data.num_sensors
            self.in_length = settings.data.in_length
            self.hidden_channels = settings.model.hidden_channels
            self.num_mlp_layers = settings.model.num_mlp_layers
            self.scale_num = settings.model.scale_num
            self.IsUseLinear = settings.model.IsUseLinear
            self.num_linear_layers = settings.model.num_linear_layers
            self.layer_names = settings.model.layer_names
            self.num_temporal_search_node = settings.model.num_temporal_search_node
            self.temporal_operations = settings.model.temporal_operations
            self.num_spatial_search_node = settings.model.num_spatial_search_node
            self.spatial_operations = settings.model.spatial_operations
            self.num_att_layers = settings.model.num_att_layers
            self.num_hop = settings.model.num_hop

    config = Config()

    # 初始化模型
    model = AutoSTF(
        in_length=settings.data.in_length,
        out_length=settings.data.out_length,
        mask_support_adj=mask_support_adj,
        adj_mx=adj_mx,
        num_sensors=settings.data.num_sensors,
        in_channels=3,
        out_channels=settings.data.out_channels,
        hidden_channels=settings.model.hidden_channels,
        end_channels=settings.model.end_channels,
        layer_names=settings.model.layer_names,
        config=config,
        device=device,
    )

    # 初始化训练器
    trainer = NPZTrainer(model, settings, scaler, device)

    # 加载模型权重
    try:
        if model_path.endswith(".pth"):
            # 加载saved_models中的模型
            checkpoint = torch.load(model_path, map_location=device)
            if "net" in checkpoint:
                model.load_state_dict(checkpoint["net"])
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"Loaded model from {model_path}")
        elif model_path.endswith(".pt"):
            # 加载model_param中的模型
            best_epoch = trainer.load(model_path)
            logging.info(f"Loaded model from {model_path} (epoch {best_epoch})")
        else:
            raise ValueError(f"Unsupported model file format: {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

    # 在测试集上评测
    logging.info("Running evaluation on test set...")
    model.eval()

    all_predictions = []
    all_truths = []

    test_loader = dataloader["test_loader"].get_iterator()

    with torch.no_grad():
        for x, y in test_loader:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            # 使用ONE_PATH_FIXED模式进行预测
            model.set_mode(Mode.ONE_PATH_FIXED)
            pred = model(x, mode=Mode.ONE_PATH_FIXED)
            # pred = trainer.

            # 反标准化预测结果，真实标签不需要反标准化
            pred_denorm = scaler.inverse_transform(pred.cpu().numpy())
            truth_original = y.cpu().numpy()  # 真实标签本身就是原始数据

            all_predictions.append(pred_denorm)
            all_truths.append(truth_original)

    # 合并所有预测结果
    predictions = np.concatenate(all_predictions, axis=0)
    truths = np.concatenate(all_truths, axis=0)

    logging.info(f"Prediction shape: {predictions.shape}")
    logging.info(f"Truth shape: {truths.shape}")

    # 保存预测结果
    os.makedirs(save_dir, exist_ok=True)

    # 生成保存文件名
    model_name = Path(model_path).stem
    save_path = os.path.join(save_dir, f"{dataset}_{model_name}_results.npz")

    np.savez(save_path, prediction=predictions, truth=truths, dataset=dataset, model_path=model_path)

    logging.info(f"Results saved to: {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AutoSTF models")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file (.pth or .pt)")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., guomao_True_True_0_small)")
    parser.add_argument("--settings", type=str, help="Settings file name (e.g., guomao_True_True_0_small)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)")
    parser.add_argument(
        "--save_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results"
    )
    parser.add_argument("--auto_eval", action="store_true", help="Automatically run ev.py after generating results")

    # 批量评测选项
    parser.add_argument("--batch_eval", action="store_true", help="Batch evaluate all models in saved_models directory")
    parser.add_argument(
        "--model_dir", type=str, default="./saved_models", help="Directory containing model files for batch evaluation"
    )

    args = parser.parse_args()

    # 检查参数
    if not args.batch_eval:
        if not args.model_path or not args.dataset or not args.settings:
            parser.error("--model_path, --dataset, and --settings are required when not using --batch_eval")

    setup_logger()

    device = torch.device(args.device)

    if args.batch_eval:
        # 批量评测模式
        logging.info("Starting batch evaluation...")

        # 查找所有模型文件
        model_files = glob.glob(os.path.join(args.model_dir, "*.pth"))
        model_files.extend(glob.glob(os.path.join(args.model_dir, "*.pt")))

        if not model_files:
            logging.error(f"No model files found in {args.model_dir}")
            return

        logging.info(f"Found {len(model_files)} model files")

        results_files = []

        for model_path in model_files:
            model_name = Path(model_path).stem
            logging.info(f"\nEvaluating model: {model_name}")

            # 从模型名称推断数据集和设置
            # 假设模型名称格式为: {dataset}_{model_des}_best
            parts = model_name.split("_")
            if len(parts) >= 3:
                # num = parts[3]
                num = parts[1]
                # 重构数据集名称
                if "bjs" in parts[0]:
                    dataset = f"bjs_True_True_{num}_small"
                elif "guomao" in parts[0]:
                    dataset = f"guomao_True_True_{num}_small"
                elif "xyl" in parts[0]:
                    dataset = f"xyl_True_True_{num}_small"
                else:
                    logging.warning(f"Cannot infer dataset from model name: {model_name}")
                    continue

                settings_file = dataset

                try:
                    result_path = load_model_and_evaluate(model_path, dataset, settings_file, device, args.save_dir)
                    if result_path:
                        results_files.append(result_path)
                except Exception as e:
                    logging.error(f"Failed to evaluate {model_name}: {e}")
                    continue
            else:
                logging.warning(f"Cannot parse model name: {model_name}")
                continue

        logging.info(f"\nBatch evaluation completed. Generated {len(results_files)} result files.")

        # 自动运行ev.py
        if args.auto_eval and results_files:
            logging.info("Running ev.py on all results...")
            try:
                subprocess.run(["python", "ev.py", "--path", args.save_dir], check=True)
                logging.info("ev.py completed successfully")
            except subprocess.CalledProcessError as e:
                logging.error(f"ev.py failed: {e}")

    else:
        # 单个模型评测模式
        result_path = load_model_and_evaluate(args.model_path, args.dataset, args.settings, device, args.save_dir)

        if result_path and args.auto_eval:
            # 自动运行ev.py
            logging.info("Running ev.py on the result...")
            try:
                write_result(result_path)
                logging.info("Evaluation completed successfully")

                # 显示CSV结果
                csv_path = result_path.replace(".npz", ".csv")
                if os.path.exists(csv_path):
                    logging.info(f"CSV results saved to: {csv_path}")
                    with open(csv_path, "r") as f:
                        content = f.read()
                        logging.info("Evaluation metrics:")
                        logging.info(content)
            except Exception as e:
                logging.error(f"ev.py failed: {e}")


if __name__ == "__main__":
    main()

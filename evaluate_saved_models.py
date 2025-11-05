#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版模型评测脚本
专门用于评测saved_models目录中的模型文件
"""

import torch
import numpy as np
import logging
import os
import glob
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
    import sys

    # 创建logs目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名
    if log_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_dir}/evaluation_saved_models_{timestamp}.log"

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


def parse_model_name(model_path):
    """从模型文件名解析数据集信息"""
    model_name = Path(model_path).stem
    logging.info(f"解析模型名称: {model_name}")

    # 解析模型名称格式: {dataset_prefix}_{variant}_AutoSTF_best
    parts = model_name.split("_")

    if len(parts) >= 2:
        prefix = parts[0]  # bjs, guomao, xyl
        variant = parts[1]  # 0, 8

        # 构造完整的数据集名称
        dataset = f"{prefix}_True_True_{variant}_small"
        settings = dataset

        logging.info(f"推断数据集: {dataset}")
        return dataset, settings
    else:
        logging.error(f"无法解析模型名称: {model_name}")
        return None, None


def evaluate_single_model(model_path, device="cuda:0", save_dir="./evaluation_results"):
    """评测单个模型"""
    logging.info(f"开始评测模型: {model_path}")

    # 解析数据集信息
    dataset, settings_file = parse_model_name(model_path)
    if not dataset or not settings_file:
        logging.error(f"无法解析模型 {model_path}")
        return None

    # 检查设置文件是否存在
    settings_path = f"model_settings/{settings_file}.yaml"
    if not os.path.exists(settings_path):
        logging.error(f"设置文件不存在: {settings_path}")
        return None

    # 检查数据目录是否存在
    # data_path = f"data/{dataset}"
    # if not os.path.exists(data_path):
    #     logging.error(f"数据目录不存在: {data_path}")
    #     return None

    try:
        # 加载设置
        settings = Settings()
        settings.load_settings(settings_file)

        # 数据处理
        logging.info("加载数据集...")
        NPZdata = NPZDataProcessing(
            dataset=dataset,
            train_prop=settings.data.train_prop,
            valid_prop=settings.data.valid_prop,
            num_sensors=settings.data.num_sensors,
            in_length=settings.data.in_length,
            out_length=settings.data.out_length,
            in_channels=3,
            batch_size_per_gpu=settings.data.batch_size,
        )

        scaler = NPZdata.scaler
        dataloader = NPZdata.dataloader
        adj_mx_gwn = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_gwn]
        adj_mx = [torch.tensor(NPZdata.adj_mx_dcrnn).to(device), adj_mx_gwn, torch.tensor(NPZdata.adj_mx_01).to(device)]

        # 设置mask支持的邻接矩阵
        mask_support_adj = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_01]

        # 计算scale列表
        scale_list = []
        for i in range(3):
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

        # 加载模型权重
        logging.info(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        if "net" in checkpoint:
            model.load_state_dict(checkpoint["net"])
            logging.info("成功加载模型权重 (net)")
        else:
            model.load_state_dict(checkpoint)
            logging.info("成功加载模型权重 (direct)")

        # 在测试集上评测
        logging.info("在测试集上运行评测...")
        model.eval()

        all_predictions = []
        all_truths = []

        test_loader = dataloader["test_loader"].get_iterator()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x = torch.tensor(x, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.float32).to(device)

                # 使用ONE_PATH_FIXED模式进行预测
                model.set_mode(Mode.ONE_PATH_FIXED)
                pred = model(x)

                # 反标准化预测结果，真实标签不需要反标准化
                pred_denorm = scaler.inverse_transform(pred.cpu().numpy())
                truth_original = y.cpu().numpy()  # 真实标签本身就是原始数据

                all_predictions.append(pred_denorm)
                all_truths.append(truth_original)

                if batch_idx % 10 == 0:
                    logging.info(f"处理批次 {batch_idx}")

        # 合并所有预测结果
        predictions = np.concatenate(all_predictions, axis=0)
        truths = np.concatenate(all_truths, axis=0)

        logging.info(f"预测结果形状: {predictions.shape}")
        logging.info(f"真实标签形状: {truths.shape}")

        # 保存预测结果
        os.makedirs(save_dir, exist_ok=True)

        model_name = Path(model_path).stem
        save_path = os.path.join(save_dir, f"{dataset}_{model_name}_results.npz")

        np.savez(save_path, prediction=predictions, truth=truths, dataset=dataset, model_path=model_path)

        logging.info(f"结果已保存到: {save_path}")

        # 计算评测指标
        logging.info("计算评测指标...")
        write_result(save_path)

        csv_path = save_path.replace(".npz", ".csv")
        if os.path.exists(csv_path):
            logging.info(f"评测报告已保存到: {csv_path}")
            with open(csv_path, "r") as f:
                content = f.read()
                logging.info("评测指标:")
                logging.info(content)

        return save_path

    except Exception as e:
        logging.error(f"评测模型 {model_path} 时出错: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    log_filename = setup_logger()

    logging.info("🚀 开始批量评测saved_models中的模型")

    # 查找所有模型文件
    model_files = glob.glob("saved_models/*.pth")

    if not model_files:
        logging.error("在saved_models/目录中未找到.pth文件")
        return

    logging.info(f"找到 {len(model_files)} 个模型文件")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    save_dir = "./evaluation_results"
    os.makedirs(save_dir, exist_ok=True)

    success_count = 0

    for i, model_path in enumerate(model_files, 1):
        logging.info(f"\n[{i}/{len(model_files)}] 评测模型: {os.path.basename(model_path)}")

        result = evaluate_single_model(model_path, device, save_dir)
        if result:
            success_count += 1
            logging.info(f"✅ 模型 {os.path.basename(model_path)} 评测成功")
        else:
            logging.error(f"❌ 模型 {os.path.basename(model_path)} 评测失败")

    logging.info(f"\n🎉 批量评测完成!")
    logging.info(f"成功评测: {success_count}/{len(model_files)} 个模型")
    logging.info(f"详细日志已保存到: {log_filename}")

    # 显示所有CSV结果
    csv_files = glob.glob(os.path.join(save_dir, "*.csv"))
    if csv_files:
        logging.info(f"\n📊 评测结果汇总:")
        for csv_file in csv_files:
            logging.info(f"\n{os.path.basename(csv_file)}:")
            with open(csv_file, "r") as f:
                content = f.read()
                logging.info(content)


if __name__ == "__main__":
    main()

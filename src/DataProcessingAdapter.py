import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler

from src.utils.helper import Scaler
from src import train_util


class NPZDataProcessing:
    """
    适配NPZ格式数据集到AutoSTF的数据处理类
    支持guomao_True_True_0_small等数据集格式
    """

    def __init__(
        self, dataset, train_prop, valid_prop, num_sensors, in_length, out_length, in_channels, batch_size_per_gpu
    ):

        self.traffic_data = {}
        self.dataset = dataset
        self.dataset_path = f"../../data/{dataset}"  # 构造数据路径
        self.num_sensors = num_sensors

        self.train_prop = train_prop
        self.valid_prop = valid_prop

        self.in_length = in_length
        self.out_length = out_length
        self.in_channels = in_channels
        self.batch_size = batch_size_per_gpu
        self.adj_type = "doubletransition"

        # 读取邻接矩阵
        self.adj_mx_01, self.adj_mx_dcrnn, self.adj_mx_gwn = self.read_adj_mat()

        self.dataloader = {}

        # 构建数据加载器
        self.build_data_loader()

    def build_data_loader(self):
        logging.info("initialize NPZ data loader")

        # 从NPZ文件读取数据
        train_data, valid_data, test_data = self.read_npz_data()

        # 使用训练数据的第一个特征维度来初始化scaler
        # 数据格式: [num_samples, input_time_steps, num_nodes, 4]
        # 第0维是特征值，用于初始化scaler
        train_values = train_data["x"][:, :, :, 0]  # 只使用第一个特征维度（特征值）
        self.scaler = Scaler(train_values.reshape(-1), missing_value=0)

        # 为搜索和训练创建数据加载器
        self.search_train = self.get_data_loader_from_npz(train_data, shuffle=True, tag="search_train")
        self.search_valid = self.get_data_loader_from_npz(valid_data, shuffle=True, tag="search_valid")

        self.train = self.get_data_loader_from_npz(train_data, shuffle=True, tag="train")
        self.valid = self.get_data_loader_from_npz(valid_data, shuffle=False, tag="valid")
        self.test = self.get_data_loader_from_npz(test_data, shuffle=False, tag="test")

    def read_npz_data(self):
        """读取NPZ格式的数据文件"""
        train_file = os.path.join(self.dataset_path, "train.npz")
        val_file = os.path.join(self.dataset_path, "val.npz")
        test_file = os.path.join(self.dataset_path, "test.npz")

        train_data = np.load(train_file)
        val_data = np.load(val_file)
        test_data = np.load(test_file)

        return (
            {"x": train_data["x"], "y": train_data["y"]},
            {"x": val_data["x"], "y": val_data["y"]},
            {"x": test_data["x"], "y": test_data["y"]},
        )

    def get_data_loader_from_npz(self, data_dict, shuffle, tag):
        """从NPZ数据创建数据加载器

        数据格式说明:
        x_data: [num_samples, input_time_steps, num_nodes, 4]
        - 维度0: 特征值 (4083个样本)
        - 维度1: 输入时间步骤的Linux时间戳 (12个时间步)
        - 维度2: 节点数或mask/adj信息 (65个节点)
        - 维度3: 输出时间步骤的Linux时间戳 (4个时间步)
        """
        if len(data_dict["x"]) == 0:
            return 0

        x_data = data_dict["x"]  # [num_samples, input_time_steps, num_nodes, 4]
        y_data = data_dict["y"]  # [num_samples, output_time_steps, num_nodes, 4]

        # 存储原始数据用于后续处理
        self.traffic_data[tag + "_data"] = data_dict
        self.traffic_data[tag + "_raw_x"] = x_data  # 保存原始x数据
        self.traffic_data[tag + "_raw_y"] = y_data  # 保存原始y数据

        num_samples, input_time_steps, num_nodes, raw_features = x_data.shape
        _, output_time_steps, _, _ = y_data.shape

        logging.info(f"Processing {tag} data: x_shape={x_data.shape}, y_shape={y_data.shape}")

        # 从x_data中提取特征和时间信息
        # 假设x_data的结构是: [特征值, 输入时间戳, mask/adj, 输出时间戳]
        traffic_features = x_data[:, :, :, 0]  # [num_samples, input_time_steps, num_nodes] - 特征值
        input_timestamps = x_data[:, :, :, 1]  # [num_samples, input_time_steps, num_nodes] - 输入时间戳
        mask_adj_info = x_data[:, :, :, 2]  # [num_samples, input_time_steps, num_nodes] - mask/adj信息
        output_timestamps = x_data[:, :, :, 3]  # [num_samples, input_time_steps, num_nodes] - 输出时间戳

        # 从时间戳中提取时间特征
        input_time_features = self.extract_temporal_features_from_timestamps(input_timestamps)

        # 构建输入特征
        if self.in_channels == 1:
            # 只使用交通流量特征
            inputs = np.expand_dims(traffic_features, axis=-1)  # [num_samples, input_time_steps, num_nodes, 1]
        elif self.in_channels == 2:
            # 使用交通流量 + HoD特征
            hod_features = input_time_features["hour_of_day"]  # [num_samples, input_time_steps, num_nodes]
            inputs = np.stack(
                [traffic_features, hod_features], axis=-1
            )  # [num_samples, input_time_steps, num_nodes, 2]
        elif self.in_channels == 3:
            # 使用交通流量 + HoD + DoW特征
            hod_features = input_time_features["hour_of_day"]  # [num_samples, input_time_steps, num_nodes]
            dow_features = input_time_features["day_of_week"]  # [num_samples, input_time_steps, num_nodes]
            inputs = np.stack(
                [traffic_features, hod_features, dow_features], axis=-1
            )  # [num_samples, input_time_steps, num_nodes, 3]
        else:
            logging.error(f"Unsupported in_channels: {self.in_channels}")
            sys.exit()

        # 标准化输入数据（只标准化第一个特征维度 - 交通流量）
        inputs_normalized = inputs.copy()
        inputs_normalized[..., 0] = self.scaler.transform(inputs[..., 0])

        # 处理输出数据 - 从y_data中提取特征值
        # 假设y_data的结构与x_data类似，我们只需要特征值部分
        labels = np.expand_dims(y_data[:, :, :, 0], axis=-1)  # [num_samples, output_time_steps, num_nodes, 1]

        # 转换维度顺序以匹配AutoSTF期望: [B, C, N, T]
        inputs_final = inputs_normalized.transpose(0, 3, 2, 1)  # [num_samples, channels, num_nodes, input_time_steps]
        labels_final = labels.transpose(0, 3, 2, 1)  # [num_samples, 1, num_nodes, output_time_steps]

        logging.info("load %s inputs & labels [ok]", tag)
        logging.info("input shape: %s", inputs_final.shape)
        logging.info("label shape: %s", labels_final.shape)

        # 创建PyTorch数据集
        dataset = TensorDataset(
            torch.from_numpy(inputs_final).to(dtype=torch.float), torch.from_numpy(labels_final).to(dtype=torch.float)
        )

        # 创建采样器
        if shuffle:
            sampler = RandomSampler(dataset, replacement=True, num_samples=self.batch_size)
        else:
            sampler = SequentialSampler(dataset)

        # 创建数据加载器
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4, drop_last=False
        )

        # 存储数据加载器和处理后的数据
        self.dataloader[tag + "_loader"] = DataLoaderM(inputs_final, labels_final, self.batch_size)
        self.dataloader["x_" + tag] = inputs_final
        self.dataloader["y_" + tag] = labels_final

        # 存储额外的信息供后续使用
        self.dataloader[tag + "_mask_adj"] = mask_adj_info  # 保存mask/adj信息
        self.dataloader[tag + "_input_timestamps"] = input_timestamps  # 保存输入时间戳
        self.dataloader[tag + "_output_timestamps"] = output_timestamps  # 保存输出时间戳
        self.dataloader[tag + "_time_features"] = input_time_features  # 保存提取的时间特征

        return data_loader

    def extract_temporal_features_from_timestamps(self, timestamps):
        """
        从Linux时间戳中提取时间特征，参考processor.py中的timestamps_to_features方法

        Args:
            timestamps: [num_samples, time_steps, num_nodes] - Linux时间戳

        Returns:
            dict: 包含时间特征的字典
        """
        # 获取时间戳的形状
        num_samples, time_steps, num_nodes = timestamps.shape

        # 假设同一时间步的所有节点共享相同的时间戳（这是时间序列数据的常见情况）
        # 我们只需要处理 [num_samples, time_steps] 的时间戳，然后扩展到所有节点

        # 取第一个节点的时间戳作为代表（假设所有节点在同一时间步的时间戳相同）
        representative_timestamps = timestamps[:, :, 0]  # [num_samples, time_steps]

        # 展平时间戳以便批量处理
        timestamps_flat = representative_timestamps.reshape(-1)  # [num_samples * time_steps]

        # 使用pandas进行向量化处理，参考processor.py的方法
        try:
            # 转换为pandas DataFrame进行批量处理
            df = pd.DataFrame({"timestamp": timestamps_flat})
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

            # 提取时间特征 - 修复：使用一天的小数部分，与原始DataProcessing.py保持一致
            # 计算一天中的时间小数部分 [0, 1)，参考DataProcessing.py第88行
            time_in_day = (df["datetime"] - df["datetime"].dt.floor("D")) / pd.Timedelta(days=1)
            timeofday = time_in_day.values  # [0, 1) 的小数
            dayofweek = df["datetime"].dt.dayofweek.values  # 0=Monday, 6=Sunday

            # 重新整形为 [num_samples, time_steps]
            timeofday = timeofday.reshape(num_samples, time_steps)
            dayofweek = dayofweek.reshape(num_samples, time_steps)

            # 确保时间特征在有效范围内
            timeofday = np.clip(timeofday, 0.0, 0.999999).astype(np.float32)  # 一天的小数部分: [0, 1)
            dayofweek = np.clip(dayofweek, 0, 6).astype(np.float32)  # 星期: 0-6

            # 扩展到所有节点 [num_samples, time_steps, num_nodes]
            hour_of_day = np.tile(timeofday[:, :, np.newaxis], (1, 1, num_nodes)).astype(np.float32)
            day_of_week = np.tile(dayofweek[:, :, np.newaxis], (1, 1, num_nodes)).astype(np.float32)

            # 调试信息：检查特征范围
            print(f"时间特征范围检查:")
            print(f"  hour_of_day: [{hour_of_day.min():.6f}, {hour_of_day.max():.6f}] (一天的小数部分)")
            print(f"  day_of_week: [{day_of_week.min():.1f}, {day_of_week.max():.1f}] (0=Monday, 6=Sunday)")

            # 验证时间特征的有效性
            if hour_of_day.min() < 0 or hour_of_day.max() >= 1.0:
                print(f"错误：hour_of_day超出有效范围[0, 1)")
            if day_of_week.min() < 0 or day_of_week.max() > 6:
                print(f"错误：day_of_week超出有效范围[0, 6]")

        except (ValueError, OSError) as e:
            # 如果时间戳处理失败，使用默认值
            print(f"警告：时间戳处理失败，使用默认值。错误：{e}")
            hour_of_day = np.zeros((num_samples, time_steps, num_nodes), dtype=np.float32)
            day_of_week = np.zeros((num_samples, time_steps, num_nodes), dtype=np.float32)

        return {"hour_of_day": hour_of_day, "day_of_week": day_of_week}

    def add_temporal_features(self, data, num_channels):
        """为数据添加时间特征，参考processor.py中的timestamps_to_features方法"""
        batch_size, time_steps, num_nodes, orig_features = data.shape

        # 创建新的数据数组
        new_data = np.zeros((batch_size, time_steps, num_nodes, num_channels))

        # 复制原始特征
        new_data[..., :orig_features] = data

        # 生成模拟时间戳序列，假设数据是连续的时间序列，每个时间步代表5分钟
        base_time = pd.Timestamp("2023-01-01 00:00:00")  # 基准时间

        if num_channels >= 2:
            # Hour of day feature - 修复：使用一天的小数部分，与原始DataProcessing.py保持一致
            for b in range(batch_size):
                # 每个batch代表不同的时间段
                batch_start_time = base_time + pd.Timedelta(hours=b * 2)  # 每个batch间隔2小时
                hour_ft = []
                for t in range(time_steps):
                    current_time = batch_start_time + pd.Timedelta(minutes=t * 5)
                    # 计算一天中的时间小数部分 [0, 1)
                    time_in_day = (current_time - current_time.floor("D")) / pd.Timedelta(days=1)
                    hour_of_day = float(time_in_day)  # [0, 1) 的小数
                    hour_ft.append(hour_of_day)

                hour_ft = np.array(hour_ft, dtype=np.float32)
                # 确保在有效范围内
                hour_ft = np.clip(hour_ft, 0.0, 0.999999)
                # 扩展到所有节点，格式: [T, N]
                hour_ft_expanded = np.tile(hour_ft, (num_nodes, 1)).T  # [T, N]
                new_data[b, :, :, 1] = hour_ft_expanded

        if num_channels >= 3:
            # Day of week feature - 参考processor.py中的dayofweek处理
            # 使用dayofweek值 (0=Monday, 6=Sunday)，与processor.py保持一致
            for b in range(batch_size):
                # 每个batch代表不同的时间段
                batch_start_time = base_time + pd.Timedelta(hours=b * 2)  # 与时间特征保持一致
                day_ft = []
                for t in range(time_steps):
                    current_time = batch_start_time + pd.Timedelta(minutes=t * 5)
                    # 直接使用dayofweek，与processor.py中的dt.dayofweek保持一致
                    day_of_week = current_time.dayofweek  # 0=Monday, 6=Sunday
                    day_ft.append(day_of_week)

                day_ft = np.array(day_ft, dtype=np.float32)
                # 确保在有效范围内 [0, 6]
                day_ft = np.clip(day_ft, 0, 6)
                # 扩展到所有节点，格式: [T, N]
                day_ft_expanded = np.tile(day_ft, (num_nodes, 1)).T  # [T, N]
                new_data[b, :, :, 2] = day_ft_expanded

        return new_data

    def read_adj_mat(self):
        """读取邻接矩阵"""
        # 优先读取edge_adj_mx.npz，与processor.py保持一致
        adj_mx_file = os.path.join(self.dataset_path, "edge_adj_mx.npz")
        if os.path.exists(adj_mx_file):
            adj_mx_data = np.load(adj_mx_file)
            adj_mx_dcrnn = adj_mx_data["adj_mx"]
        else:
            # 如果没有edge_adj_mx.npz，尝试读取adj_mx.npz
            adj_mx_file = os.path.join(self.dataset_path, "adj_mx.npz")
            if os.path.exists(adj_mx_file):
                adj_mx_data = np.load(adj_mx_file)
                adj_mx_dcrnn = adj_mx_data["adj_mx"]
            else:
                # 如果都没有邻接矩阵，创建单位矩阵
                adj_mx_dcrnn = np.eye(self.num_sensors)

        # 创建0-1邻接矩阵
        adj_mx_01 = np.zeros((self.num_sensors, self.num_sensors))
        adj_mx_01[adj_mx_dcrnn > 0] = 1
        np.fill_diagonal(adj_mx_01, 1)

        # 创建GraphWaveNet格式的邻接矩阵
        adj_mx_gwn = [train_util.asym_adj(adj_mx_dcrnn), train_util.asym_adj(np.transpose(adj_mx_dcrnn))]

        return adj_mx_01, adj_mx_dcrnn, adj_mx_gwn


class DataLoaderM(object):
    """自定义数据加载器，兼容AutoSTF的接口"""

    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

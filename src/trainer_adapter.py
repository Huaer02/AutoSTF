import torch
import src.metrics_adapter as metrics_adapter
from src.model.mode import Mode


class NPZTrainer:
    """
    适配NPZ数据格式的训练器
    """

    def __init__(self, model, settings, scaler, device):
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.settings = settings
        self.device = device

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = settings.data.out_length
        self.max_value = scaler.max_value

        # 使用适配的损失函数
        self.loss = metrics_adapter.masked_mae

        self.weight_optimizer = torch.optim.Adam(
            self.model.weight_parameters(), lr=settings.trainer.weight_lr, weight_decay=settings.trainer.weight_decay
        )

        self.arch_optimizer = torch.optim.Adam(
            self.model.arch_parameters(), lr=settings.trainer.arch_lr, weight_decay=settings.trainer.arch_decay
        )

        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.weight_optimizer,
            milestones=settings.trainer.weight_lr_decay_milestones,
            gamma=settings.trainer.weight_lr_decay_ratio,
        )
        self.arch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.arch_optimizer,
            milestones=settings.trainer.arch_lr_decay_milestones,
            gamma=settings.trainer.arch_lr_decay_ratio,
        )

    def train_weight(self, input, real_val, mode=Mode.TWO_PATHS):
        """训练权重参数"""
        self.weight_optimizer.zero_grad()
        self.model.train()

        # input shape: [batch_size, in_channels, num_nodes, time_steps]
        output = self.model(input, mode)
        # output shape: [batch_size, out_channels, num_nodes, time_steps]

        # 确保输出和真实值的维度匹配
        if real_val.dim() == 3:
            real = torch.unsqueeze(real_val, dim=1)
        else:
            real = real_val

        # 反标准化预测结果
        predict = self.scaler.inverse_transform(output)

        # 限制预测值范围
        predict = torch.clamp(predict, min=0.0, max=self.max_value)

        # 计算损失
        loss = self.loss(predict, real, 0.0)
        loss.backward(retain_graph=False)

        # 计算评估指标
        mae = metrics_adapter.masked_mae(predict, real, 0.0).item()
        mape = metrics_adapter.masked_mape(predict, real, 0.0).item()
        rmse = metrics_adapter.masked_rmse(predict, real, 0.0).item()

        # 梯度裁剪和优化
        torch.nn.utils.clip_grad_norm_(self.model.weight_parameters(), self.settings.trainer.weight_clip_gradient)
        self.weight_optimizer.step()
        self.weight_optimizer.zero_grad()

        return loss.item(), mae, mape, rmse

    def train_arch(self, input, real_val, mode=Mode.TWO_PATHS):
        """训练架构参数"""
        self.model.train()
        self.arch_optimizer.zero_grad()

        # input shape: [batch_size, in_channels, num_nodes, time_steps]
        output = self.model(input, mode)
        # output shape: [batch_size, out_channels, num_nodes, time_steps]

        # 确保输出和真实值的维度匹配
        if real_val.dim() == 3:
            real = torch.unsqueeze(real_val, dim=1)
        else:
            real = real_val

        # 反标准化预测结果
        predict = self.scaler.inverse_transform(output)

        # 限制预测值范围
        predict = torch.clamp(predict, min=0.0, max=self.max_value)

        # 计算损失和指标
        loss = self.loss(predict, real, 0.0)
        mae = metrics_adapter.masked_mae(predict, real, 0.0).item()
        mape = metrics_adapter.masked_mape(predict, real, 0.0).item()
        rmse = metrics_adapter.masked_rmse(predict, real, 0.0).item()

        # 反向传播和优化
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.settings.trainer.arch_clip_gradient)
        self.arch_optimizer.step()

        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, mode=Mode.ONE_PATH_FIXED)

        # 确保输出和真实值的维度匹配
        if real_val.dim() == 3:
            real = torch.unsqueeze(real_val, dim=1)
        else:
            real = real_val

        # 反标准化预测结果
        predict = self.scaler.inverse_transform(output)

        # 限制预测值范围
        predict = torch.clamp(predict, min=0.0, max=self.max_value)

        # 计算损失和指标
        loss = self.loss(predict, real, 0.0)
        mae = metrics_adapter.masked_mae(predict, real, 0.0).item()
        mape = metrics_adapter.masked_mape(predict, real, 0.0).item()
        rmse = metrics_adapter.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mae, mape, rmse

    def infer(self, input):
        """推理"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, mode=Mode.ONE_PATH_FIXED)
        return output

    def load(self, model_path):
        """加载模型"""
        states = torch.load(model_path, map_location=self.device)

        # 加载网络参数
        self.model.load_state_dict(states["net"])

        # 加载优化器状态
        self.arch_optimizer.load_state_dict(states["arch_optimizer"])
        self.arch_scheduler.load_state_dict(states["arch_scheduler"])

        self.weight_optimizer.load_state_dict(states["weight_optimizer"])
        self.weight_scheduler.load_state_dict(states["weight_scheduler"])

        # 返回最佳epoch
        return states["best_epoch"]

    def save(self, model_path, epoch, best_epoch=None):
        """保存模型"""
        states = {
            "net": self.model.state_dict(),
            "arch_optimizer": self.arch_optimizer.state_dict(),
            "arch_scheduler": self.arch_scheduler.state_dict(),
            "weight_optimizer": self.weight_optimizer.state_dict(),
            "weight_scheduler": self.weight_scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch if best_epoch is not None else epoch,
        }
        torch.save(states, model_path)

    def step_schedulers(self):
        """更新学习率调度器"""
        self.weight_scheduler.step()
        self.arch_scheduler.step()

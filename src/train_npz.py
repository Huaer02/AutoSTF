import torch
import numpy as np
import argparse
import time
import logging
import os
import sys
import random
import wandb
import datetime
import warnings

from src.settings import Settings, load_server_config
from src import train_util
from src.trainer_adapter import NPZTrainer
from src.model.TrafficForecasting import AutoSTF
from src.DataProcessingAdapter import NPZDataProcessing
from src.model.mode import Mode
from src import metrics_adapter

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")


# 设置日志 - 优化版本
def setup_logger(dataset_name, model_des):
    """
    设置日志系统，同时输出到控制台和文件

    Args:
        dataset_name: 数据集名称
        model_des: 模型描述
    """
    # 创建logs目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 生成日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{dataset_name}_{model_des}_{timestamp}.log"

    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有的handlers（避免重复）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

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


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
parser.add_argument("--dataset", type=str, default="guomao_True_True_0_small", help="NPZ dataset name")
parser.add_argument("--settings", type=str, default="guomao_True_True_0_small", help="model settings file name")

parser.add_argument("--scale_num", type=int, default=3, help="number of scale")
parser.add_argument("--in_channels", type=int, default=3, help="number of input feature")
parser.add_argument("--num_mlp_layers", type=int, default=2, help="number of star mlp layer")

parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
parser.add_argument("--run_times", type=int, default=1, help="number of run")
parser.add_argument("--model_des", type=str, default="NPZ_AutoSTF", help="save model param")

# 新增参数
parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
parser.add_argument("--save_model", action="store_true", help="save trained model")
parser.add_argument("--load_model", type=str, default="", help="path to load pretrained model")

args = parser.parse_args()

# 设置日志系统
log_filename = setup_logger(args.dataset, args.model_des)

# 加载设置
settings = Settings()
settings.load_settings(args.settings)

server_config = load_server_config()

logging.info("Arguments:")
logging.info(f"{args}")
logging.info("Data settings:")
for key, value in settings.data_dict.items():
    logging.info(f"\t{key}: {value}")

logging.info("Trainer settings:")
for key, value in settings.trainer_dict.items():
    logging.info(f"\t{key}: {value}")

logging.info("Model settings:")
for key, value in settings.model_dict.items():
    if str(key) not in "candidate_op_profiles":
        logging.info(f"\t{key}: {value}")
    else:
        logging.info(f"\t{key}:")
        for i in value:
            logging.info(f"\t\t{i[0]}: {i[1]}")

device = torch.device(args.device)

# 计算scale列表
scale_list = []
for i in range(args.scale_num):
    scale_list.append(int(settings.data.in_length / args.scale_num))
logging.info(f"Scale list: {scale_list}")

# 数据处理 - 使用NPZ适配器
logging.info("Loading NPZ dataset...")
NPZdata = NPZDataProcessing(
    dataset=args.dataset,
    train_prop=settings.data.train_prop,
    valid_prop=settings.data.valid_prop,
    num_sensors=settings.data.num_sensors,
    in_length=settings.data.in_length,
    out_length=settings.data.out_length,
    in_channels=args.in_channels,
    batch_size_per_gpu=settings.data.batch_size,
)

scaler = NPZdata.scaler
dataloader = NPZdata.dataloader
adj_mx_gwn = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_gwn]
adj_mx = [torch.tensor(NPZdata.adj_mx_dcrnn).to(device), adj_mx_gwn, torch.tensor(NPZdata.adj_mx_01).to(device)]

# 设置mask支持的邻接矩阵
if args.dataset in ["PEMS-BAY", "METR-LA"]:
    mask_support_adj = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_gwn]
else:
    mask_support_adj = [torch.tensor(i).to(device) for i in NPZdata.adj_mx_01]

# mask_support_adj = [torch.tensor(NPZdata.adj_mx_01).to(device)]

logging.info("Dataset loaded successfully!")


def main(run_id):
    """主训练函数"""
    seed = run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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
        in_channels=args.in_channels,
        out_channels=settings.data.out_channels,
        hidden_channels=settings.model.hidden_channels,
        end_channels=settings.model.end_channels,
        layer_names=settings.model.layer_names,
        config=config,
        device=device,
    )

    # 初始化训练器
    trainer = NPZTrainer(model, settings, scaler, device)

    # 加载搜索阶段的模型（最优架构）
    save_folder = "./model_param/" + args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    search_model_path = os.path.join(save_folder, args.dataset + "_best_search_model_" + args.model_des + ".pt")
    train_model_path = os.path.join(
        save_folder, args.dataset + "_best_train_model_" + args.model_des + "_run_" + str(run_id) + ".pt"
    )

    # 如果指定了预训练模型，则加载它
    if args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        trainer.load(args.load_model)
    else:
        # 否则尝试加载搜索阶段的模型
        try:
            best_epoch = trainer.load(search_model_path)
            logging.info("load architecture [epoch %d] from %s [done]", best_epoch, search_model_path)
        except Exception as e:
            logging.info(f"load search model failed: {e}")
            logging.info("search first...")
            sys.exit(1)

    # 初始化wandb（如果启用）
    if args.use_wandb:
        wandb.init(
            project="AutoSTF_NPZ",
            config={
                "dataset": args.dataset,
                "settings": args.settings,
                "epochs": args.epochs,
                "model_des": args.model_des,
                "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "server_name": server_config.server_name,
                "GPU_type": server_config.GPU_type,
                **settings.data_dict,
                **settings.trainer_dict,
                **settings.model_dict,
            },
        )

    # 训练循环
    logging.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_maes = []
        train_mapes = []
        train_rmses = []

        # 获取训练数据
        train_loader = dataloader["train_loader"].get_iterator()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            # 训练权重（使用ONE_PATH_FIXED模式，固定架构只训练权重）
            loss, mae, mape, rmse = trainer.train_weight(x, y, Mode.ONE_PATH_FIXED)
            train_losses.append(loss)
            train_maes.append(mae)
            train_mapes.append(mape)
            train_rmses.append(rmse)

        # 验证阶段
        model.eval()
        val_losses = []
        val_maes = []
        val_mapes = []
        val_rmses = []

        val_loader = dataloader["valid_loader"].get_iterator()

        with torch.no_grad():
            for x, y in val_loader:
                x = torch.tensor(x, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.float32).to(device)

                loss, mae, mape, rmse = trainer.eval(x, y)
                val_losses.append(loss)
                val_maes.append(mae)
                val_mapes.append(mape)
                val_rmses.append(rmse)

        # 计算平均指标
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_train_mape = np.mean(train_mapes)
        avg_train_rmse = np.mean(train_rmses)

        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_val_mape = np.mean(val_mapes)
        avg_val_rmse = np.mean(val_rmses)

        # 更新学习率
        trainer.step_schedulers()

        # 日志记录
        logging.info(f"Epoch {epoch+1}/{args.epochs}:")
        logging.info(
            f"  Train - Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}, MAPE: {avg_train_mape:.4f}, RMSE: {avg_train_rmse:.4f}"
        )
        logging.info(
            f"  Val   - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, MAPE: {avg_val_mape:.4f}, RMSE: {avg_val_rmse:.4f}"
        )

        # wandb记录
        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "train_mae": avg_train_mae,
                    "train_mape": avg_train_mape,
                    "train_rmse": avg_train_rmse,
                    "val_loss": avg_val_loss,
                    "val_mae": avg_val_mae,
                    "val_mape": avg_val_mape,
                    "val_rmse": avg_val_rmse,
                }
            )

        # 早停检查和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            # 保存最佳模型
            if args.save_model:
                model_save_path = f"./saved_models/{args.model_des}_best.pth"
                os.makedirs("./saved_models", exist_ok=True)
                trainer.save(model_save_path, epoch, epoch)
                logging.info(f"Best model saved to {model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= settings.trainer.early_stop_steps:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # 保存最终训练模型（使用之前定义的train_model_path）
    logging.info(f"Saving final trained model to {train_model_path}")
    states = {
        "net": trainer.model.state_dict(),
        "weight_optimizer": trainer.weight_optimizer.state_dict(),
        "weight_scheduler": trainer.weight_scheduler.state_dict(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(obj=states, f=train_model_path)
    logging.info(f"Final trained model saved to {train_model_path}")

    # 测试阶段
    logging.info("Starting testing...")
    model.eval()
    test_losses = []
    test_maes = []
    test_mapes = []
    test_rmses = []

    test_loader = dataloader["test_loader"].get_iterator()

    with torch.no_grad():
        for x, y in test_loader:
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            loss, mae, mape, rmse = trainer.eval(x, y)
            test_losses.append(loss)
            test_maes.append(mae)
            test_mapes.append(mape)
            test_rmses.append(rmse)

    # 计算测试指标
    avg_test_loss = np.mean(test_losses)
    avg_test_mae = np.mean(test_maes)
    avg_test_mape = np.mean(test_mapes)
    avg_test_rmse = np.mean(test_rmses)

    logging.info("Testing completed!")
    logging.info(
        f"Test Results - Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}, MAPE: {avg_test_mape:.4f}, RMSE: {avg_test_rmse:.4f}"
    )

    # wandb记录测试结果
    if args.use_wandb:
        wandb.log(
            {
                "test_loss": avg_test_loss,
                "test_mae": avg_test_mae,
                "test_mape": avg_test_mape,
                "test_rmse": avg_test_rmse,
            }
        )
        wandb.finish()

    return avg_test_mae, avg_test_mape, avg_test_rmse


if __name__ == "__main__":
    # 运行多次实验
    all_maes = []
    all_mapes = []
    all_rmses = []

    for run in range(args.run_times):
        logging.info(f"Starting run {run+1}/{args.run_times}")
        mae, mape, rmse = main(run)
        all_maes.append(mae)
        all_mapes.append(mape)
        all_rmses.append(rmse)

    # 计算统计结果
    if args.run_times > 1:
        logging.info("Final Results Summary:")
        logging.info(f"MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
        logging.info(f"MAPE: {np.mean(all_mapes):.4f} ± {np.std(all_mapes):.4f}")
        logging.info(f"RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}")
    else:
        logging.info("Single run completed!")

    logging.info("All experiments finished!")

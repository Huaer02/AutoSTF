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
from src.model.mode import Mode, create_mode
from src import metrics_adapter

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

parser.add_argument(
    "--mode_name", type=str, default="TWO_PATHS", help="ONE_PATH_FIXED, ONE_PATH_RANDOM, " "TWO_PATHS, ALL_PATHS"
)
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to search")
parser.add_argument("--model_des", type=str, default="NPZ_AutoSTF", help="save model param")
parser.add_argument("--comments", type=str, default="", help="save model param")

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

logging.info("Dataset loaded successfully!")

# 创建模式
mode = create_mode(args.mode_name)


def main(runid):
    """主搜索函数"""
    seed = runid
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

    # 计算参数数量
    num_Params = sum([p.nelement() for p in model.parameters()])
    logging.info("Number of model parameters is %d", num_Params)

    # 初始化wandb（如果启用）
    if args.use_wandb:
        wandb.init(
            project="AutoSTF_NPZ_Search",
            config={
                "model_name": "AutoSTF",
                "tag": "search",
                "dataset": args.dataset,
                "settings": args.settings,
                "epochs": args.epochs,
                "model_des": args.model_des,
                "mode_name": args.mode_name,
                "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "server_name": server_config.server_name,
                "GPU_type": server_config.GPU_type,
                "weight_lr": settings.trainer.weight_lr,
                "weight_lr_decay_ratio": settings.trainer.weight_lr_decay_ratio,
                "weight_decay": settings.trainer.weight_decay,
                "weight_clip_gradient": settings.trainer.weight_clip_gradient,
                "arch_lr": settings.trainer.arch_lr,
                "arch_lr_decay_ratio": settings.trainer.arch_lr_decay_ratio,
                "arch_decay": settings.trainer.arch_decay,
                "arch_clip_gradient": settings.trainer.arch_clip_gradient,
                "batch_size": settings.data.batch_size,
                "early_stop": settings.trainer.early_stop,
                "early_stop_steps": settings.trainer.early_stop_steps,
                "train_prop": float(settings.data.train_prop),
                "valid_prop": float(settings.data.valid_prop),
                "num_sensors": settings.data.num_sensors,
                "in_channels": args.in_channels,
                "out_channels": settings.data.out_channels,
                "in_length": settings.data.in_length,
                "out_length": settings.data.out_length,
                "end_channels": settings.model.end_channels,
                "hidden_channels": settings.model.hidden_channels,
                "num_mlp_layers": args.num_mlp_layers,
                "IsUseLinear": settings.model.IsUseLinear,
                "num_linear_layers": settings.model.num_linear_layers,
                "scale_num": args.scale_num,
                "scale_list": scale_list,
                "num_att_layers": settings.model.num_att_layers,
                "num_hop": settings.model.num_hop,
                "layer_names": settings.model.layer_names,
                "num_temporal_search_node": settings.model.num_temporal_search_node,
                "num_spatial_search_node": settings.model.num_spatial_search_node,
                "temporal_operations": settings.model.temporal_operations,
                "spatial_operations": settings.model.spatial_operations,
            },
        )
        wandb.log({"num_params": num_Params})

    # 初始化训练器
    trainer = NPZTrainer(model, settings, scaler, device)

    # 加载预训练模型（如果指定）
    save_folder = "./model_param/" + args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder, args.dataset + "_best_search_model_" + args.model_des + ".pt")

    if args.load_model:
        logging.info(f"Loading pretrained model from {args.load_model}")
        trainer.load(args.load_model)
    else:
        try:
            best_epoch = trainer.load(model_path)
            logging.info("load architecture [epoch %d] from %s [done]", best_epoch, model_path)
        except:
            logging.info("load architecture [fail]")
            best_epoch = 0

    # 初始化训练变量
    his_valid_time = []
    his_train_time = []
    his_valid_loss = []
    min_valid_loss = 1000
    all_start_time = time.time()

    logging.info("Starting searching...")

    for epoch in range(best_epoch + 1, best_epoch + args.epochs + 1):
        epoch_start_time = time.time()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []

        # 获取搜索训练数据
        search_train_loader = dataloader["search_train_loader"].get_iterator()
        train_start_time = time.time()

        for batch_idx, (x, y) in enumerate(search_train_loader):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            # 训练权重
            metrics = trainer.train_weight(x, y, mode=mode)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])

        trainer.weight_scheduler.step()
        train_end_time = time.time()

        t_time = train_end_time - train_start_time
        his_train_time.append(t_time)

        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info("\n")
        logging.info(now_time)
        log_loss = (
            "%s, Epoch: %03d,\n"
            "Search Loss: %.4f, Search MAE: %.4f, Search MAPE: %.4f, Search RMSE: %.4f, Search Time: %.4f"
        )
        logging.info(
            log_loss, args.dataset, epoch, mean_train_loss, mean_train_mae, mean_train_mape, mean_train_rmse, t_time
        )

        # 验证阶段
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        search_valid_loader = dataloader["search_valid_loader"].get_iterator()
        valid_start_time = time.time()

        for batch_idx, (x, y) in enumerate(search_valid_loader):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)

            # 训练架构
            metrics = trainer.train_arch(x, y, mode=mode)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        trainer.arch_scheduler.step()

        valid_end_time = time.time()
        v_time = valid_end_time - valid_start_time
        his_valid_time.append(v_time)

        epoch_time = time.time() - epoch_start_time

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mae = np.mean(valid_mae)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)

        his_valid_loss.append(mean_valid_loss)

        log_loss = "Valid Loss: %.4f, Valid MAE: %.4f, Valid MAPE: %.4f, Valid RMSE: %.4f, Valid Time: %.4f"
        logging.info(log_loss, mean_valid_loss, mean_valid_mae, mean_valid_mape, mean_valid_rmse, v_time)
        logging.info("Epoch Search Time: %.4f", epoch_time)

        # 保存最佳模型
        if mean_valid_loss < min_valid_loss:
            best_epoch = epoch
            states = {
                "net": trainer.model.state_dict(),
                "arch_optimizer": trainer.arch_optimizer.state_dict(),
                "arch_scheduler": trainer.arch_scheduler.state_dict(),
                "weight_optimizer": trainer.weight_optimizer.state_dict(),
                "weight_scheduler": trainer.weight_scheduler.state_dict(),
                "best_epoch": best_epoch,
            }
            torch.save(obj=states, f=model_path)
            logging.info("\n[save] epoch %d, save parameters to %s", best_epoch, model_path)
            min_valid_loss = mean_valid_loss

        # 早停检查
        elif settings.trainer.early_stop and epoch - best_epoch > settings.trainer.early_stop_steps:
            logging.info("\n")
            logging.info("-" * 40)
            logging.info("Early Stopped, best search epoch: %d", best_epoch)
            logging.info("-" * 40)
            break

    all_end_time = time.time()

    logging.info("\n")
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    best_id = np.argmin(his_valid_loss)
    logging.info("Searching finished.")

    # 加载最佳模型
    best_epoch = trainer.load(model_path)
    logging.info(
        "The valid loss on best search model is %s, epoch:%d\n", str(round(his_valid_loss[best_id], 4)), best_epoch
    )

    logging.info("All Search Time: %.4f secs", all_end_time - all_start_time)
    logging.info("Average Search Time: %.4f secs/epoch", np.mean(his_train_time))
    logging.info("Average Inference Time: %.4f secs/epoch", np.mean(his_valid_time))

    logging.info("\n")
    logging.info("Best Search Model Loaded")

    # 在验证集上评估
    outputs = []
    true_valid_y = []
    valid_loader = dataloader["valid_loader"].get_iterator()

    for batch_idx, (x, y) in enumerate(valid_loader):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        with torch.no_grad():
            preds = trainer.model(x, mode=Mode.ONE_PATH_FIXED)

        outputs.append(preds.squeeze(dim=1))
        true_valid_y.append(y)

    valid_yhat = torch.cat(outputs, dim=0)
    true_valid_y = torch.cat(true_valid_y, dim=0)
    valid_pred = scaler.inverse_transform(valid_yhat)
    valid_pred = torch.clamp(valid_pred, min=scaler.min_value, max=scaler.max_value)
    valid_mae, valid_mape, valid_rmse = train_util.metric(valid_pred, true_valid_y)

    log = "12 Average Performance on Valid Data - Valid MAE: %.4f, Valid MAPE: %.4f, Valid RMSE: %.4f"
    logging.info(log, valid_mae, valid_mape, valid_rmse)

    # 在测试集上评估
    outputs = []
    true_test_y = []
    test_loader = dataloader["test_loader"].get_iterator()

    for batch_idx, (x, y) in enumerate(test_loader):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        with torch.no_grad():
            preds = trainer.model(x, mode=Mode.ONE_PATH_FIXED)

        outputs.append(preds.squeeze(dim=1))
        true_test_y.append(y)

    test_yhat = torch.cat(outputs, dim=0)
    true_test_y = torch.cat(true_test_y, dim=0)
    test_pred = scaler.inverse_transform(test_yhat)
    test_pred = torch.clamp(test_pred, min=scaler.min_value, max=scaler.max_value)
    test_mae, test_mape, test_rmse = train_util.metric(test_pred, true_test_y)

    log = "12 Average Performance on Test Data - Test MAE: %.4f, Test MAPE: %.4f, Test RMSE: %.4f \n"
    logging.info(log, test_mae, test_mape, test_rmse)

    # 单步测试
    # logging.info('Single steps test:')
    # mae = []
    # mape = []
    # rmse = []
    # for i in [2, 5, 8, 11]:
    #     pred_single_step = test_pred[:, :, i]
    #     real = true_test_y[:, :, i]
    #     metrics_single = train_util.metric(pred_single_step, real)
    #     log = 'horizon %d, Test MAE: %.4f, Test MAPE: %.4f, Test RMSE: %.4f'
    #     logging.info(log, i + 1, metrics_single[0], metrics_single[1], metrics_single[2])

    #     mae.append(metrics_single[0])
    #     mape.append(metrics_single[1])
    #     rmse.append(metrics_single[2])

    # # 平均步测试
    # logging.info('\nAverage steps test:')
    # mae_avg = []
    # mape_avg = []
    # rmse_avg = []
    # for i in [3, 6, 9, 12]:
    #     pred_avg_step = test_pred[:, :, :i]
    #     real = true_test_y[:, :, :i]
    #     metrics_avg = train_util.metric(pred_avg_step, real)
    #     log = 'average %d, Test MAE: %.4f, Test MAPE: %.4f, Test RMSE: %.4f'
    #     logging.info(log, i, metrics_avg[0], metrics_avg[1], metrics_avg[2])
    #     mae_avg.append(metrics_avg[0])
    #     mape_avg.append(metrics_avg[1])
    #     rmse_avg.append(metrics_avg[2])

    # 结束wandb
    if args.use_wandb:
        wandb.finish()

    # return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse, mae, mape, rmse, mae_avg, mape_avg, rmse_avg
    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse


if __name__ == "__main__":
    # valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse, mae, mape, rmse, mae_avg, mape_avg, rmse_avg = main(1)
    valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse = main(1)
    logging.info("Search Finish!\n")

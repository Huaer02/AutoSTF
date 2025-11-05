import torch
import numpy as np


def masked_mae_loss(y_pred, y_true, null_val=0.0):
    """
    适配版本的MAE损失函数，兼容两个系统的接口
    """
    if null_val is not None:
        mask = (y_true != null_val).float()
    else:
        mask = (y_true > 1e-4).float()

    mask /= mask.mean()  # 归一化权重
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # 处理NaN值
    loss[loss != loss] = 0
    return loss.mean()


def masked_mape_loss(y_pred, y_true, null_val=0.0):
    """
    适配版本的MAPE损失函数
    """
    if null_val is not None:
        mask = (y_true != null_val).float()
    else:
        mask = (y_true > 1e-4).float()

    mask /= mask.mean()
    # 避免除零错误
    y_true_safe = torch.where(torch.abs(y_true) < 1e-4, torch.ones_like(y_true), y_true)
    loss = torch.abs((y_pred - y_true) / y_true_safe)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true, null_val=0.0):
    """
    适配版本的RMSE损失函数
    """
    if null_val is not None:
        mask = (y_true != null_val).float()
    else:
        mask = (y_true > 1e-4).float()

    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def masked_mae(preds, labels, null_val=0.0):
    """
    兼容AutoSTF原始接口的MAE函数
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    """
    兼容AutoSTF原始接口的MAPE函数
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # 避免除零错误
    labels_safe = torch.where(torch.abs(labels) < 1e-4, torch.ones_like(labels), labels)
    loss = torch.abs(preds - labels) / labels_safe
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    """
    兼容AutoSTF原始接口的RMSE函数
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def metric(pred, real, null_val=0.0):
    """
    计算所有评估指标
    """
    mae = masked_mae(pred, real, null_val).item()
    mape = masked_mape(pred, real, null_val).item()
    rmse = masked_rmse(pred, real, null_val).item()
    return mae, mape, rmse


def evaluate_predictions(pred, truth, mask_value=None):
    """
    评估预测结果，返回多种指标
    """
    if mask_value is not None:
        mask = torch.gt(truth, mask_value)
        pred_masked = torch.masked_select(pred, mask)
        truth_masked = torch.masked_select(truth, mask)
    else:
        pred_masked = pred
        truth_masked = truth

    mae = torch.mean(torch.abs(truth_masked - pred_masked)).item()
    mape = torch.mean(torch.abs(torch.div((truth_masked - pred_masked), truth_masked))).item() * 100
    rmse = torch.sqrt(torch.mean(torch.square(pred_masked - truth_masked))).item()

    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "masked_MAE": mae if mask_value is not None else None,
        "masked_MAPE": mape if mask_value is not None else None,
        "masked_RMSE": rmse if mask_value is not None else None,
    }

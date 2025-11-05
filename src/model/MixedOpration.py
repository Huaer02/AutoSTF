import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.CandidateOpration import create_op
from src.model.mode import Mode


class TemporalLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag):
        super(TemporalLayerMixedOp, self).__init__()

        used_operations = None
        if tag == "temporal":
            used_operations = config.temporal_operations
        elif tag == "spatial":
            used_operations = config.spatial_operations
        else:
            print("search operations error")

        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, config, device)]
        # self._candidate_alphas = nn.Parameter(torch.normal(mean=torch.zeros(self._num_ops), std=1), requires_grad=True)
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True)

        # self.start_linear = linear(in_channels, in_channels//self._k)
        # self.end_linear = linear(in_channels//self._k, in_channels)

        self.set_mode(Mode.NONE)

    def set_mode(self, mode):
        self._mode = mode
        if mode == Mode.NONE:
            self._sample_idx = None

        elif mode == Mode.ONE_PATH_FIXED:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item()
            # 检查索引是否越界
            if op >= self._num_ops:
                raise IndexError(f"set_mode ONE_PATH_FIXED索引越界: op {op} >= num_ops {self._num_ops}")
            if op < 0:
                raise IndexError(f"set_mode ONE_PATH_FIXED索引越界: op {op} < 0")
            self._sample_idx = np.array([op], dtype=np.int32)

        elif mode == Mode.ONE_PATH_RANDOM:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            sample_result = torch.multinomial(probs, 1, replacement=True).cpu().numpy()
            # 检查索引是否越界
            if sample_result.max() >= self._num_ops:
                raise IndexError(
                    f"set_mode ONE_PATH_RANDOM索引越界: max {sample_result.max()} >= num_ops {self._num_ops}"
                )
            if sample_result.min() < 0:
                raise IndexError(f"set_mode ONE_PATH_RANDOM索引越界: min {sample_result.min()} < 0")
            self._sample_idx = sample_result

        elif mode == Mode.TWO_PATHS:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            sample_result = torch.multinomial(probs, 2, replacement=True).cpu().numpy()
            # 检查索引是否越界
            if sample_result.max() >= self._num_ops:
                raise IndexError(f"set_mode TWO_PATHS索引越界: max {sample_result.max()} >= num_ops {self._num_ops}")
            if sample_result.min() < 0:
                raise IndexError(f"set_mode TWO_PATHS索引越界: min {sample_result.min()} < 0")
            self._sample_idx = sample_result

        elif mode == Mode.ALL_PATHS:
            all_indices = np.arange(self._num_ops)
            # 检查索引是否越界（理论上不会，但为了完整性）
            if all_indices.max() >= self._num_ops:
                raise IndexError(f"set_mode ALL_PATHS索引越界: max {all_indices.max()} >= num_ops {self._num_ops}")
            if all_indices.min() < 0:
                raise IndexError(f"set_mode ALL_PATHS索引越界: min {all_indices.min()} < 0")
            self._sample_idx = all_indices

    def forward(self, inputs, mask):
        inputs = inputs[0]
        # inputs = self.start_linear(inputs)

        if isinstance(self._sample_idx, np.ndarray):
            device = self._candidate_alphas.device
            sample_idx_tensor = torch.from_numpy(self._sample_idx).to(device)

        else:
            sample_idx_tensor = self._sample_idx

        a = self._candidate_alphas[sample_idx_tensor]
        probs = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            # 添加对_candidate_ops索引的检查
            if idx >= len(self._candidate_ops):
                raise IndexError(f"_candidate_ops索引越界: idx {idx} >= len(_candidate_ops) {len(self._candidate_ops)}")
            if idx < 0:
                raise IndexError(f"_candidate_ops索引越界: idx {idx} < 0")
            output += probs[i] * self._candidate_ops[idx](inputs, mask=mask)
        # output = self.end_linear(output)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p


class SpatialLayerMixedOp(nn.Module):
    def __init__(self, node_embedding_1, node_embedding_2, adj_mx, config, device, tag):
        super(SpatialLayerMixedOp, self).__init__()

        self.scale_list = config.scale_list
        used_operations = config.spatial_operations

        self._num_ops = len(used_operations)
        self._candidate_ops = nn.ModuleList()
        for op_name in used_operations:
            self._candidate_ops += [create_op(op_name, node_embedding_1, node_embedding_2, adj_mx, config, device)]

    def forward(self, inputs, candidate_alphas, mask):
        inputs = inputs[0]
        # inputs = self.start_linear(inputs)

        probs = F.softmax(candidate_alphas, dim=0)
        sample_idx = torch.multinomial(probs, 2, replacement=True).cpu().numpy()

        # 检查sample_idx是否越界
        if sample_idx.max() >= self._num_ops:
            raise IndexError(f"SpatialLayerMixedOp索引越界: max index {sample_idx.max()} >= num_ops {self._num_ops}")
        if sample_idx.min() < 0:
            raise IndexError(f"SpatialLayerMixedOp索引越界: min index {sample_idx.min()} < 0")

        # 检查candidate_alphas索引是否越界
        if sample_idx.max() >= candidate_alphas.shape[0]:
            raise IndexError(
                f"candidate_alphas索引越界: max index {sample_idx.max()} >= candidate_alphas.shape[0] {candidate_alphas.shape[0]}"
            )

        a = candidate_alphas[sample_idx]
        p = F.softmax(a, dim=0)
        output = 0
        for i, idx in enumerate(sample_idx):
            # 添加对_candidate_ops索引的检查
            if idx >= len(self._candidate_ops):
                raise IndexError(
                    f"SpatialLayerMixedOp _candidate_ops索引越界: idx {idx} >= len(_candidate_ops) {len(self._candidate_ops)}"
                )
            if idx < 0:
                raise IndexError(f"SpatialLayerMixedOp _candidate_ops索引越界: idx {idx} < 0")
            output += p[i] * self._candidate_ops[idx](inputs, mask=mask)
        # output = self.end_linear(output)
        return output

    def weight_parameters(self):
        for i in range(self._num_ops):
            for p in self._candidate_ops[i].parameters():
                yield p

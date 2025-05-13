import torch
import numpy as np


def init_weights(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm3d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            torch.nn.init.constant_(m.bias, 0)


def count_trainable_parameters(model: torch.nn.Module) -> str:
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "trainable": f"{train_params:,}",
        "total": f"{total_params:,}",
    }


def compute_tensor_size_conv(size: tuple[int, int, int], model: torch.nn.Module):
    t_size = np.copy(size)
    for param in model.children():
        if hasattr(param, "weight") and hasattr(param, "kernel_size"):
            k, s, p = param.kernel_size, param.stride, param.padding
            for i in range(len(size)):
                t_size[i] = np.floor((t_size[i] + 2 * p[i] - k[i]) / s[i] + 1)
    return t_size

from __future__ import print_function, division

import torch
from torch import nn
import torch.nn.functional as F



def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer. 

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in ["relu", "leaky_relu", "elu", "gelu",
                          "swish", "efficient_swish", "none"], \
        "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]


def get_functional_act(activation: str = 'relu'):
    """Get the specified activation function. 

    Args:
        activation (str): one of ``'relu'``, ``'tanh'``, ``'elu'``, ``'sigmoid'``, 
            ``'softmax'`` and ``'none'``. Default: ``'sigmoid'``
    """
    assert activation in ["relu", "tanh", "elu", "sigmoid", "softmax", "none"], \
        "Get unknown activation_fn key {}".format(activation)
    activation_dict = {
        'relu': F.relu_,
        'tanh': torch.tanh,
        'elu': F.elu_,
        'sigmoid': torch.sigmoid,
        'softmax': lambda x: F.softmax(x, dim=1),
        'none': lambda x: x,
    }
    return activation_dict[activation]

# ----------------------
# Normalization Layers
# ----------------------


def get_norm_3d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 3D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    if norm == "gn": assert out_channels%8 == 0, "GN requires channels to separable into 8 groups"
    norm = {
        "bn": nn.BatchNorm3d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm3d,
        "gn": lambda channels: nn.GroupNorm(8, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_norm_2d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 2D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    norm = {
        "bn": nn.BatchNorm2d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm2d,
        "gn": lambda channels: nn.GroupNorm(16, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_norm_1d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 1D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    norm = {
        "bn": nn.BatchNorm1d,
        "sync_bn": nn.BatchNorm1d,
        "in": nn.InstanceNorm1d,
        "gn": lambda channels: nn.GroupNorm(16, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_num_params(model):
    num_param = sum([param.nelement() for param in model.parameters()])
    return num_param

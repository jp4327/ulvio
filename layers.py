import torch
import torch.nn as nn

from collections import OrderedDict


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act_func == "gelu":
        return nn.GELU()
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        use_norm=True,
        norm_func='LN',
        norm_chan_per_group=None,
        use_act=True,
        act_func="relu",
        dropout_rate=0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_norm = use_norm
        self.norm_func = norm_func
        self.use_act = use_act
        self.act_func = act_func
        self.dropout_rate = dropout_rate

        # default normal 3x3_Conv with bn and relu
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        """ modules """
        modules = {}
        # norm layer
        if self.use_norm:
            if norm_func == "BN":
                modules["norm"] = nn.BatchNorm2d(out_channels)
            if norm_func == "LN":
                modules["norm"] = nn.GroupNorm(1, out_channels)
            if norm_func == "GN":
                modules["norm"] = nn.GroupNorm(out_channels//norm_chan_per_group, out_channels)
        else:
            modules["norm"] = None
        # activation
        if use_act:
            modules["act"] = build_activation(
                self.act_func, self.use_norm
            )
        else:
            modules["act"] = None
        # dropout
        if self.dropout_rate > 0:
            modules["dropout"] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules["dropout"] = None
        # weight
        modules["weight"] = self.weight_op()

        # add modules
        for op in ["weight", "norm", "act"]:
            if modules[op] is None:
                continue
            elif op == "weight":
                # dropout before weight operation
                if modules["dropout"] is not None:
                    self.add_module("dropout", modules["dropout"])
                for key in modules["weight"]:
                    self.add_module(key, modules["weight"][key])
            else:
                self.add_module(op, modules[op])

    def weight_op(self):
        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        weight_dict = OrderedDict(
            {
                "conv": nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding,
                    dilation=self.dilation,
                    bias=self.bias,
                )
            }
        )
        return weight_dict

    def forward(self, x):
        # similar to nn.Sequential
        for module in self._modules.values():
            x = module(x)
        return x

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from op.upfirdn2d import *




def get_weight(module, shape, weight_var='weight', kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""
    return module.param(weight_var, kernel_init, shape)





def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k





def _shape(x, dim):
    return x.shape[dim]





def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))





def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))


def upsample_2d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     up=factor, pad=((p + 1) // 2 + factor - 1, p // 2))


def downsample_2d(x, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor
    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     down=factor, pad=((p + 1) // 2, p // 2))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




def conv_upsample_2d(x, w, k, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1

    
    assert len(w.shape) == 4
    convH = w.shape[2]
    convW = w.shape[3]
    inC = w.shape[1]
    outC = w.shape[0]

    assert convW == convH

    
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    stride = (factor, factor)

    
    stride = [1, 1, factor, factor]
    output_shape = ((_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW)
    output_padding = (output_shape[0] - (_shape(x, 2) - 1) * stride[0] - convH,
                      output_shape[1] - (_shape(x, 3) - 1) * stride[1] - convW)
    assert output_padding[0] >= 0 and output_padding[1] >= 0
    num_groups = _shape(x, 1) // inC

    
    w = torch.reshape(w, (num_groups, -1, inC, convH, convW))
    w = w[..., ::-1, ::-1].permute(0, 2, 1, 3, 4)
    w = torch.reshape(w, (num_groups * inC, -1, convH, convW))

    x = F.conv_transpose2d(x, w, stride=stride, output_padding=output_padding, padding=0)

    return upfirdn2d(x, torch.tensor(k, device=x.device),
                     pad=((p + 1) // 2 + factor - 1, p // 2 + 1))





def conv_downsample_2d(x, w, k=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    _outC, _inC, convH, convW = w.shape
    assert convW == convH
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    s = [factor, factor]
    x = upfirdn2d(x, torch.tensor(k, device=x.device),
                  pad=((p + 1) // 2, p // 2))
    return F.conv2d(x, w, stride=s, padding=0)


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, up=False, down=False, resample_kernel=(1, 3, 3, 1), use_bias=True,
                 kernel_init=None):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = conv_upsample_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x

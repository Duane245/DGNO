"""Galerkin neural operator components: global / local kernel-integral operators."""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out


class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, x, name='0'):

        B, C, H, W = x.shape
        bias = x

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias


class Win_SRNO(nn.Module):
    """
    Defocus-Aware SRNO (Local-only)
    Local kernel integral operator
    """
    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.scale = window * window

        self.qkv_proj = nn.Conv2d(dim, 3 * dim, 1)
        self.out_proj1 = nn.Conv2d(dim, dim, 1)
        self.out_proj2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.unfold = nn.Unfold(
            kernel_size=window,
            padding=window // 2,
            stride=1
        )

        self.kln = LayerNorm((self.heads, 1, self.head_dim))
        self.vln = LayerNorm((self.heads, 1, self.head_dim))

    def forward(self, x, name='0'):
        
        # ---- QKV ----
        B, C, H, W = x.shape
        bias = x

        # qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H*W, self.heads, 3*self.head_dim)
        # qkv = qkv.permute(0, 2, 1, 3)
        # q, k, v = qkv.chunk(3, dim=-1)

        qkv = self.qkv_proj(x).permute(0, 2, 3, 1)
        q, k, v = qkv.chunk(3, dim=-1)

        # print(k.shape)
        q = window_partition(q, self.window).reshape(-1, self.window * self.window, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = window_partition(k, self.window).reshape(-1, self.window * self.window, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = window_partition(v, self.window).reshape(-1, self.window * self.window, self.heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.kln(k)
        v = self.vln(v)
        
        v = torch.matmul(k.transpose(-2,-1), v) / (self.window*self.window)
        
        v = torch.matmul(q, v)
        # windows: (num_windows*B, window_size, window_size, C)
        v = v.reshape(-1, self.window, self.window, self.heads*self.head_dim)
        v = window_reverse(v, self.window, H, W)

        ret = v.permute(0, 3, 1, 2) + bias
        out = self.out_proj2(self.act(self.out_proj1(ret))) + bias

        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



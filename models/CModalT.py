import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from models.vit_utils import DropPath, trunc_normal_


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=96):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)

        self.out = nn.Linear(heads * dim_head, dim)

    def forward(self, x, context):
        b, n, _, h = *x.shape, self.heads
        m = context.shape[1]
        q = self.to_q(x).view(b, n, h, -1)
        k = self.to_k(context).view(b, m, h, -1)
        v = self.to_v(context).view(b, m, h, -1)

        dots = torch.einsum('bqhd,bkhd->bhqk', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
        out = out.reshape(b, n, -1)
        return self.out(out)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class CVAFM(nn.Module):
    """ Cross Modal Vision and Audio Fusion Module
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.crossattn = CrossAttention(dim, heads=num_heads)
        self.attn = Attention(dim, num_heads=num_heads)
        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_v:torch.Tensor, x_a:torch.Tensor):
        x_cls = x_v[:,0].unsqueeze(1)
        x_v = x_v[:,1:]
        x_vn = rearrange(x_v, 'b (t n) d -> t b n d', t=8)
        x_l = []
        for x_v in x_vn:
            x = self.drop_path(self.crossattn(self.norm1(x_v), self.norm1(x_a)))
            x = torch.mean(x, dim=1)
            # (b, d)
            x_l.append(x)
        x_l=torch.stack(x_l, dim=1)
        x = torch.cat([x_cls, x_l], dim=1)
        # x = self.drop_path(self.attn(self.norm3(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x[:,0]

class CrossTransformer(nn.Module):
    """ Cross Modal Transformer Encoder
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.crossattn = CrossAttention(dim, heads=num_heads)
        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_v:torch.Tensor, x_a:torch.Tensor):
        x = self.drop_path(self.crossattn(self.norm1(x_v), self.norm1(x_a)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x[:,0]

def classif_head(in_dim, out_dim, drop= 0.5):
    return Mlp(in_dim, int(in_dim/2), out_dim, drop=drop)
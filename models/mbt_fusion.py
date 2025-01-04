import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .vit_utils import DropPath

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AddPositionEmbs(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(AddPositionEmbs, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, emb_size))
        nn.init.normal_(self.position_embeddings, std=0.02)
    
    def forward(self, x):
        # x shape: (batch, len, emb)
        assert x.ndim == 3, "Input tensor must be 3-dimensional"
        seq_len = x.size(1)
        return x + self.position_embeddings[:, :seq_len, :]

class Encoder(nn.Module):
    def __init__(self, modality_fusion, fusion_layer = 4, num_layers=4, num_heads=8, dropout_rate=0.1, attention_dropout_rate=0.1,use_bottleneck=True, share_encoder=False):
        super().__init__()
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_bottleneck = use_bottleneck
        self.share_encoder = share_encoder
        # position embedding
        # self.pos_emb_layer = AddPositionEmbs(emb_size=768)
        # encoder
        self.encoder = torch.nn.ModuleDict()
        for modality in self.modality_fusion:
            self.encoder[modality] = nn.ModuleList([Block(dim=768, num_heads=self.num_heads) for i in range(self.num_layers)])
        self.ln = nn.LayerNorm(768)
  
    def forward(self, x, bottleneck=None, train=True):
        # Add positional embeddings
        # for modality in self.modality_fusion:
        #     x[modality] = self.pos_emb_layer(x[modality])    
        # Assuming modalities is a dictionary where keys are modality names and values are tensors
        for lyr in range(self.num_layers):
            # for modality in self.modality_fusion:
            #     x[modality] = self.encoder[modality][lyr](x[modality])
        # Bottleneck
            if self.use_bottleneck:
                bottle = []
                for modality in self.modality_fusion:
                    t_mod = x[modality].shape[1]
                    in_mod = torch.concat([x[modality], bottleneck], axis=1)
                    out_mod = self.encoder[modality][lyr](in_mod)
                    x[modality] = out_mod[:, :t_mod]
                    bottle.append(out_mod[:, t_mod:])
                bottleneck = torch.mean(torch.stack(bottle, axis=-1), axis=-1)
        output = torch.concat(list(x.values()), axis=1)
        encoded = self.ln(output)
        return encoded

# MBT(4, 2, 8, 768)
class MBT(nn.Module):
    def __init__(self, num_layers, num_heads, num_classes, hidden_size, cls_token = True ,dropout_rate =0.1, attention_dropout_rate=0.1, classifier= 'token' ,fusion_layer: int = 4, use_bottleneck=True, n_bottlenecks=4, share_encoder=False):
        super(MBT, self).__init__()
        self.modality_fusion: Tuple[str] = ('visual','audio')
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.fusion_layer = fusion_layer
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.share_encoder = share_encoder
        self.classifier = classifier
        self.representation_size = None
        # cls token
        if cls_token:
            self.cls_token = {}
            for modality in self.modality_fusion:
                self.cls_token[modality] = nn.Parameter(torch.zeros(1, 1, hidden_size)).cuda()
                # setattr(self, f'{modality}_cls', nn.Parameter(torch.zeros(1, 1, hidden_size)))
                # nn.init.normal_(self.cls_token[modality], mean=0, std=0.02)
            # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # nn.init.normal_(self.cls_token, mean=0, std=0.02)
        else:
            self.cls_token = None
        # bottleneck
        self.bottleneck = None
        if self.use_bottleneck:
            self.bottleneck = nn.Parameter(torch.empty(1, n_bottlenecks, 768), requires_grad=True)
        nn.init.normal_(self.bottleneck, mean=0, std=0.02)
        #fusion
        self.encoder = Encoder(
        modality_fusion=self.modality_fusion,
        fusion_layer=self.fusion_layer,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        use_bottleneck=self.use_bottleneck,
        share_encoder=self.share_encoder)
        # representation size
        # self.representation_size = 768
        # self.pre_logits_fc = nn.Sequential(
        #     nn.Linear(hidden_size, self.representation_size),
        #     nn.Tanh()
        #     )
        # classifier
        self.output_projection_fc = nn.Linear(hidden_size, num_classes)
        nn.init.zeros_(self.output_projection_fc.weight)
        nn.init.zeros_(self.output_projection_fc.bias)

    def forward(self, v, a, train=True):
        n, v_t, dim = v.size()  # Assuming x is your input tensor and its first dimension is the batch size
        # Tile the bottleneck to match the batch size
        x = {}
        for modality in self.modality_fusion:
            x[modality] = v if modality == 'visual' else a
        bottleneck_expanded = self.bottleneck.expand(n, -1, -1)  # Using expand instead of repeat for memory efficiency
        # cat cls
        if self.cls_token is not None:
            for modality in self.modality_fusion:
                cls_tokens = self.cls_token[modality].expand(n, -1, -1)
                x[modality] = torch.cat((cls_tokens, x[modality]), dim=1)
        #fusion
        x = self.encoder(x, bottleneck_expanded, train=train)

        # Obtaining the CLS tokens for each modality.
        if self.classifier in ['token']:
            # Obtaining the CLS tokens for each modality.
            x_out = {}
            counter = 0
            for modality in self.modality_fusion:
                x_out[modality] = x[:, counter]
                counter += (v_t+1)
        elif self.classifier in ['gap', 'gmp', 'gsp']:
            if self.classifier == 'gap':
                x_out = x.mean(dim=1)
            elif self.classifier == 'gmp':
                x_out = x.max(dim=1).values
            elif self.classifier == 'gsp':
                x_out = x.sum(dim=1)        
        # representation_size
        if self.representation_size is not None:
            if isinstance(x_out, dict):
                for modality in x_out:
                    x_out[modality] = self.pre_logits_fc(x_out[modality])
        # output_projection_fc
        # x_pool = torch.tensor(0.0).to(v.device)
        for modality in self.modality_fusion:
            x_out[modality] = self.output_projection_fc(x_out[modality])
        x_pool = x_out[self.modality_fusion[0]]
        return x_pool


if __name__ == '__main__':
    model = MBT(2, 2, 8, 768)
    v = torch.randn(2, 50, 768)
    a = torch.randn(2, 1, 768)
    out = model(v, a)
    print(out.shape)


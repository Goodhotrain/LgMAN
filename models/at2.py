import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from einops import rearrange
from models.vit_utils import DropPath, trunc_normal_
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
torchaudio.set_audio_backend("soundfile")
from transforms.audio import TSNAudio

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
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

class PatchEmbed(nn.Module):
    """ Audio to Patch Embedding
    """
    def __init__(self, ipt_size=256, patch_size=20, in_chans=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(ipt_size, embed_dim, kernel_size=patch_size, stride=in_chans)

    def forward(self, x: torch.Tensor):
        # size: B * n_segment, 257, 20
        B_n_s, s_l, d = x.shape
        x = self.proj(x)
        x = x.squeeze(2)
        return x

def preprocess_audio(audio_path='test.wav'):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except:
        audio_path = '/media/Harddisk/Datasets/Micro_Video/MeiTu/audio/941004182.mp3'
        waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    return waveform, sample_rate



class AudioTransformer(nn.Module):
    """ Audio Transformer
    """
    def __init__(self, audio_n_segments=8,segment_len=256, embedded_len=20,  num_classes=6, embed_dim=256, depth=1,
                 num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, dropout=0.):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, audio_n_segments+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_embed = PatchEmbed(ipt_size=segment_len, patch_size=embedded_len, in_chans=1, embed_dim=embed_dim)
        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        self.transform = transforms.MFCC(
        sample_rate=44100,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 20, "center": False})
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.processor = Wav2Vec2Processor.from_pretrained("/media/Harddisk/ghy/models/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("/media/Harddisk/ghy/models/wav2vec2-base-960h")

        self.audio_transform = TSNAudio()

    def forward(self, x, MFCC=False):
        waveform1, sample_rate = preprocess_audio(x)
        with torch.no_grad():
            if sample_rate != self.processor.feature_extractor.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.processor.feature_extractor.sampling_rate)
                waveform = resampler(waveform1)
            chunk_size = 16000 * 120
            if len(waveform) > chunk_size:
                waveform = waveform[:chunk_size]
            if len(waveform) < 5*16000:
                waveform = torch.cat([waveform, torch.zeros(16000*8-len(waveform))])
            inputs = self.processor(waveform, return_tensors="pt", sampling_rate=self.processor.feature_extractor.sampling_rate)['input_values'].cuda()
            features = self.model(inputs).last_hidden_state
            features.squeeze_(0)
        # temporal
        if MFCC:
            waveform1 = waveform1.cuda()
            mfcc = self.transform(waveform1)
            feature = mfcc.transpose(0, 1)
            timeseries_length = 4096
            k = timeseries_length // feature.shape[0] + 1
            feature = torch.tile(feature, dims=(k, 1))

            total_frames = feature.shape[0]
            audio_indices = self.audio_transform(total_frames)
            audios = []
            for ai in audio_indices:
                audios.append(feature[ai,:])
            audios = torch.stack(audios, dim = 0)
            x = self.forward_features(audios)
            return features[0,:], features,  x[1:,:]
            # x = self.head(x[:, 0])
        return features[0, :]
    
    def forward_features(self, x):
        # input size: B, n_segment, 256, 20 -> 
        # B, n_segment, seg_len, F_d = x.size()
        # x = rearrange(x, 'b n s f -> (b n) s f')
        # size: B * n_segment, 256, 20
        
        x = self.pos_drop(x)
        # size: B * n_segment, 256, 20
        x = self.patch_embed(x)
        
        x = rearrange(x, '(b s) w -> b s w', s=8)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # size: B , n_segment, 256
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).squeeze(0) # B, n_segment+1, 256

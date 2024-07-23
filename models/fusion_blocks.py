import torch
from torch import nn
from timm.models.vision_transformer import DropPath, Mlp


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        (B, N1, C), N2 = x1.shape, x2.shape[1]
        q = self.q(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        return x1, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#########################################################################################
#
#  m1 m2 ... mF
#  |/ \|/   \|
#  m1 m2 ... mF   --- vi \forall i -- aj \forall j ---
#
#########################################################################################
class CrossAttention_LocalAVTokens(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = int(dim * dim_ratio)
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xmm, xv, xa):
        (bs, nmm, _), nv, na = xmm.shape, xv.shape[1], xa.shape[1]

        x_src = torch.cat((xv, xa), dim=1)
        q = self.q(xmm).reshape(bs, nmm, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv(x_src).reshape(bs, nv+na, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xmm = (attn @ v).transpose(1, 2).reshape(bs, nmm, self.dim)
        xmm = self.proj(xmm)
        xmm = self.proj_drop(xmm)
        return xmm, attn


class FusionBlock_LocalAVTokens(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_mm = norm_layer(dim)
        self.norm1_aud = norm_layer(dim)
        self.norm1_img = norm_layer(dim)
        self.attn = CrossAttention_LocalAVTokens(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xmm, xa, xv, return_attention=False):
        xmm, xv, xa = self.norm1_mm(xmm), self.norm1_img(xv), self.norm1_aud(xa)
        res_fusion, attn = self.attn(xmm, xv, xa)
        xmm = xmm + self.drop_path(res_fusion)
        if return_attention:
            return attn

        res_fusion = self.mlp(self.norm2(xmm))
        xmm = xmm + self.drop_path(res_fusion)
        return xmm


#########################################################################################
#
#  m1 m2 ... mF
#  |/ \|/   \|
#  m1 m2 ... mF    ---  cat(vi,aj) \forall i,j  ---
#
#########################################################################################
class CrossAttention_DenseAVInteractions(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = int(dim * dim_ratio)

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(dim * 2, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xmm, xa, xv):
        (bs, nmm, _), nv, na = xmm.shape, xv.shape[1], xa.shape[1]

        xva = torch.cat((
            xv.unsqueeze(2).repeat(1, 1, na, 1),
            xa.unsqueeze(1).repeat(1, nv, 1, 1),
        ), dim=3).flatten(1, 2)

        q = self.q(xmm).reshape(bs, nmm, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = self.kv(xva).reshape(bs, nv*na, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xmm = (attn @ v).transpose(1, 2).reshape(bs, nmm, self.dim)
        xmm = self.proj(xmm)
        xmm = self.proj_drop(xmm)
        return xmm, attn


class FusionBlock_DenseAVInteractions(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_mm = norm_layer(dim)
        self.norm1_aud = norm_layer(dim)
        self.norm1_img = norm_layer(dim)
        self.attn = CrossAttention_DenseAVInteractions(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xmm, xv, xa, return_attention=False):
        xmm, xv, xa = self.norm1_mm(xmm), self.norm1_img(xv), self.norm1_aud(xa)
        res_fusion, attn = self.attn(xmm, xv, xa)
        xmm = xmm + self.drop_path(res_fusion)
        if return_attention:
            return attn

        res_fusion = self.mlp(self.norm2(xmm))
        xmm = xmm + self.drop_path(res_fusion)
        return xmm


class CrossAttention_FactorizedAVInteractions(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_ratio=1., fusion_tkns=(8, 4, 4)):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = int(dim * dim_ratio)
        self.fusion_tkns = fusion_tkns

        self.attn_v = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.attn_a = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim * 2, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim * 2, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xmm, xv, xa):
        bs = xmm.shape[0]
        nmm, nv, na = self.fusion_tkns

        # 16     8      8
        xmm2, xmm_v, xmm_a = xmm.split((nmm, nv, na), dim=1)
        xmm_v, _ = self.attn_v(xmm_v, xv)     # Linearly with #V
        xmm_a, _ = self.attn_a(xmm_a, xa)     # Linearly with #A

        # All VA pairs
        xva = torch.cat((
            xmm_v.unsqueeze(2).repeat(1, 1, na, 1),
            xmm_a.unsqueeze(1).repeat(1, nv, 1, 1),
        ), dim=3).flatten(1, 2)

        q = self.q(xmm2).reshape(bs, nmm, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(xva).reshape(bs, nv*na, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(xva).reshape(bs, nv*na, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        xmm2 = (attn @ v).transpose(1, 2).flatten(2)
        xmm2 = self.proj(xmm2)
        xmm2 = self.proj_drop(xmm2)

        xmm = torch.cat((xmm2, xmm_v, xmm_a), dim=1)
        return xmm, attn


class FusionBlock_FactorizedAVInteractions(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False, fusion_tkns=(8, 4, 4),
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_mm = norm_layer(dim)
        self.norm1_aud = norm_layer(dim)
        self.norm1_img = norm_layer(dim)
        self.attn = CrossAttention_FactorizedAVInteractions(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio, fusion_tkns=fusion_tkns)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, xmm, xv, xa, return_attention=False):
        xmm, xv, xa = self.norm1_mm(xmm), self.norm1_img(xv), self.norm1_aud(xa)
        res_fusion, attn = self.attn(xmm, xv, xa)
        xmm = xmm + self.drop_path(res_fusion)
        if return_attention:
            return attn

        res_fusion = self.mlp(self.norm2(xmm))
        xmm = xmm + self.drop_path(res_fusion)
        return xmm

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


PRETRAINED_WEIGHTS = {
    'vit_base_audiomae_as2m': ('assets/models/vitbase_audiomae_as2m.pth', ''),
    'vit_base_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', ''),
    'vit_large_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth', ''),
    'vit_huge_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth', ''),
}

class ViT(nn.Module):
    """ VisionTransformer backbone
    """
    def __init__(self,
                 input_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, use_cls_token=False,
                 drop_path=0., attn_drop=0., drop=0.):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(input_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path, attn_drop=attn_drop, proj_drop=drop)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_checkpoint(self, ckpt_fn, prefix='', skip_keys_prefix=('decoder', 'mask_token')):
        try:
            ckpt = torch.load(ckpt_fn, map_location="cpu")
        except Exception:
            ckpt = torch.hub.load_state_dict_from_url(url=ckpt_fn, map_location="cpu")

        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        ckpt = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith(skip_keys_prefix)}

        if self.cls_token is None and 'cls_token' in ckpt:
            del ckpt['cls_token']
        ckpt['pos_embed'] = self.state_dict()['pos_embed']
        self.load_state_dict(ckpt, strict=True)

    def params_layer_ids(self):
        params_layer_ids = []
        params_layer_ids.extend([(p, 0) for p in self.patch_embed.parameters()])
        params_layer_ids.extend([(self.cls_token, 0)])
        for i, blk in enumerate(self.blocks):
            params_layer_ids.extend([(p, i+1) for p in blk.parameters()])
        params_layer_ids.extend([(p, len(self.blocks)+1) for p in self.norm.parameters()])
        return params_layer_ids

    def prepare_patch_tokens(self, x, ids_keep=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking
        if ids_keep is not None:
            x = x.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        # append cls token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        return x

    def forward(self, x, ids_keep=None):
        # prepare patches
        x = self.prepare_patch_tokens(x, ids_keep=ids_keep)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


def vit_small_patch16(pretrained=False, **kwargs):
    assert pretrained == False
    model = ViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_base_patch16(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained is not None and pretrained != '':
        assert pretrained in {'vit_base_mae_in1k', 'vit_base_audiomae_as2m'}
        url, prefix = PRETRAINED_WEIGHTS[pretrained]
        model.load_checkpoint(url, prefix=prefix)

    return model


def vit_large_patch16(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained is not None:
        assert pretrained in {'vit_large_mae_in1k'}
        url, prefix = PRETRAINED_WEIGHTS[pretrained]
        model.load_checkpoint(url, prefix=prefix)

    return model


def vit_huge_patch14(pretrained=None, **kwargs):
    model = ViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    if pretrained is not None:
        assert pretrained in {'vit_huge_mae_in1k'}
        url, prefix = PRETRAINED_WEIGHTS[pretrained]
        model.load_checkpoint(url, prefix=prefix)

    return model


# set recommended archs
vit_small = vit_small_patch16
vit_base = vit_base_patch16
vit_large = vit_large_patch16
vit_huge = vit_huge_patch14
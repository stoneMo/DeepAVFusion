import torch
from torch import nn
from models import fusion_blocks, vits


class DeepAVFusion(nn.Module):
    def __init__(
            self,
            image_arch='vit_base', image_pretrained=True, image_size=(224, 224),
            audio_arch='vit_base', audio_pretrained=True, audio_size=(128, 192),
            fusion_arch='factorized_mmi',
            fusion_layers='all',
            num_fusion_tkns=(4, 8, 4),
            fusion_mlp_ratio=1.0, fusion_attn_ratio=0.25, fusion_num_heads=12,
            drop_path=0., attn_drop=0., drop=0.,
    ):
        super(DeepAVFusion, self).__init__()

        # Audio and visual encoders
        self.image = vits.__dict__[image_arch](pretrained=image_pretrained, input_size=image_size, in_chans=3, use_cls_token=False, drop_path=drop_path, attn_drop=attn_drop, drop=drop)
        self.audio = vits.__dict__[audio_arch](pretrained=audio_pretrained, input_size=audio_size, in_chans=1, use_cls_token=False, drop_path=drop_path, attn_drop=attn_drop, drop=drop)
        self.embed_dim = self.image.embed_dim
        self.fusion_arch = fusion_arch

        # NOTE: multi-modal fusion blocks and tokens
        self.num_fusion = num_fusion_tkns
        self.fusion_tokens = nn.Parameter(torch.zeros(1, sum(num_fusion_tkns), self.embed_dim))

        FusionBlock = None
        if fusion_arch == 'token':
            FusionBlock = fusion_blocks.FusionBlock_LocalAVTokens
        elif fusion_arch == 'dense_mmi':
            FusionBlock = fusion_blocks.FusionBlock_DenseAVInteractions
        elif fusion_arch == 'factorized_mmi':
            from functools import partial
            FusionBlock = partial(fusion_blocks.FusionBlock_FactorizedAVInteractions, fusion_tkns=num_fusion_tkns)

        max_depth = max(len(self.image.blocks), len(self.audio.blocks))
        if fusion_layers == 'all':
            fusion_layers = set(range(max_depth))
        elif fusion_layers == 'none':
            fusion_layers = set([])
        elif isinstance(fusion_layers, int):
            fusion_layers = {fusion_layers}
        else:
            fusion_layers = set([int(l) for l in fusion_layers.split('-')])
        self.fusion_blocks = nn.ModuleList([
            None if i not in fusion_layers or FusionBlock is None else FusionBlock(
                dim=self.embed_dim, num_heads=fusion_num_heads, attn_ratio=fusion_attn_ratio, mlp_ratio=fusion_mlp_ratio, qkv_bias=True,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=nn.LayerNorm)
            for i in range(max_depth)])
        self.fusion_norm = nn.LayerNorm(self.embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.fusion_tokens, std=.02)
        self.fusion_blocks.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def params_layer_ids(self):
        params_layer_ids = []
        params_layer_ids.extend(self.image.params_layer_ids())
        params_layer_ids.extend(self.audio.params_layer_ids())
        params_layer_ids.extend([(self.fusion_tokens, 0)])
        for i, blk in enumerate(self.fusion_blocks):
            if blk is not None:
                params_layer_ids.extend([(p, i+1) for p in blk.parameters()])
        params_layer_ids.extend([(p, len(self.fusion_blocks)+1) for p in self.fusion_norm.parameters()])
        return params_layer_ids

    def load_checkpoint(self, ckpt_fn, prefix):
        ckpt = torch.load(ckpt_fn, map_location='cpu')
        ckpt = ckpt['state_dict']
        ckpt = {k[len(prefix):]: ckpt[k] for k in ckpt if k.startswith(prefix)}
        self.load_state_dict(ckpt, strict=True)
        print(f"Loaded pre-trained checkpoint: {ckpt_fn}")

    def forward(self, image, audio, image_ids_keep=None, audio_ids_keep=None, return_embs=False):
        B = image.shape[0]

        # embed patches
        x_image = self.image.prepare_patch_tokens(image, image_ids_keep)
        x_audio = self.audio.prepare_patch_tokens(audio, audio_ids_keep)

        # apply blocks
        embs = []
        x_fusion = self.fusion_tokens.expand(B, -1, -1)
        nI, nA, nF = x_image.shape[1], x_audio.shape[1], self.fusion_tokens.shape[1]
        for blk_image, blk_audio, blk_fusion in zip(self.image.blocks, self.audio.blocks, self.fusion_blocks):
            if blk_fusion is None:
                x_image = blk_image(x_image)
                x_audio = blk_audio(x_audio)
            else:
                _, _x_image = blk_image(torch.cat((x_fusion, x_image), dim=1)).split((nF, nI), dim=1)
                _, _x_audio = blk_audio(torch.cat((x_fusion, x_audio), dim=1)).split((nF, nA), dim=1)
                x_fusion = blk_fusion(x_fusion, x_image, x_audio)
                x_image, x_audio = _x_image, _x_audio
            if return_embs:
                embs.append((x_image, x_audio, x_fusion))

        x_image = self.image.norm(x_image)
        x_audio = self.audio.norm(x_audio)
        x_fusion = self.fusion_norm(x_fusion)

        if not return_embs:
            return x_image, x_audio, x_fusion
        else:
            return x_image, x_audio, x_fusion, embs

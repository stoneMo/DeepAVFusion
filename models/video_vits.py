from functools import partial
from typing import Tuple, Optional, Union
from einops import rearrange, repeat

import torch
import torch.nn as nn

from util.pos_embed import PatchEmbed3D, get_3d_sincos_pos_embed
from timm.models.vision_transformer import Attention, Mlp, DropPath

PRETRAINED_WEIGHTS = {
    'vit_base_audiomae_as2m': ('assets/models/vitbase_audiomae_as2m.pth', ''),
    'vit_base_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth', ''),
    'vit_large_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth', ''),
    'vit_huge_mae_in1k': ('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth', ''),
}

class Block(nn.Module):
	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
				 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='joint_all'):
		super().__init__()
		assert attention_type in ['joint_all', 'divided_space_time']
		self.attention_type = attention_type

		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

		## Temporal Attention Parameters
		if self.attention_type == 'divided_space_time':
			self.temporal_norm1 = norm_layer(dim)
			self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
			self.temporal_fc = nn.Linear(dim, dim)

		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

	def forward(self, x, return_attention=False, has_cls_token=False, T=1):
		'''
			@x: (b, l, c)
		'''
		assert not return_attention
		b = x.shape[0]
		if self.attention_type == 'joint_all':
			x = x + self.drop_path(self.attn(self.norm1(x)))

		elif self.attention_type == "divided_space_time":
			# https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py#L89
			if has_cls_token:
				init_cls_token = x[:, 0:1, :]
				x = x[:, 1:]

				# Temporal
				xt = rearrange(x, "b (t s) c -> (b s) t c", t=T)
				res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
				res_temporal = rearrange(res_temporal, '(b s) t c -> b (t s) c', b=b)
				res_temporal = self.temporal_fc(res_temporal)
				x = x + res_temporal

				# Spatial
				cls_token = init_cls_token.repeat(1, T, 1)
				cls_token = rearrange(cls_token, 'b t c -> (b t) 1 c')
				xs = rearrange(x, 'b (t s) c -> (b t) s c', t=T)
				xs = torch.cat((cls_token, xs), 1)
				res_spatial = self.drop_path(self.attn(self.norm1(xs)))
				# Taking care of CLS token
				cls_token = res_spatial[:, 0, :]
				cls_token = rearrange(cls_token, '(b t) c -> b t c', t=T)
				cls_token = torch.mean(cls_token, 1, keepdim=True)  ## averaging for every frame
				res_spatial = res_spatial[:, 1:, :]
				res_spatial = rearrange(res_spatial, '(b t) s c -> b (t s) c', t=T)

				x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res_spatial), 1)

			else:
				# Temporal
				xt = rearrange(x, "b (t s) c -> (b s) t c", t=T)
				res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
				res_temporal = rearrange(res_temporal, '(b s) t c -> b (t s) c', b=b)
				res_temporal = self.temporal_fc(res_temporal)
				x = x + res_temporal

				# Spatial
				xs = rearrange(x, 'b (t s) c -> (b t) s c', t=T)
				res_spatial = self.drop_path(self.attn(self.norm1(xs)))
				res_spatial = rearrange(res_spatial, '(b t) s c -> b (t s) c', t=T)
				x = x + res_spatial

		else:
			raise Exception()

		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class VideoViTEncoder(nn.Module):
	def __init__(
			self,
			input_size: Tuple[int] = (16, 224, 224),
			patch_size: Tuple[int] = (2, 16, 16),
			stride: Optional[Union[Tuple, int]] = None,
			in_chans: int = 3,
			embed_dim: int = 1024,
			depth: int = 24,
			num_heads: int = 16,
			mlp_ratio: float = 4.,
			norm_layer: str = "layer_norm",
			norm_eps: float = 1e-6,
			use_cls_token=False,
			pos_trainable=False,
			drop_path: float = 0.,
			attn_drop: float = 0.,
			drop: float = 0.,
			attention_type: str = 'joint_all'
	):
		super().__init__()

		self.embed_dim = embed_dim
		self.input_size = input_size
		self.patch_size = patch_size
		if stride is None: stride = patch_size
		self.stride = stride
		self.use_cls_token = use_cls_token
		self.attention_type = attention_type

		if norm_layer == "layer_norm":
			norm_layer = partial(nn.LayerNorm, eps=norm_eps)
		else:
			raise Exception()

		# --------------------------------------------------------------------------
		# MAE encoder specifics
		self.patch_embed = PatchEmbed3D(
			input_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=stride
		)
		num_patches = self.patch_embed.num_patches

		if use_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
			# fixed sin-cos embedding
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)
		else:
			self.cls_token = None
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=pos_trainable)

		self.encoder_depth = depth
		self.blocks = nn.ModuleList([
			Block(
				embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
				drop_path=drop_path, attn_drop=attn_drop, drop=drop, attention_type=attention_type
			)
			for _ in range(depth)])
		self.norm = norm_layer(embed_dim)

		# --------------------------------------------------------------------------
		self.initialize_weights()

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
		if self.patch_embed.proj.weight.ndim > ckpt['patch_embed.proj.weight'].ndim:
			ckpt['patch_embed.proj.weight'] = ckpt['patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1, self.patch_size[0], 1, 1)
		self.load_state_dict(ckpt, strict=True)

	def params_layer_ids(self):
		params_layer_ids = []
		params_layer_ids.extend([(p, 0) for p in self.patch_embed.parameters()])
		params_layer_ids.extend([(self.cls_token, 0)])
		for i, blk in enumerate(self.blocks):
			params_layer_ids.extend([(p, i + 1) for p in blk.parameters()])
		params_layer_ids.extend([(p, len(self.blocks) + 1) for p in self.norm.parameters()])
		return params_layer_ids

	def initialize_weights(self):
		# initialization
		# initialize (and freeze) pos_embed by sin-cos embedding
		pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1],
											self.patch_embed.patch_thw,
											cls_token=self.use_cls_token)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		# initialize patch_embed like nn.Linear (instead of nn.Conv2d)
		w = self.patch_embed.proj.weight.data
		torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

		if self.use_cls_token:
			# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
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

	def prepare_patch_tokens(self, x, ids_keep=None):
		# embed patches
		x = self.patch_embed(x)

		if self.use_cls_token:
			# add pos embed w/o cls token
			x = x + self.pos_embed[:, 1:, :]
			# masking
			if ids_keep is not None:
				x = x.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

			# append cls token
			cls_token = self.cls_token + self.pos_embed[:, :1, :]
			cls_tokens = cls_token.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)
		else:
			# masking
			if ids_keep is not None:
				x = x.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
			x = x + self.pos_embed

		return x

	def forward(self, x):
		# prepare patches
		x = self.prepare_patch_tokens(x)
		T = self.patch_embed.patch_thw[0]
		# apply Transformer blocks
		for n, blk in enumerate(self.blocks):
			x = blk(x, has_cls_token=self.use_cls_token, T=T)
		x = self.norm(x)
		return x


# set recommended archs
def timesformer_small(pretrained=None, **kwargs):
	assert pretrained is None
	model = VideoViTEncoder(attention_type='divided_space_time',
							patch_size=(2, 16, 16), embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
	return model


def video_vit_small(pretrained="pretrained/videomae_vit-s_k400.pth", **kwargs):
	model = VideoViTEncoder(
		patch_size=(2, 16, 16), embed_dim=384, depth=12, num_heads=6,
		mlp_ratio=4, **kwargs
	)

	if pretrained:
		print(f"Loading audio vit from {pretrained}")
		checkpoint = torch.load(pretrained, map_location='cpu')['state_dict']
		# rename keys
		checkpoint_ = {}
		for key in list(checkpoint.keys()):
			if key.startswith("module.base_encoder.") and not key.startswith("module.base_encoder.head"):
				key_ = key[20:]
				checkpoint_[key_] = checkpoint[key]
		checkpoint = checkpoint_
		# reshape patch_embed.proj.weight
		checkpoint['patch_embed.proj.weight'] = repeat(checkpoint['patch_embed.proj.weight'],
													   "b c h w -> b c 2 h w").contiguous()

		state_dict = model.state_dict()

		for k in list(checkpoint.keys()):
			if k not in state_dict or (checkpoint[k].shape != state_dict[k].shape):
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint[k]
		# load pre-trained model
		model.load_state_dict(checkpoint, strict=False)

	return model


def timesformer_base(pretrained=None, **kwargs):
	assert pretrained is None
	model = VideoViTEncoder(attention_type='divided_space_time',
							patch_size=(2, 16, 16), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs)
	return model


def video_vit_base(pretrained=None, **kwargs):
	model = VideoViTEncoder(
		patch_size=(2, 16, 16), embed_dim=768, depth=12, num_heads=12,
		mlp_ratio=4, **kwargs)

	if pretrained is not None and pretrained != '':
		assert pretrained in {'vit_base_mae_in1k'}
		url, prefix = PRETRAINED_WEIGHTS[pretrained]
		model.load_checkpoint(url, prefix=prefix)

	return model


def timesformer_large(pretrained=None, **kwargs):
	assert pretrained is None
	model = VideoViTEncoder(attention_type='divided_space_time',
							patch_size=(2, 16, 16), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs)
	return model


def video_vit_large(pretrained="pretrained/mae_pretrain_vit_large.pth", **kwargs):
	model = VideoViTEncoder(
		patch_size=(2, 16, 16), embed_dim=1024, depth=24, num_heads=16,
		mlp_ratio=4, **kwargs)

	if pretrained:
		print(f"Loading video vit from {pretrained}")
		checkpoint = torch.load(pretrained, map_location='cpu')['model']
		state_dict = model.state_dict()

		for k in list(checkpoint.keys()):
			if k not in state_dict or (checkpoint[k].shape != state_dict[k].shape):
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint[k]
		# load pre-trained model
		model.load_state_dict(checkpoint, strict=False)

	return model


def timesformer_huge(pretrained=None, **kwargs):
	assert pretrained is None
	model = VideoViTEncoder(attention_type='divided_space_time',
							patch_size=(2, 14, 14), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
	return model


def video_vit_huge(pretrained="pretrained/mae_pretrain_vit_huge.pth", **kwargs):
	model = VideoViTEncoder(
		patch_size=(2, 14, 14), embed_dim=1280, depth=32, num_heads=16,
		mlp_ratio=4, **kwargs)

	if pretrained:
		print(f"Loading video vit from {pretrained}")
		checkpoint = torch.load(pretrained, map_location='cpu')['model']
		state_dict = model.state_dict()

		for k in list(checkpoint.keys()):
			if k not in state_dict or (checkpoint[k].shape != state_dict[k].shape):
				print(f"Removing key {k} from pretrained checkpoint")
				del checkpoint[k]
		# load pre-trained model
		model.load_state_dict(checkpoint, strict=False)

	return model


if __name__ == '__main__':
	import time
	import torch.amp
	model = video_vit_base(pretrained='', input_size=(24, 224, 224)).cuda()
	# model = video_vit_base(pretrained='', attention_type='divided_space_time').cuda()
	for b in range(10):
		bs = 2**b
		x = torch.randn(bs, 3, 24, 224, 224).cuda()
		ts = time.time()
		with torch.amp.autocast(device_type='cuda'):
			v = model(x)
		v.sum().backward()
		gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
		print(f"BS={bs}\t GPU mem: {gpu_mem:.1f} Gb\t Fwd Bwd: {time.time()-ts:.2f} sec")
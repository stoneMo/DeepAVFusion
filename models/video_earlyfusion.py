from typing import List, Tuple, Optional, Union

import torch
from torch import nn
from models import video_vits, vits
from models.fusion_blocks import FusionBlock_FactorizedAVInteractions


class VideoEarlyFusion(nn.Module):
	def __init__(
			self,
			video_arch: str = 'video_vit_base',
			video_pretrained: str = '',
			video_size: Tuple[int] = (24, 224, 224),
			audio_arch: str = 'audio_vit_base',
			audio_pretrained: str = '',
			audio_size: Tuple[int] = (128, 298),
			# fusion setting
			fusion_layers: str = 'all',
			num_fusion_tkns: Tuple[int] = (8, 16, 16),
			fusion_mlp_ratio: float = 1.,
			fusion_attn_ratio: float = .25,
			fusion_num_heads: int = 12,
			drop_path: float = 0.,
			attn_drop: float = 0.,
			drop: float = 0.,
	):
		super().__init__()

		# Audio and visual encoders
		self.video = video_vits.__dict__[video_arch](pretrained=video_pretrained, input_size=video_size, in_chans=3, use_cls_token=False, drop_path=drop_path, attn_drop=attn_drop, drop=drop)
		self.audio = vits.__dict__[audio_arch](pretrained=audio_pretrained, input_size=audio_size, in_chans=1, use_cls_token=False, drop_path=drop_path, attn_drop=attn_drop, drop=drop)
		self.embed_dim = self.video.embed_dim

		# NOTE: multi-modal fusion blocks and tokens
		self.num_fusion = num_fusion_tkns
		self.fusion_tokens = nn.Parameter(torch.zeros(1, sum(num_fusion_tkns), self.embed_dim))

		max_depth = max(len(self.video.blocks), len(self.audio.blocks))
		if fusion_layers == 'all':
			fusion_layers = set(range(max_depth))
		elif fusion_layers == 'none':
			fusion_layers = set([])
		elif isinstance(fusion_layers, int):
			fusion_layers = {fusion_layers}
		else:
			fusion_layers = set([int(l) for l in fusion_layers.split('-')])
		self.fusion_blocks = nn.ModuleList([
			None if i not in fusion_layers else FusionBlock_FactorizedAVInteractions(
				dim=self.embed_dim, fusion_tkns=num_fusion_tkns, num_heads=fusion_num_heads,
				attn_ratio=fusion_attn_ratio, mlp_ratio=fusion_mlp_ratio, qkv_bias=True,
				drop=drop, attn_drop=attn_drop, drop_path=drop_path)
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
		params_layer_ids.extend(self.video.params_layer_ids())
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
		# Adapt image model checkpoint for video
		ckpt = {k.replace('image.', 'video.'): ckpt[k] for k in ckpt}
		ckpt['video.pos_embed'] = self.video.state_dict()['pos_embed']
		if self.video.patch_embed.proj.weight.ndim > ckpt['video.patch_embed.proj.weight'].ndim:
			ckpt['video.patch_embed.proj.weight'] = ckpt['video.patch_embed.proj.weight'].unsqueeze(2).repeat(1, 1, self.video.patch_size[0], 1, 1)
		self.load_state_dict(ckpt, strict=True)
		print(f"Loaded pre-trained checkpoint: {ckpt_fn}")
	
	def forward(self, video, audio, video_ids_keep=None, audio_ids_keep=None, return_embs=False):
		'''
			@video: (b c t h w)
			@audio: (b c n t)
		'''
		B = video.shape[0]
		
		# embed patches
		x_video = self.video.prepare_patch_tokens(video, video_ids_keep)  # (b, t*h*w, c)
		x_audio = self.audio.prepare_patch_tokens(audio, audio_ids_keep)  # (b, n*t, c)

		# apply blocks
		embs = []
		x_fusion = self.fusion_tokens.expand(B, -1, -1)
		nV, nA, nF = x_video.shape[1], x_audio.shape[1], self.fusion_tokens.shape[1]
		for blk_video, blk_audio, blk_fusion in zip(self.video.blocks, self.audio.blocks, self.fusion_blocks):
			if blk_fusion is None:
				x_video = blk_video(x_video)
				x_audio = blk_audio(x_audio)
			else:
				_, _x_video = blk_video(torch.cat((x_fusion, x_video), dim=1)).split((nF, nV), dim=1)
				_, _x_audio = blk_audio(torch.cat((x_fusion, x_audio), dim=1)).split((nF, nA), dim=1)

				x_fusion = blk_fusion(x_fusion, x_video, x_audio)
				x_video, x_audio = _x_video, _x_audio
			
			if return_embs:
				embs.append((x_video, x_audio, x_fusion))
		
		x_video = self.video.norm(x_video)
		x_audio = self.audio.norm(x_audio)
		x_fusion = self.fusion_norm(x_fusion)

		if not return_embs:
			return x_video, x_audio, x_fusion
		else:
			return x_video, x_audio, x_fusion, embs


# set recommended archs
def video_efav_small(video_pretrained='', audio_pretrained='', **kwargs):
	assert video_pretrained == ''
	assert audio_pretrained == ''
	model = VideoEarlyFusion(
		video_arch='video_vit_small', video_pretrained=video_pretrained,
		audio_arch='vit_small', audio_pretrained=audio_pretrained,
		fusion_layers='all', num_fusion_tkns=(8, 4, 4), fusion_num_heads=6, **kwargs)
	return model


def video_efav_base(video_pretrained='', audio_pretrained='', **kwargs):
	assert video_pretrained == ''
	assert audio_pretrained == ''
	model = VideoEarlyFusion(
		video_arch='video_vit_base', video_pretrained=video_pretrained,
		audio_arch='vit_base', audio_pretrained=audio_pretrained,
		fusion_layers='all', num_fusion_tkns=(16, 8, 8), fusion_num_heads=12, **kwargs)
	return model


def video_efav_large(video_pretrained='', audio_pretrained='', **kwargs):
	assert video_pretrained == ''
	assert audio_pretrained == ''
	model = VideoEarlyFusion(
		video_arch='video_vit_large', video_pretrained=video_pretrained,
		audio_arch='vit_large', audio_pretrained=audio_pretrained,
		fusion_layers='all', num_fusion_tkns=(32, 12, 12), fusion_num_heads=16, **kwargs)
	return model


def video_efav_huge(video_pretrained='', audio_pretrained='', **kwargs):
	assert video_pretrained == ''
	assert audio_pretrained == ''
	model = VideoEarlyFusion(
		video_arch='video_vit_huge', video_pretrained=video_pretrained,
		audio_arch='vit_huge', audio_pretrained=audio_pretrained,
		fusion_layers='all', num_fusion_tkns=(64, 16, 16), fusion_num_heads=16, **kwargs)
	return model


if __name__ == '__main__':
	import time
	import torch.amp
	model = video_efav_base(video_pretrained='', audio_pretrained='', video_size=(8, 224, 224), audio_size=(128, 192)).cuda()
	for b in range(10):
		bs = 2**b
		xv = torch.randn(bs, 3, 8, 224, 224).cuda()
		xa = torch.randn(bs, 1, 128, 192).cuda()
		ts = time.time()
		with torch.amp.autocast(device_type='cuda'):
			vv, va, vf = model(xv, xa)
		(vv.sum()+va.sum()+vf.sum()).backward()
		gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
		print(f"BS={bs}\t GPU mem: {gpu_mem:.1f} Gb\t Fwd Bwd: {time.time()-ts:.2f} sec")
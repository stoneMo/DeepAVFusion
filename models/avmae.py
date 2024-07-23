import torch
from torch import nn

from util.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block
from models.swin import SwinTransformerBlock


class AVMAE(nn.Module):
    def __init__(
            self, encoder, encoder_dim,
            image_decoder_arch='plain', image_decoder_depth=8, image_mask_ratio=0.75, image_norm_loss=False,
            audio_decoder_arch='plain', audio_decoder_depth=8, audio_mask_ratio=0.8, audio_norm_loss=False,
            decoder_dim=512, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm
    ):
        super(AVMAE, self).__init__()

        self.image_mask_ratio = image_mask_ratio
        self.image_norm_loss = image_norm_loss
        self.audio_mask_ratio = audio_mask_ratio
        self.audio_norm_loss = audio_norm_loss
        self.decoder_dim = decoder_dim

        # -------------------------------------------------------------------------- #
        # Audio visual encoder
        self.encoder = encoder
        self.image_gs, self.audio_gs = encoder.image.patch_embed.grid_size, encoder.audio.patch_embed.grid_size
        self.image_ps, self.audio_ps = encoder.image.patch_embed.patch_size, encoder.audio.patch_embed.patch_size

        # -------------------------------------------------------------------------- #
        # Audio decoder
        self.audio_decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.audio_decoder_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.audio_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.audio_gs[0]*self.audio_gs[1], decoder_dim))  # fixed sin-cos embedding

        self.audio_decoder_arch = audio_decoder_arch
        if self.audio_decoder_arch == 'swin':
            self.audio_decoder_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=decoder_dim,
                    input_resolution=self.audio_gs,
                    window_size=4,
                    shift_size=(index % 2)*2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                )
                for index in range(audio_decoder_depth)])
        else:
            self.audio_decoder_blocks = nn.ModuleList([
                Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(audio_decoder_depth)])

        self.audio_decoder_norm = norm_layer(decoder_dim)
        self.audio_decoder_pred = nn.Linear(decoder_dim, self.audio_ps[0]*self.audio_ps[1], bias=True) # decoder to patch

        # -------------------------------------------------------------------------- #
        # Image decoder
        self.image_decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.image_decoder_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.image_decoder_pos_embed = nn.Parameter(torch.zeros(1, self.image_gs[0]*self.image_gs[1], decoder_dim))  # fixed sin-cos embedding

        self.image_decoder_arch = image_decoder_arch
        if self.image_decoder_arch == 'swin':
            self.image_decoder_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=decoder_dim,
                    input_resolution=self.image_gs,
                    window_size=4,
                    shift_size=(index % 2)*2,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                )
                for index in range(image_decoder_depth)])
        else:
            self.image_decoder_blocks = nn.ModuleList([
                Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(image_decoder_depth)])

        self.image_decoder_norm = norm_layer(decoder_dim)
        self.image_decoder_pred = nn.Linear(decoder_dim, self.image_ps[0]*self.image_ps[1]*3, bias=True) # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):

        # initialize (and freeze) pos_embed by sin-cos embedding
        pes = get_2d_sincos_pos_embed(self.decoder_dim, self.image_gs, cls_token=False)
        self.image_decoder_pos_embed.data.copy_(torch.from_numpy(pes).float().unsqueeze(0))
        pes = get_2d_sincos_pos_embed(self.decoder_dim, self.audio_gs, cls_token=False)
        self.audio_decoder_pos_embed.data.copy_(torch.from_numpy(pes).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.image_decoder_mask_token, std=.02)
        torch.nn.init.normal_(self.audio_decoder_mask_token, std=.02)

        # Initialize decoder (avoid init encoder to prevent overriding pretrained weights)
        for n, m in self.named_modules():
            if not n.startswith('encoder'):
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, N, L, mask_ratio, device):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """

        # sort noise for each sample
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep first subset
        len_keep = int(L * (1 - mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def forward_encoder(self, image, audio):
        return self.encoder(image, audio)

    def forward_decoder(self, x, x_fusion, ids_restore, modality='image'):
        bs, nFus, nMask = x.shape[0], x_fusion.shape[1], ids_restore.shape[1]-x.shape[1]
        embed = self.__getattr__(f'{modality}_decoder_embed')
        mask_token = self.__getattr__(f'{modality}_decoder_mask_token')
        pes = self.__getattr__(f'{modality}_decoder_pos_embed')
        arch = self.__getattribute__(f'{modality}_decoder_arch')
        blocks = self.__getattr__(f'{modality}_decoder_blocks')
        norm = self.__getattr__(f'{modality}_decoder_norm')
        pred = self.__getattr__(f'{modality}_decoder_pred')

        # embed tokens
        x, x_fusion = embed(x), embed(x_fusion)

        # append mask tokens to sequence and unshuffle
        x = torch.cat([x, mask_token.repeat(bs, nMask, 1)], dim=1)
        x = x.gather(dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + pes

        # apply Transformer blocks
        if arch == 'plain':
            x = torch.cat([x_fusion, x], dim=1)
            for blk in blocks:
                x = blk(x)
            x = x[:, nFus:, :]

        elif arch == 'swin':
            for blk in blocks:
                x, x_fusion = blk(x, x_fusion)

        # apply predictor
        x = pred(norm(x))
        return x

    @staticmethod
    def forward_loss(target, pred, mask, norm_pix_loss=True):
        """
        target: [N, L, p*p*3]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    @staticmethod
    def patchify(x, patch_size):
        """
        x: (N, C, H, W)
        patch_size = (H/p)*(W/p)
        """
        bs, c = x.shape[:2]
        pH, pW = patch_size
        gH, gW = x.shape[2] // pH, x.shape[3] // pW

        x = x.reshape(shape=(bs, c, gH, pH, gW, pW))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(bs, gH * gW, pH*pW * c))

        return x

    def forward(self, image, audio):
        B, device = image.shape[0], image.device

        # embed patches
        image_ids_keep, image_mask, image_ids_restore = self.random_masking(B, self.image_gs[0]*self.image_gs[1], self.image_mask_ratio, device=device)
        audio_ids_keep, audio_mask, audio_ids_restore = self.random_masking(B, self.audio_gs[0]*self.audio_gs[1], self.audio_mask_ratio, device=device)

        # Encoder image and audio
        x_image, x_audio, x_fusion = self.encoder(image, audio, image_ids_keep=image_ids_keep, audio_ids_keep=audio_ids_keep)

        # Decode image
        target_image = self.patchify(image, self.image_ps)
        pred_image = self.forward_decoder(x_image, x_fusion, image_ids_restore, modality='image')
        loss_image = self.forward_loss(target_image, pred_image, image_mask, norm_pix_loss=self.image_norm_loss)

        # Decode audio
        target_audio = self.patchify(audio, self.audio_ps)
        pred_audio = self.forward_decoder(x_audio, x_fusion, audio_ids_restore, modality='audio')
        loss_audio = self.forward_loss(target_audio, pred_audio, audio_mask, norm_pix_loss=self.audio_norm_loss)

        return loss_image, loss_audio, pred_image, pred_audio

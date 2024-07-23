import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, in2_channels=0, factor=2, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + in2_channels, out_channels, in_channels // factor)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // factor, kernel_size=factor, stride=factor)
            self.conv = DoubleConv(in_channels // factor + in2_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            return self.conv(torch.cat([x1, x2], dim=1))
        else:
            return self.conv(x1)


class AVSrcSepUNet(nn.Module):
    def __init__(self, embed_dim, bilinear=False):
        super().__init__()

        self.xv_norm = nn.LayerNorm(embed_dim)
        self.xa_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(5)])

        self.cond5 = nn.Linear(embed_dim, embed_dim)
        self.cond4 = nn.Linear(embed_dim, embed_dim // 2)
        self.cond3 = nn.Linear(embed_dim, embed_dim // 4)
        self.cond2 = nn.Linear(embed_dim, embed_dim // 8)
        self.cond1 = nn.Linear(embed_dim, embed_dim // 16)

        self.top = DoubleConv(embed_dim*2, embed_dim)

        self.lat4 = Up(embed_dim, embed_dim // 2, factor=2, bilinear=bilinear)
        self.lat3 = Up(embed_dim, embed_dim // 4, factor=4, bilinear=bilinear)
        self.lat2 = Up(embed_dim, embed_dim // 8, factor=8, bilinear=bilinear)
        self.lat1 = Up(embed_dim, embed_dim // 16, factor=16, bilinear=bilinear)

        self.up4 = Up(embed_dim // 1, embed_dim // 2, in2_channels=embed_dim // 1, bilinear=bilinear)
        self.up3 = Up(embed_dim // 2, embed_dim // 4, in2_channels=embed_dim // 2, bilinear=bilinear)
        self.up2 = Up(embed_dim // 4, embed_dim // 8, in2_channels=embed_dim // 4, bilinear=bilinear)
        self.up1 = Up(embed_dim // 8, embed_dim // 16, in2_channels=embed_dim // 8, bilinear=bilinear)

        self.pred = nn.Conv2d(embed_dim // 16, 1, kernel_size=(3, 3), padding=(1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, xa_embs, xv, audio_gs=(8, 12)):
        bs = xa_embs[0].shape[0]
        enc_idx = np.linspace(0, len(xa_embs)-1, 5, endpoint=True)[::-1].astype(int)        # [11 8 5 2 0]
        xa1, xa2, xa3, xa4, xa5 = [self.xa_norm[i](xa_embs[e]).view(bs, *audio_gs, -1).permute(0, 3, 1, 2)
                                   for i, e in enumerate(enc_idx)]
        xv = self.xv_norm(xv).mean(1)

        xv5 = self.cond5(xv)[:, :, None, None].repeat(1, 1, *audio_gs)
        x = self.top(torch.cat((xa5, xv5), dim=1))

        xv4 = self.cond4(xv)[:, :, None, None].repeat(1, 1, audio_gs[0]*2, audio_gs[1]*2)
        lat4 = torch.cat((self.lat4(xa4), xv4), dim=1)
        x = self.up4(x, lat4)

        xv3 = self.cond3(xv)[:, :, None, None].repeat(1, 1, audio_gs[0]*4, audio_gs[1]*4)
        lat3 = torch.cat((self.lat3(xa3), xv3), dim=1)
        x = self.up3(x, lat3)

        xv2 = self.cond2(xv)[:, :, None, None].repeat(1, 1, audio_gs[0]*8, audio_gs[1]*8)
        lat2 = torch.cat((self.lat2(xa2), xv2), dim=1)
        x = self.up2(x, lat2)

        xv1 = self.cond1(xv)[:, :, None, None].repeat(1, 1, audio_gs[0]*16, audio_gs[1]*16)
        lat1 = torch.cat((self.lat1(xa1), xv1), dim=1)
        x = self.up1(x, lat1)

        logits = self.pred(x)
        return logits


class AVSrcSep(nn.Module):
    def __init__(self, encoder, log_freq=True, weighted_loss=True, binary_mask=True):
        super().__init__()
        self.log_freq = log_freq
        self.weighted_loss = weighted_loss
        self.binary_mask = binary_mask

        self.encoder = encoder
        self.avss_decoder = AVSrcSepUNet(embed_dim=self.encoder.embed_dim)

    def params_layer_ids(self):
        params_layer_ids = []
        params_layer_ids.extend(self.encoder.params_layer_ids())
        params_layer_ids.extend([(p, len(self.encoder.image.blocks)+1) for p in self.avss_decoder.parameters()])
        return params_layer_ids

    def loss_mask_prediction(self, pred_mask, log_spec_mix, log_spec):
        spec = torch.pow(10., log_spec)
        spec_mix = torch.pow(10., log_spec_mix)

        # Calculate loss weighting coefficient: magnitude of input mixture
        if self.weighted_loss:
            weight = torch.log1p(spec_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(spec_mix)

        # Compute ground truth masks
        if self.binary_mask:
            gt_masks = (spec > spec_mix).float()
        else:
            gt_masks = spec / (spec + spec_mix + 1e-5)
            gt_masks.clamp_(0., 1.)

        loss_avss = F.binary_cross_entropy_with_logits(pred_mask, gt_masks, weight)
        return loss_avss, gt_masks

    def forward(self, image, audio_mix, audio_gt=None):
        # Encode audio and visuals
        _, _, _, all_embs = self.encoder(image, audio_mix, return_embs=True)
        xv = all_embs[-1][0]
        xa_embs = [x[1] for x in all_embs]

        # Prediction head
        audio_gs = self.encoder.audio.patch_embed.grid_size
        logits_mask = self.avss_decoder(xa_embs, xv, audio_gs)

        # source separation loss
        loss = gt_masks = None
        if audio_gt is not None:
            loss, gt_masks = self.loss_mask_prediction(logits_mask, audio_mix, audio_gt)
            loss = loss + torch.stack([p.sum()*0. for p in self.parameters()]).sum()

        return loss, logits_mask, gt_masks

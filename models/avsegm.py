import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from models.avsrcsep import Up, DoubleConv


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class AVSegmSimple(nn.Module):
    def __init__(self, encoder, num_classes=71):
        super(AVSegmSimple, self).__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        scales = [1, 2, 4, 8]
        embed_dim = self.encoder.embed_dim
        layer_dims = [max(128, embed_dim // scale) for scale in scales]
        self.normv = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(len(scales))])
        self.proja = nn.ModuleList([nn.Linear(embed_dim, layer_dims[d]) for d in range(len(scales))])
        self.norma = nn.ModuleList([nn.LayerNorm(layer_dims[d]) for d in range(len(scales))])
        self.top = DoubleConv(embed_dim*2, embed_dim)
        self.lat = nn.ModuleList([Up(embed_dim, layer_dims[d], factor=scales[d], bilinear=False) for d in range(1, len(scales))])
        self.up = nn.ModuleList([Up(layer_dims[d], layer_dims[d+1], in2_channels=layer_dims[d+1]*2, bilinear=False)
                                 for d in range(len(scales)-1)])

        self.predictor = nn.Sequential(
            nn.Conv2d(layer_dims[-1], 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        # Initialize decoder weights
        for n, m in self.named_modules():
            if not n.startswith('encoder'):
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def params_layer_ids(self):
        params_layer_ids = []
        params_layer_ids.extend(self.encoder.params_layer_ids())
        params_layer_ids.extend([(p, len(self.encoder.image.blocks)+1)
                                 for n, p in self.named_parameters() if not n.startswith('encoder')])
        return params_layer_ids

    def forward(self, image, audio, gt_segm=None):
        # Forward encoder
        _, _, _, all_embs = self.encoder(image, audio, return_embs=True)
        image_gs = self.encoder.image.patch_embed.grid_size
        xv_list = [all_embs[d][0] for d in np.linspace(0, len(all_embs)-1, len(self.norma), endpoint=True).astype(int)]
        xa_norm_list = [norm(proj(all_embs[-1][1])).mean(dim=1)
                        for norm, proj in zip(self.norma, self.proja)]
        xv_norm_list = [norm(xv).view(image.shape[0], *image_gs, -1).permute(0, 3, 1, 2)
                        for norm, xv in zip(self.normv, xv_list)]
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        xa_top = xa_norm_list[0][:, :, None, None].repeat(1, 1, *image_gs)
        x = self.top(torch.cat((xv_norm_list[0], xa_top), dim=1))
        for i, (xv, xa) in enumerate(zip(xv_norm_list[1:], xa_norm_list[1:])):
            xv = self.lat[i](xv)
            xa = xa[:, :, None, None].repeat(1, 1, *xv.shape[2:])
            x = self.up[i](x, torch.cat((xv, xa), dim=1))

        logits = self.predictor(x)   # BF x C x 224 x 224

        loss = None
        if gt_segm is not None:
            if self.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits[:, 0], gt_segm)
            else:
                loss = F.cross_entropy(logits, gt_segm)
            loss = loss + torch.stack([p.sum()*0. for p in self.parameters()]).sum()
        return loss, logits
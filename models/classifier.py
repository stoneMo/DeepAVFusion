import torch
from torch import nn

class AVClassifier(nn.Module):
    def __init__(self, encoder, num_classes, freeze_encoder=False, input_norm=False):
        super(AVClassifier, self).__init__()
        self.encoder = encoder

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.input_norm = input_norm
        if self.input_norm:
            self.image_norm = nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6)
            self.audio_norm = nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6)
            self.fusion_norm = nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6)

        self.image_head = nn.Linear(self.encoder.embed_dim, num_classes)
        self.audio_head = nn.Linear(self.encoder.embed_dim, num_classes)
        self.fusion_head = nn.Linear(self.encoder.embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.image_head.weight)
        nn.init.zeros_(self.image_head.bias)
        nn.init.xavier_uniform_(self.audio_head.weight)
        nn.init.zeros_(self.audio_head.bias)
        nn.init.xavier_uniform_(self.fusion_head.weight)
        nn.init.zeros_(self.fusion_head.bias)

    def params_layer_ids(self):
        params_layer_ids = []
        params_layer_ids.extend(self.encoder.params_layer_ids())
        params_layer_ids.extend([(p, len(self.encoder.audio.blocks)+1) for p in self.image_head.parameters()])
        params_layer_ids.extend([(p, len(self.encoder.audio.blocks)+1) for p in self.audio_head.parameters()])
        params_layer_ids.extend([(p, len(self.encoder.audio.blocks)+1) for p in self.fusion_head.parameters()])
        return params_layer_ids

    def forward(self, image, audio):
        if self.freeze_encoder:
            with torch.no_grad():
                x_image, x_audio, x_fusion = self.encoder(image, audio)
        else:
            x_image, x_audio, x_fusion = self.encoder(image, audio)

        x_image, x_audio, x_fusion = x_image.mean(dim=1), x_audio.mean(dim=1), x_fusion.mean(dim=1)
        if self.input_norm:
            x_image = self.image_norm(x_image)
            x_audio = self.audio_norm(x_audio)
            x_fusion = self.fusion_norm(x_fusion)

        pred_image = self.image_head(x_image)
        pred_audio = self.audio_head(x_audio)
        pred_fusion = self.fusion_head(x_fusion)

        return pred_image, pred_audio, pred_fusion

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.train(False)


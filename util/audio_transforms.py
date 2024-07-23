import torch
import random
from torchaudio.transforms import *
import torchaudio.functional as F
from torchvision.transforms import Compose


class RandomVol(torch.nn.Module):
    def __init__(self, gain=(-6, 6)):
        super(RandomVol, self).__init__()
        self.gain = gain

    def forward(self, waveform):
        gain = random.uniform(self.gain[0], self.gain[1])
        waveform = F.gain(waveform, gain)
        waveform = torch.clamp(waveform, -1, 1)
        return waveform

class Pad(torch.nn.Module):
    def __init__(self, dur, rate):
        super(Pad, self).__init__()
        self.samples = int(dur * rate)

    def forward(self, waveform):
        while waveform.shape[-1] < self.samples:
            waveform = torch.cat((waveform, torch.flip(waveform, dims=(1,))), dim=1)
        return waveform[:, :self.samples]

class Log(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(Log, self).__init__()
        self.eps = eps

    def forward(self, spec):
        return torch.log10(spec + self.eps)
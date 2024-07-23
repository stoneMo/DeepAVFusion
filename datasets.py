import glob
import json
import math
import os
import csv
import random
from collections import defaultdict
import numpy as np

import torch
import av
import avreader
from torch.utils.data import Dataset

from PIL import Image
import torchaudio
from torchaudio.transforms import Resample


def load_image(fn, format='RGB'):
    img = Image.open(fn)
    if format is not None:
        return img.convert(format)
    return img


class FolderVideoDataset(Dataset):
    def __init__(self, path, samples, audio_dur=3., audio_rate=8000, audio_mixture=1,
                 visual_transform=None, audio_transform=None, class_labels=None,
                 temporal_jitter=True, dense=False, oversample=None, return_semantics=False):
        super().__init__()
        assert audio_mixture == 1
        self.path = path
        self.samples = samples
        self.class_labels = class_labels

        self.audio_dur = audio_dur
        self.audio_rate = audio_rate
        self.temporal_jitter = temporal_jitter

        self.visual_transform = visual_transform
        self.audio_transform = audio_transform
        self.oversample = oversample if oversample is not None else 1
        self.dense = dense
        self.return_semantics = return_semantics

    def read_data(self, file_id, frame_no, audio_start_time, class_labels):
        # Read frame
        frame_fn = f"{self.path}/{file_id}/frames/{frame_no}.jpg"
        segm_fn = f"{self.path}/{file_id}/labels_semantic/{frame_no}.png"
        frame = load_image(frame_fn, format='RGB')
        segm_map = load_image(segm_fn, format='L')
        if self.visual_transform is not None:
            frame, (segm_map, ) = self.visual_transform(frame, (segm_map, ))
        for lbl in range(71):
            segm_map[segm_map==lbl] = lbl if lbl+1 in class_labels else 0

        # Read audio
        areader = avreader.AudioReader(filename=f"{self.path}/{file_id}/audio.wav", rate=self.audio_rate)
        waveform = areader.read(t_min=audio_start_time, t_max=audio_start_time+self.audio_dur)
        waveform = torch.tensor(waveform).float()
        if self.audio_transform is not None:
            audio = self.audio_transform(waveform)[:, :, :-1]
        else:
            audio = waveform
        return frame, segm_map, audio

    def getitem(self, idx):
        anno = {}

        # if self.class_labels is not None:
        #     anno['class'] = self.class_labels[idx]

        file_id = self.samples[idx]
        n_frames = len(glob.glob(f"{self.path}/{file_id}/labels_semantic/*.png"))
        if n_frames == 0:
            return self[random.sample(range(len(self.samples)), 1)[0]]

        # Sample clip
        areader = avreader.AudioReader(filename=f"{self.path}/{file_id}/audio.wav", rate=self.audio_rate)
        if self.temporal_jitter:
            frame_no = random.sample(range(n_frames), 1)[0]
            frame_ts = frame_no + 0.5
            jit = random.uniform(-self.audio_dur*0.33, self.audio_dur*0.33)
            start_time = max(min(frame_ts + jit - self.audio_dur / 2, areader.duration - self.audio_dur), 0)
        else:
            frame_no = n_frames // 2
            frame_ts = frame_no + 0.5
            start_time = max(min(frame_ts - self.audio_dur / 2, areader.duration - self.audio_dur), 0)

        frame, segm_map, audio = self.read_data(file_id, frame_no, start_time, self.class_labels[idx])

        if self.return_semantics:
            anno['gt_map'] = segm_map
        else:
            anno['gt_map'] = (segm_map > 0).float()

        return frame, audio, anno, file_id

    def getitem_dense(self, idx):
        anno = {}

        file_id = self.samples[idx]
        n_frames = len(glob.glob(f"{self.path}/{file_id}/labels_semantic/*.png"))
        if n_frames == 0:
            # print(f"{self.path}/{file_id}/labels_semantic/")
            return self[random.sample(range(len(self.samples)), 1)[0]]

        # Sample clip
        areader = avreader.AudioReader(filename=f"{self.path}/{file_id}/audio.wav", rate=self.audio_rate)
        frame_list, segm_list, audio_list = [], [], []
        for frame_no in range(n_frames):
            frame_ts = frame_no + 0.5
            start_time = max(min(frame_ts - self.audio_dur / 2, areader.duration - self.audio_dur), 0)

            frame, segm_map, audio = self.read_data(file_id, frame_no, start_time, self.class_labels[idx])
            frame_list.append(frame)
            audio_list.append(audio)
            segm_list.append(segm_map)

        if self.return_semantics:
            anno['gt_map'] = torch.stack(segm_list)
        else:
            anno['gt_map'] = (torch.stack(segm_list) > 0).float()
        return torch.stack(frame_list), torch.stack(audio_list), anno, file_id

    def sample_item(self, idx):
        return idx % len(self.samples)

    def __len__(self):
        return int(len(self.samples) * self.oversample)

    def __getitem__(self, idx):
        try:
            if self.dense:
                return self.getitem_dense(self.sample_item(idx))
            else:
                return self.getitem(self.sample_item(idx))
        except Exception:
            return self[random.sample(range(len(self.samples)), 1)[0]]


class BaseVideoDataset(Dataset):
    def __init__(self, base_path, video_files,
                 audio_dur=3., audio_rate=8000,
                 class_labels=None, class_desc=None, temporal_jitter=False):
        super().__init__()
        self.base_path = base_path
        self.video_files = video_files
        self.class_labels = class_labels
        self.class_desc = class_desc

        self.audio_dur = audio_dur
        self.audio_rate = audio_rate
        self.temporal_jitter = temporal_jitter

        # Compute class distribution. Useful for class-balanced recognition losses
        self.class_dist = torch.zeros(len(self.class_desc))
        for lbl in self.class_labels:
            if not isinstance(lbl, (list, tuple)):
                lbl = [lbl]
            for l in lbl:
                self.class_dist[l] += 1
        self.class_dist /= self.class_dist.sum()

    def get_sample_metadata(self, idx):
        file_id = self.video_files[idx].split('.')[0]
        filename = f"{self.base_path}/{self.video_files[idx]}"
        lbl = self.class_labels[idx] if self.class_labels is not None else None
        if isinstance(lbl, (list, tuple)):
            lbl = torch.stack([torch.eye(len(self.class_desc))[l] for l in lbl]).sum(0)
        anno = {} if lbl is None else {'class': lbl, 'file_id': file_id}
        return file_id, filename, anno

    @staticmethod
    def load_audio(areader, start_time, duration, rate=None):
        waveform = areader.read(t_min=start_time, t_max=start_time+duration)
        waveform = torch.tensor(waveform).float().mean(0, keepdims=True)
        if rate is not None and areader.rate != rate:
            waveform = Resample(areader.rate, rate)(waveform)
        return waveform

    @staticmethod
    def load_frame(vreader, start_time, duration, precise=False):
        if precise:
            frame, ts = vreader.precise_frame(t=start_time+duration/2)
        else:
            frame, ts = vreader.quick_random_frame(t_min=start_time, t_max=start_time+duration)
        return frame, ts

    @staticmethod
    def load_clip(vreader, start_time, duration):
        return vreader.get_clip(t_start=start_time, t_end=start_time+duration)

    def getitem(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self[random.sample(range(len(self)), 1)[0]]


class VideoDataset(BaseVideoDataset):
    def __init__(self, base_path, video_files,
                 audio_dur=3., audio_rate=8000,
                 class_labels=None, class_desc=None, temporal_jitter=False,
                 visual_transform=None, audio_transform=None):
        super().__init__(
            base_path=base_path, video_files=video_files,
            audio_dur=audio_dur, audio_rate=audio_rate,
            class_labels=class_labels, class_desc=class_desc, temporal_jitter=temporal_jitter
        )

        self.visual_transform = visual_transform
        self.audio_transform = audio_transform

    def sample_timestamps(self, vreader):
        if self.temporal_jitter:
            midpoint = random.uniform(vreader.start_time + self.audio_dur / 2, vreader.start_time + vreader.duration - self.audio_dur / 2)
        else:
            midpoint = vreader.start_time + vreader.duration / 2.
        start_time = midpoint - self.audio_dur / 2
        return start_time

    def get_sample(self, filename):
        container = av.open(filename)
        vreader = avreader.VideoReader(container=container)
        areader = avreader.AudioReader(container=container)
        start_time = self.sample_timestamps(vreader)

        # Read frame
        frame, ts = self.load_frame(vreader, start_time, self.audio_dur, precise=False)
        frame = self.visual_transform(frame)

        # Read audio
        waveform = self.load_audio(areader, start_time, self.audio_dur, self.audio_rate)
        mel_spec = self.audio_transform(waveform)[:, :, :-1]
        return frame, mel_spec

    def getitem(self, idx):
        file_id, filename, anno = self.get_sample_metadata(idx)
        frame, mel_spec = self.get_sample(filename)
        return frame, mel_spec, anno

    def __repr__(self):
        return f"VideoDataset\n  - Path: {self.base_path}\n  - No Samples: {len(self)}"


class DenseVideoDataset(BaseVideoDataset):
    def __init__(self, base_path, video_files,
                 audio_dur=3., audio_rate=8000,
                 visual_transform=None, audio_transform=None,
                 class_labels=None, class_desc=None, temporal_jitter=False,
                 dense_n=10, dense_span=10):
        super().__init__(
            base_path=base_path, video_files=video_files,
            audio_dur=audio_dur, audio_rate=audio_rate,
            class_labels=class_labels, class_desc=class_desc, temporal_jitter=temporal_jitter
        )

        self.visual_transform = visual_transform
        self.audio_transform = audio_transform
        self.dense_n = dense_n
        self.dense_span = dense_span

    def sample_timestamps(self, vreader):
        if self.temporal_jitter:
            start_time = random.uniform(vreader.start_time, vreader.start_time + vreader.duration - self.dense_span)
        else:
            start_time = max(vreader.start_time + vreader.duration / 2. - self.dense_span / 2, vreader.start_time)
        clip_ts = np.linspace(start_time, start_time + self.dense_span - self.audio_dur, self.dense_n) + self.audio_dur / 2
        return clip_ts

    def getitem(self, idx):
        file_id, filename, anno = self.get_sample_metadata(idx)

        # Read
        container = av.open(filename)
        vreader = avreader.VideoReader(container=container)
        areader = avreader.AudioReader(container=container)

        clip_ts = self.sample_timestamps(vreader)
        video, ts = self.load_clip(vreader, clip_ts[0], clip_ts[-1]-clip_ts[0])
        fno = np.linspace(0, len(ts)-1, self.dense_n, endpoint=True).astype(int)
        dense_frames = [video[i] for i in fno]
        dense_frames = torch.stack([self.visual_transform(frame) for frame in dense_frames], dim=1)

        waveform = self.load_audio(areader, clip_ts[0]-self.audio_dur/2, clip_ts[-1]+self.audio_dur/2, self.audio_rate)
        wlen = int(self.audio_dur * self.audio_rate)
        fno = np.linspace(0, waveform.shape[1] - wlen, self.dense_n, endpoint=True).astype(int)
        dense_waveforms = torch.stack([waveform[:, i:i+wlen] for i in fno])
        dense_specs = self.audio_transform(dense_waveforms)[:, :, :, :-1]
        return dense_frames, dense_specs, anno

    def __repr__(self):
        return f"DenseVideoDataset\n  - Path: {self.base_path}\n  - No Samples: {len(self)}\n  - Class Resample: {self.class_resample}\n  - Mixture: {self.num_mixtures}"


class MixtureVideoDataset(BaseVideoDataset):
    def __init__(self, base_path, video_files, video_files_mix=None,
                 audio_dur=3., audio_rate=8000, num_mixtures=2,
                 visual_transform=None, audio_transform=None,
                 class_labels=None, class_desc=None, temporal_jitter=False):
        super().__init__(
            base_path=base_path, video_files=video_files,
            audio_dur=audio_dur, audio_rate=audio_rate,
            class_labels=class_labels, class_desc=class_desc, temporal_jitter=temporal_jitter
        )
        self.video_files_mix = video_files_mix
        self.num_mixtures = num_mixtures
        self.visual_transform = visual_transform
        self.audio_transform = audio_transform
        assert num_mixtures >= 2

    def get_sample_metadata(self, idx):
        file_ids = [self.video_files[idx].split('.')[0]]
        filenames = [f"{self.base_path}/{self.video_files[idx]}"]
        if self.video_files_mix is not None:
            assert self.num_mixtures == 2
            file_ids += [self.video_files_mix[idx].split('.')[0]]
            filenames += [f"{self.base_path}/{self.video_files_mix[idx]}"]
        else:
            other_idx = [r for r in range(len(self.video_files)) if r != idx]
            mix_idx_list = np.random.choice(other_idx, size=self.num_mixtures-1, replace=False).tolist()
            file_ids += [self.video_files[mix_idx].split('.')[0] for mix_idx in mix_idx_list]
            filenames += [f"{self.base_path}/{self.video_files[mix_idx]}" for mix_idx in mix_idx_list]

        return file_ids, filenames, {}

    def sample_timestamps(self, start, end):
        if self.temporal_jitter:
            tc = random.uniform(start + self.audio_dur / 2, end - self.audio_dur / 2)
        else:
            tc = (start + end) / 2.
        return tc

    def get_sample(self, filenames):
        frames, waveforms, mel_specs = [], [], []
        for filename in filenames:
            container = av.open(filename)
            vreader = avreader.VideoReader(container=container)
            areader = avreader.AudioReader(container=container)
            tc = self.sample_timestamps(
                start=max(vreader.start_time, areader.start_time),
                end=min(vreader.start_time+vreader.duration, areader.start_time+areader.duration)
            )

            # Read frame
            frame, _ = self.load_frame(vreader, tc-self.audio_dur/2, self.audio_dur, precise=False)
            frames.append(self.visual_transform(frame))

            # Read audio
            waveform = self.load_audio(areader, tc-self.audio_dur/2, self.audio_dur, self.audio_rate)
            waveforms.append(waveform)
            mel_specs.append(self.audio_transform(waveform)[:, :, :-1])
        mix_waveform = torch.stack(waveforms).sum(0)
        mix_spec = self.audio_transform(mix_waveform)[:, :, :-1]
        return mix_spec, frames, mel_specs, waveforms

    def getitem(self, idx):
        file_ids, filenames, anno = self.get_sample_metadata(idx)
        mix_spec, frames, mel_specs, waveforms = self.get_sample(filenames)
        anno['waveforms'] = torch.stack(waveforms)
        anno['mel_specs'] = torch.stack(mel_specs)
        return frames, mix_spec, anno

    def __repr__(self):
        return f"MixVideoDataset\n  - Path: {self.base_path}\n  - No Samples: {len(self)}\n  - Mixture: {self.num_mixtures}"


class ImageAudioDataset(Dataset):
    def __init__(self, data_path, image_files, audio_files,
                 audio_dur=3., audio_rate=8000, num_mixtures=1,
                 visual_transform=None, audio_transform=None,
                 anno_files=None, anno_loader=None,
                 class_labels=None, class_desc=None,
                 class_resample=0, video_files_mix=None, oversample=None):
        super().__init__()
        self.data_path = data_path
        self.image_files = image_files
        self.audio_files = audio_files
        self.anno_files = anno_files
        self.class_labels = class_labels
        self.class_desc = class_desc

        self.audio_dur = audio_dur
        self.audio_rate = audio_rate
        self.num_mixtures = num_mixtures

        self.visual_transform = visual_transform
        self.audio_transform = audio_transform
        self.anno_loader = anno_loader

        self.class_resample = class_resample
        if self.class_resample:
            self.class2samples = defaultdict(list)
            if isinstance(self.class_labels[0], (list, tuple)):
                [self.class2samples[lbl].append(idx) for idx, lbl_list in enumerate(self.class_labels) for lbl in lbl_list]
            else:
                [self.class2samples[lbl].append(idx) for idx, lbl in enumerate(self.class_labels)]
        self.video_files_mix = video_files_mix

        self.oversample = oversample if oversample is not None else 1

    def sample(self, idx):
        idx = idx % len(self.image_files)
        if self.class_resample:
            lbl = random.sample(range(len(self.class2samples)), 1)[0]
            idx = random.sample(self.class2samples[lbl], 1)[0]
        return idx

    def get_sample_meta(self, idx):
        file_id = self.image_files[idx].split('.')[0]
        image_filename = f"{self.data_path}/{self.image_files[idx]}"
        audio_filename = f"{self.data_path}/{self.audio_files[idx]}"

        anno = {}
        if self.class_labels is not None:
            anno['class'] = self.class_labels[idx]
            if isinstance(anno['class'], (list, tuple)):
                anno['class'] = torch.stack([torch.eye(len(self.class_desc))[l] for l in anno['class']]).sum(0)
        anno_fn = f"{self.data_path}/{self.anno_files[idx]}" if self.anno_files is not None else None
        anno.update(self.anno_loader(anno_fn))
        return file_id, image_filename, audio_filename, anno

    def read_audio(self, start_time, duration, areader, seek=True):
        waveform = areader.read(t_min=start_time, t_max=start_time+duration, seek=seek)
        waveform = torch.tensor(waveform).float().mean(0, keepdims=True)
        if self.audio_rate is not None:
            waveform = Resample(areader.rate, self.audio_rate)(waveform)
        return waveform

    def read_frame(self, start_time, duration, vreader, precise=False, seek=True):
        if precise:
            frame, ts = vreader.precise_frame(t=start_time+duration/2, seek=seek)
        else:
            frame, ts = vreader.quick_random_frame(t_min=start_time, t_max=start_time+duration, seek=seek)
        if self.visual_transform is not None:
            frame = self.visual_transform(frame)

        return frame, ts

    def get_avdata(self, image_fn, audio_fn, anno=None):
        # Read frame
        frame = load_image(image_fn)
        if self.visual_transform is not None:
            if anno and 'gt_map' in anno:
                frame_prep, pixel_anno = self.visual_transform(frame, anno['gt_map'])
                anno['gt_map'] = np.array(pixel_anno[0])
            else:
                frame_prep, _ = self.visual_transform(frame)
        else:
            frame_prep = frame

        # Read audio
        # areader = avreader.AudioReader(container=container)
        # waveform = areader.read(t_min=start_time, t_max=start_time+self.audio_dur)
        # waveform = torch.tensor(waveform).float()
        ameta = torchaudio.info(audio_fn)
        audio_dur = ameta.num_frames / ameta.sample_rate
        start_time = (audio_dur - self.audio_dur) / 2
        waveform, arate = torchaudio.load(audio_fn, frame_offset=int(start_time*ameta.sample_rate), num_frames=int(self.audio_dur*ameta.sample_rate))
        waveform = waveform.mean(0, keepdims=True)
        if self.audio_rate is not None:
            waveform = Resample(arate, self.audio_rate)(waveform)
        audio_prep = waveform
        if self.audio_transform is not None:
            audio_prep = self.audio_transform(waveform)[:, :, :-1]

        return frame_prep, audio_prep, frame, waveform, anno

    def getitem(self, idx):
        file_id, image_fn, audio_fn, anno = self.get_sample_meta(idx)
        frame, audio, frame_orig, waveform, anno = self.get_avdata(image_fn, audio_fn, anno)

        # Mix waveform with another ones
        if self.num_mixtures > 1:
            mix_waveforms, frames = [waveform], [frame]
            mix_idx_list = np.random.choice([r for r in range(len(self.image_files)) if r != idx],
                                            size=self.num_mixtures-1, replace=False).tolist()
            filenames_mix = [self.get_sample_meta(mix_idx)[1:3] for mix_idx in mix_idx_list]

            for mix_image_fn, mix_audio_fn in filenames_mix:
                mix_frame, _, _, mix_wav, _ = self.get_avdata(mix_image_fn, mix_audio_fn)
                frames.append(mix_frame)
                mix_waveforms.append(mix_wav)

            mixed_waveform = torch.stack(mix_waveforms).sum(0)
            mix_audio = mixed_waveform
            if self.audio_transform is not None:
                mix_audio = self.audio_transform(mixed_waveform)[:, :, :-1]
            anno['waveforms'] = torch.stack(mix_waveforms)
            anno['frames'] = torch.stack(frames)
            anno['mixed_audio'] = mix_audio
        return frame, audio, anno, file_id

    def __len__(self):
        return int(len(self.image_files) * self.oversample)

    def __getitem__(self, idx):
        idx = self.sample(idx)
        return self.getitem(idx)

    def __repr__(self):
        return f"VideoDataset\n  - Path: {self.data_path}\n  - No Samples: {len(self)}\n  - Class Resample: {self.class_resample}\n  - Mixture: {self.num_mixtures}"


def get_vggsound(data_path, dataset=VideoDataset, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    data = list(csv.reader(open(f"{data_path}/annotations/vggsound.csv")))
    data = [dt for dt in data if dt[-1] == partition]
    dictionary = sorted(os.listdir(f"{data_path}/clips/"))
    all_filenames, all_labels = [], []
    for yid, t, cls, part in data:
        cls = cls .replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_')
        all_filenames.append(f"{cls}/{yid}_{int(t):06d}_{int(t)+10:06d}.mp4")
        all_labels.append(dictionary.index(cls))

    files_available = set(['/'.join(fn.split('/')[-2:]) for fn in glob.glob(f"{data_path}/clips/*/*.mp4")])
    filenames = [fn for fn, lbl in zip(all_filenames, all_labels) if fn in files_available]
    class_labels = [lbl for fn, lbl in zip(all_filenames, all_labels) if fn in files_available]
    # available = [idx for idx, fn in enumerate(filenames) if os.path.isfile(f"{data_path}/clips/{fn}")]

    return dataset(
        video_files=filenames,
        base_path=f"{data_path}/clips",
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        class_desc=dictionary,
        **kwargs
    )


def get_vggsound_music(data_path, dataset=VideoDataset, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    if partition == 'train':
        data = list(csv.reader(open(f"metadata/vggmusic_train.txt")))
        vocab = sorted(list(set([cls.replace('violin', 'violin__fiddle').replace('steel_guitar', 'steel_guitar__slide_guitar') for yid, cls in data])))
        filenames, class_labels = defaultdict(list), defaultdict(list)
        for yid, cls in data:
            cls = cls.replace('violin', 'violin__fiddle').replace('steel_guitar', 'steel_guitar__slide_guitar')
            fn = f"playing_{cls}/{yid[:11]}_{int(yid[-6:]):06d}_{int(yid[-6:])+10:06d}.mp4"
            if not os.path.exists(f"{data_path}/clips/{fn}"):
                continue
            filenames[yid[:11]].append(fn)
            class_labels[yid[:11]].append(vocab.index(cls))
        filenames2 = None

    else:
        data = list(csv.reader(open(f"metadata/vggmusic_eval_ss.csv")))[1:]
        filenames = [f"playing_{cls1}/{yid1[:11]}_{int(yid1[-6:]):06d}_{int(yid1[-6:])+10:06d}.mp4"
                     for yid1, yid2, cls1, cls2, _ in data]
        filenames2 = [f"playing_{cls2}/{yid2[:11]}_{int(yid2[-6:]):06d}_{int(yid2[-6:])+10:06d}.mp4" for
                      yid1, yid2, cls1, cls2, _ in data]
        class_labels = None

    return dataset(
        base_path=f"{data_path}/clips",
        video_files=filenames,
        video_files_mix=filenames2,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        **kwargs,
    )


def get_music(data_path, dataset=VideoDataset, partition='train', version='solo', visual_transform=None, audio_transform=None, **kwargs):
    if version == 'solo':
        data = [list(smp) + ['solo'] for smp in csv.reader(open(f"{data_path}/anno/music_solo.csv"))][1:]
    elif version == 'solo21':
        data = [list(smp) + ['solo'] for smp in csv.reader(open(f"{data_path}/anno/music21_solo.csv"))][1:]
    elif version == 'music':
        data = [list(smp) + ['solo'] for smp in csv.reader(open(f"{data_path}/anno/music_solo.csv"))][1:]
        data += [list(smp) + ['duet'] for smp in csv.reader(open(f"{data_path}/anno/music21_duet.csv"))][1:]
    else:
        raise ValueError(f'Unknown MUSIC dataset version: {version}')

    vocab = sorted(list(set([cls.replace(' ', '_') for yid, cls, _, dtype in data])))
    filenames, class_labels, sample_type = defaultdict(list), defaultdict(list), {}
    for yid, cls, _, dtype in data:
        cls = cls.replace(' ', '_')
        fns = [fn.replace(f"{data_path}/clips_360p_segm/", "")
              for fn in glob.glob(f"{data_path}/clips_360p_segm/{cls}/{yid}.*.mp4")]
        if len(fns) > 0:
            filenames[yid].extend(fns)
            class_labels[yid].extend([vocab.index(cls)] * len(fns))
            sample_type[yid] = dtype

    # Random Train/Test Partition
    all_video_ids = sorted(list(filenames.keys()))
    solo_video_ids = sorted([yid for yid, dtype in sample_type.items() if dtype == 'solo'])
    duets_video_ids = sorted([yid for yid, dtype in sample_type.items() if dtype == 'duet'])
    eval_vids = set(solo_video_ids[::len(solo_video_ids)//130])
    test_vids = set(duets_video_ids[::len(duets_video_ids)//85]) if duets_video_ids else set()
    train_vids = set(all_video_ids) - eval_vids - test_vids
    if partition == 'train':
        filenames = {vid: filenames[vid] for vid in filenames if vid in train_vids}
        class_labels = {vid: class_labels[vid] for vid in class_labels if vid in train_vids}
    else:
        filenames = {vid: filenames[vid] for vid in filenames if vid in eval_vids}
        class_labels = {vid: class_labels[vid] for vid in class_labels if vid in eval_vids}
    oversample = int(math.ceil(sum([len(filenames[vid]) for vid in filenames]) / len(filenames)))

    return dataset(
        base_path=f"{data_path}/clips_360p_segm",
        video_files=filenames,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        oversample=oversample,
        **kwargs,
    )


def get_audioset(data_path, dataset=VideoDataset, partition='unbalanced_train', visual_transform=None, audio_transform=None, class_resample=0, **kwargs):
    ontology = list(csv.reader(open(f"{data_path}/annotations/class_labels_indices.csv")))[1:]
    labels = {cls: int(idx) for idx, cls, desc in ontology}
    desc = [desc for idx, cls, desc in ontology]
    data = list(csv.reader(open(f"{data_path}/annotations/{partition}_segments.csv")))[3:]
    data = [    # VIDEO_ID,START_SECONDS,END_SECONDS,LABELS
        (d[0], float(d[1].strip()), float(d[2].strip()), [labels[cls.strip().replace('"', '')] for cls in d[3:]])
        for d in data
    ]

    files_available = set(['/'.join(fn.split('/')[-2:]) for fn in glob.glob(f"{data_path}/clips/*/*.mp4")])
    filenames, class_labels = [], []
    for yid, st, et, cls in data:
        fn = f"{yid[:2]}/{yid}_{int(st):06d}_{int(et):06d}.mp4"
        if fn in files_available:
            filenames.append(fn)
            class_labels.append(cls)
    print("Done checking files.")

    return dataset(
        video_files=filenames,
        base_path=f"{data_path}/clips",
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        class_desc=desc,
        class_resample=class_resample,
        **kwargs
    )


def get_avsbench_s4(data_path, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    data = list(csv.reader(open(f"{data_path}/metadata.csv")))[1:]
    classes = json.load(open(f"{data_path}/label2idx.json"))
    samples, class_labels = [], []
    data = [dt for dt in data if dt[-2] == partition and dt[-1] == 'v1s']
    for vid, uid, s_min, s_sec, a_obj, split, label in data:
        folder = f"{label}/{uid}"
        if os.path.exists(f"{data_path}/{folder}"):
            samples.append(folder)
            class_labels.append([classes[a_obj]])
    oversample = 10 if partition == 'train' else 1

    return FolderVideoDataset(
        path=data_path,
        samples=samples,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        oversample=oversample,
        return_semantics=False,
        **kwargs
    )


def get_avsbench_ms3(data_path, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    data = list(csv.reader(open(f"{data_path}/metadata.csv")))[1:]
    classes = json.load(open(f"{data_path}/label2idx.json"))
    samples, class_labels = [], []
    data = [dt for dt in data if dt[-2] == partition and dt[-1] == 'v1m']
    for vid, uid, s_min, s_sec, a_obj, split, label in data:
        folder = f"{label}/{uid}"
        if os.path.exists(f"{data_path}/{folder}"):
            samples.append(folder)
            class_labels.append([classes[cls] for cls in a_obj.split('_')])
    oversample = 100 if partition == 'train' else 1

    return FolderVideoDataset(
        path=data_path,
        samples=samples,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        oversample=oversample,
        return_semantics=False,
        **kwargs
    )


def get_avsbench_avss(data_path, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    data = list(csv.reader(open(f"{data_path}/metadata.csv")))[1:]
    classes = json.load(open(f"{data_path}/label2idx.json"))
    data = [dt for dt in data if dt[-2] == partition]
    samples, class_labels = [], []
    for vid, uid, s_min, s_sec, a_obj, split, label in data:
        folder = f"{label}/{uid}"
        if os.path.exists(f"{data_path}/{folder}"):
            samples.append(folder)
            class_labels.append([classes[cls.replace('off-the-screen', 'background')] for cls in a_obj.split('_')])
    oversample = 5 if partition == 'train' else 1

    return FolderVideoDataset(
        path=data_path,
        samples=samples,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        class_labels=class_labels,
        oversample=oversample,
        return_semantics=True,
        **kwargs
    )


def flickr_anno_parser(fn):
    import xml.etree.ElementTree as ET
    bboxes = [node for field in ET.parse(fn).getroot() for node in field if node.tag == 'bbox']
    bboxes = [[int(ch.text) * 224 // 256 for ch in bb[1:]] for bb in bboxes]

    # Annotation consensus
    loc_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        loc_map[ymin:ymax, xmin:xmax] += 1
    loc_map = np.clip(loc_map / 2, a_min=0, a_max=1)

    return {'gt_map': Image.fromarray(loc_map)}


def load_flickr_soundnet(data_path, partition='train', visual_transform=None, audio_transform=None, **kwargs):
    assert partition == 'val'
    video_ids = [vid for vid, t in csv.reader(open(f"metadata/flickr_test.csv"))]
    frame_fns = [f"frames/{vid}.jpg" for vid in video_ids]
    audio_fns = [f"audio/{vid}.wav" for vid in video_ids]
    anno_fns = [f"Annotations/{vid}.xml" for vid in video_ids]
    assert all([os.path.isfile(f"{data_path}/{fn}") for fn in frame_fns])
    assert all([os.path.isfile(f"{data_path}/{fn}") for fn in audio_fns])
    assert all([os.path.isfile(f"{data_path}/{fn}") for fn in anno_fns])

    return ImageAudioDataset(
        data_path, frame_fns, audio_fns,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        anno_files=anno_fns,
        anno_loader=flickr_anno_parser,
        **kwargs
    )


def load_dataset(dataset, data_path, dataset_type='simple', visual_transform=None, audio_transform=None, train=True, **kwargs):
    if dataset_type == 'simple':
        dataset_class = VideoDataset
    elif dataset_type == 'dense':
        dataset_class = DenseVideoDataset
    elif dataset_type == 'mixed_audio':
        dataset_class = MixtureVideoDataset
    else:
        raise NotImplemented

    if dataset == 'audioset':
        return get_audioset(data_path, dataset=dataset_class, partition='unbalanced_train' if train else 'eval', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'audioset-bal':
        return get_audioset(data_path, dataset=dataset_class, partition='unbalanced_train' if train else 'eval', visual_transform=visual_transform, audio_transform=audio_transform, class_resample=100, **kwargs)
    elif dataset == 'audioset-bal-orig':
        return get_audioset(data_path, dataset=dataset_class, partition='balanced_train' if train else 'eval', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'vggsound':
        return get_vggsound(data_path, dataset=dataset_class, partition='train' if train else 'test', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'vggsound_music':
        return get_vggsound_music(data_path, dataset=dataset_class, partition='train' if train else 'test', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'music':
        return get_music(data_path, dataset=dataset_class, partition='train' if train else 'test', version='music', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'music_solo':
        return get_music(data_path, dataset=dataset_class, partition='train' if train else 'test', version='solo', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'music_solo21':
        return get_music(data_path, dataset=dataset_class, partition='train' if train else 'test', version='solo21', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'avsbench_s4':
        return get_avsbench_s4(data_path, partition='train' if train else 'val', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'avsbench_ms3':
        return get_avsbench_ms3(data_path, partition='train' if train else 'val', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'avsbench_avss':
        return get_avsbench_avss(data_path, partition='train' if train else 'val', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    elif dataset == 'flickr_soundnet_5k':
        return load_flickr_soundnet(data_path, partition='train' if train else 'val', visual_transform=visual_transform, audio_transform=audio_transform, **kwargs)
    else:
        raise NotImplementedError


NUM_CLASSES = {
    'audioset': 527,
    'audioset-bal': 527,
    'audioset-bal-orig': 527,
    'vggsound': 310,
    'avsbench_avss': 71,
    'avsbench_s4': 2,
    'avsbench_ms3': 2,
    'music_solo': 11,
    'music_solo21': 21,
}
MULTI_CLASS_DBS = {
    'audioset': True,
    'audioset-bal': True,
    'audioset-bal-orig': True,
    'vggsound': False,
}

if __name__ == '__main__':
    from pytorchvideo import transforms as vT
    from util import audio_transforms as aT
    from util import data as data_utils
    from tqdm import tqdm
    db = load_dataset(
        'vggsound', '/home/datasets/vggsounds',
        dataset_type='avsync',
        audio_rate=16000,
        visual_transform=aT.Compose([
            vT.UniformTemporalSubsample(16),
            vT.RandomResizedCrop(224, 224, scale=(0.5, 1.), aspect_ratio=(3/4, 4/3)),
            vT.Div255(),
            vT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        audio_transform=aT.Compose([
            aT.Pad(rate=16000, dur=3),
            aT.RandomVol(),
            aT.MelSpectrogram(sample_rate=16000, n_fft=int(16000 * 0.05), hop_length=int(16000 / 64), n_mels=128),
            aT.Log()
        ]),
        temporal_jitter=True,
        sync_prob=0.5,
        asyn_gap=(0.125, float('inf'))
    )
    db[0]
    loader = data_utils.get_dataloader(db, distributed=True, batch_size=32, workers=4)
    for x1, x2, x3, x4 in tqdm(loader):
        pass

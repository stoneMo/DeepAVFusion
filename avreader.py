import glob
import random
import av
from fractions import Fraction
import numpy as np


class VideoReader:
    def __init__(self, filename=None, container=None):
        self.container = av.open(filename) if container is None else container
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 4

    def quick_random_frame(self, t_min=None, t_max=None):
        t_min = self.start_time if t_min is None else t_min
        t_max = self.start_time + self.duration if t_max is None else t_max
        rnd_t = random.uniform(t_min, t_max)
        self.container.seek(int(rnd_t * av.time_base))
        for frame in self.container.decode(video=0):
            frame_ts = float(frame.pts * frame.time_base)
            frame = frame.to_image()
            return frame, frame_ts

    def precise_frame(self, t, seek=True):
        if seek:
            self.container.seek(int(t * av.time_base))
        for frame in self.container.decode(video=0):
            frame_ts = float(frame.pts * frame.time_base)
            if t - frame_ts < 1 / self.fps:
                frame = frame.to_image()
                return frame, frame_ts

    def get_clip(self, t_start=None, t_end=None):
        t_start = self.start_time if t_start is None else t_start
        t_end = self.start_time + self.duration if t_end is None else t_end
        self.container.seek(int(t_start * av.time_base))
        clip, ts = [], []
        for frame in self.container.decode(video=0):
            frame_ts = float(frame.pts * frame.time_base)
            if frame_ts < t_start:
                continue
            if frame_ts > t_end:
                return clip, ts
            clip.append(frame.to_image()), ts.append(frame_ts)
        return clip, ts

    def __iter__(self):
        for frame in self.container.decode(video=0):
            frame_ts = float(frame.pts * frame.time_base)
            frame = frame.to_image()
            yield frame, frame_ts

    def __len__(self):
        return self.num_frames

    @property
    def fps(self):
        return self.stream.average_rate

    @property
    def num_frames(self):
        return self.stream.frames

    @property
    def duration(self):
        return self.stream.duration * self.stream.time_base

    @property
    def start_time(self):
        return self.stream.start_time * self.stream.time_base


class AudioReader:
    def __init__(self, filename=None, container=None, rate=None, layout='mono'):
        self.container = av.open(filename) if container is None else container
        self.stream = self.container.streams.audio[0]
        self.stream.thread_count = 4
        self.resampler = None
        self.rate = self.orig_rate
        if rate is not None:
            self.resampler = av.audio.resampler.AudioResampler(format="s16p", layout=layout, rate=rate)
            self.rate = rate

    def read(self, t_min=None, t_max=None, seek=True):
        t_min = self.start_time if t_min is None else t_min
        t_max = self.start_time + self.duration if t_max is None else t_max
        if seek:
            self.container.seek(int(t_min * av.time_base))

        # Read data
        chunks = []
        for chunk in self.container.decode(audio=0):
            chunk_ts = chunk.pts * chunk.time_base
            chunk_end = chunk_ts + Fraction(chunk.samples, chunk.rate)
            if chunk_end < t_min:   # Skip until start time
                continue
            if chunk_ts > t_max:       # Exit if clip has been extracted
                break

            # Resample
            chunk.pts = None
            if self.resampler is not None:
                chunk = self.resampler.resample(chunk)
                if isinstance(chunk, list):
                    assert len(chunk) == 1
                    chunk = chunk[0]
                chunk = chunk.to_ndarray()
                chunk = chunk / np.iinfo(chunk.dtype).max
            else:
                chunk = chunk.to_ndarray()

            if chunk_ts < t_min:
                chunk = chunk[:, int((t_min - chunk_ts) * self.rate):]
            if chunk_end > t_max:
                chunk = chunk[:, :-int((chunk_end - t_max) * self.rate)]
            chunks.append(chunk)

        # Trim for frame accuracy
        audio = np.concatenate(chunks, 1)

        nframes = int((t_max - t_min) * self.rate)
        if nframes > audio.shape[1]:
            audio = np.pad(audio, [(0, 0), (0, nframes-audio.shape[1])],mode='symmetric')
        if nframes < audio.shape[1]:
            audio = audio[:, :nframes]

        return audio

    @property
    def orig_rate(self):
        return self.stream.rate

    @property
    def num_frames(self):
        return self.stream.frames

    @property
    def duration(self):
        return self.stream.duration * self.stream.time_base

    @property
    def start_time(self):
        return self.stream.start_time * self.stream.time_base if self.stream.start_time is not None else 0.


if __name__ == '__main__':
    import time
    fns = glob.glob('/home/pmorgado/datasets/vggsound/clips/*/*.mp4')
    t_open, t_load, t_load_audio, t_load_prec = 0., 0., 0., 0.
    for i in range(100):
        t = time.time()
        vreader = VideoReader(fns[random.randint(0, len(fns)-1)])
        areader = AudioReader(fns[random.randint(0, len(fns)-1)], rate=20500)
        midpoint = vreader.start_time + vreader.duration / 2.
        t_open += time.time() - t

        t = time.time()
        frame, ts = vreader.quick_random_frame(midpoint - 3/2, midpoint + 3/2)
        t_load += time.time() - t

        t = time.time()
        frame = areader.read(midpoint - 3/2, midpoint + 3/2)
        t_load_audio += time.time() - t

        t = time.time()
        frame, ts = vreader.precise_frame(random.uniform(midpoint - 3/2, midpoint + 3/2))
        t_load_prec += time.time() - t
    print(t_open/100, t_load/100, t_load_prec/100)
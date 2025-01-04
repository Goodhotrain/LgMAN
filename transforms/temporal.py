import random


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size, loop the indices as many times as necessary.
    """

    def __init__(self, size, seed=0):
        self.size = size

    def __call__(self, frame_indices):
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin = random.randint(0, rand_end)
        end = min(begin + self.size, len(frame_indices))
        out = frame_indices[begin:end]
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at the center.
    If the number of frames is less than the size, loop the indices as many times as necessary.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        center_index = len(frame_indices) // 2
        begin = max(0, center_index - (self.size // 2))
        end = min(begin + self.size, len(frame_indices))

        out = frame_indices[begin:end]
        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        return out


class TSN(object):
    def __init__(self, n_frames=8, center=False):
        self.n_frames = n_frames
        self.crop = center

    def __call__(self, frame_indices):
        total_frames = len(frame_indices)
        if total_frames == 0:
            assert False, "total_frames is 0"
        if total_frames < self.n_frames:
            frame_indices =frame_indices*self.n_frames
            frame_indices = sorted(frame_indices)
            total_frames = len(frame_indices)
        stride = max(1, total_frames // self.n_frames)
        start_pos = 0 if not self.crop else (total_frames - stride * (self.n_frames - 1)) // 2
        snippets = [frame_indices[i] for i in range(start_pos, total_frames, stride)]
        return snippets[:self.n_frames]

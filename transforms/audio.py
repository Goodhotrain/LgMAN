import random
import numpy as np

class TSNAudio(object):
    def __init__(self, n_frames=8, center=False):
        self.n_frames = n_frames
        self.crop = center

    def __call__(self, audio_frames):
        if audio_frames == 0:
            assert False, "audio_frames is 0"

        stride = max(1, audio_frames // self.n_frames)
        start_pos = 0 if not self.crop else (audio_frames - stride * (self.n_frames - 1)) // 2
        indices = [m for m in range(start_pos, audio_frames, stride)]
        indices = indices[:self.n_frames]

        snippets = []
        for i in indices:
            s = max(0, i - 128)
            e = min(audio_frames, i + 128)
            snippet = [o for o in range(s, e)]
            if len(snippet) < 256:
                if s == 0: 
                    snippet.extend(range(e, e + 256 - len(snippet)))
                else:
                    snippet = list(range(s - (256 - len(snippet)), s)) + snippet
            snippets.append(snippet)
        return snippets


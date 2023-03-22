import numpy as np
import librosa
import torchaudio
import torch
from logzero import logger

class splitter:

    def split_audio_tensor(self, path):

        y = librosa.load(path, sr = None, mono = True)

        segment_dur_secs = 5
        segment_length = y[1] * segment_dur_secs
        num_sections = int(np.ceil(len(y[0]) / segment_length))
        
        split = []

        for i in range(num_sections):
            t = y[0][i * segment_length: (i + 1) * segment_length]
            split.append([torch.tensor([t]), y[1]])
            
        return split
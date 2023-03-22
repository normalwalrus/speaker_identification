from utils.audio_dataloader import dataLoader_extraction
from utils.load_model import model_loader
import torch
import os
import numpy as np

class tester():

    def __init__(self) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_features(self, audio_path: str, audio_length: int):

        DL = dataLoader_extraction(audio_path)

        DL.rechannel(1)
        DL.pad_trunc(audio_length)
        DL.resample(16000)

        features = DL.MFCC_extraction(n_mfcc = 80, mean = False, max_ms=audio_length)
        features = np.array([features])

        return features

    def predict(self, audio_path: str, audio_length: int):

        features = self.load_features(audio_path, audio_length)

        features = torch.from_numpy(features).type(torch.double).to(self.device)

        model = model_loader.load_model().load_state_dict(torch.load(os.environ.get('PATH_TO_MODEL')))

        predicted = torch.argmax(model(features))

        return predicted
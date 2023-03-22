from utils.audio_dataloader import dataLoader_extraction
from utils.load_model import model_loader, LABELS
import torch
import numpy as np

class tester:

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_features(self, audio, audio_length):

        DL = dataLoader_extraction(audio)

        DL.rechannel(1)
        DL.pad_trunc(audio_length)
        DL.resample(16000)

        features = DL.MFCC_extraction(n_mfcc = 80, mean = False, max_ms=audio_length)
        features = np.array([features])

        return features

    def predict(self, audio_path, audio_length = 5000):

        features = self.load_features(audio_path, audio_length)

        features = torch.from_numpy(features).type(torch.double).to(self.device)

        model = model_loader.load_model().to(self.device)

        predicted = torch.argmax(model(features))

        predicted = LABELS[predicted]

        return predicted
import os
import torch
from dotenv import load_dotenv
from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.models.neural_net import FeedForwardNN
from utils.audio_dataloader import dataLoader_extraction
from utils.models.encoder import Encoder
import numpy as np
from logzero import logger

LABELS = ['Lee Hsien Loong', 'Amy Khor Lean Suan', 'Leong Mun Wai', 'Indranee Rajah', 
            'Ong Ye Kung', 'K Shanmugam', 'Tan See Leng', 'Teo Chee Hean', 'Vivian Balakrishnan', 
            'Lawrence Wong', 'Sylvia Lim', 'He Ting Ru', 'Rahayu Mahzam', 'Tan Chuan-Jin', 
            'Desmond Lee', 'Josephine Teo', 'Janil Puthucheary', 'Gan Kim Yong', 'Murali Pillai', 'Tin Pei Ling']

load_dotenv()

class model_loader:

    def __init__(self, device) -> None:
        self.device = device

    def load_features(self, audio, audio_length, datatype, classifier):

        DL = dataLoader_extraction(audio)

        DL.rechannel(1)
        DL.pad_trunc(int(audio_length))
        DL.resample(16000)

        match datatype:

            case 'MFCC':
                logger.info('MFCC Extraction...')
                features = DL.MFCC_extraction(n_mfcc = 80, mean = False, max_ms=audio_length)
                features = np.array([features])
            case 'Audio':
                logger.info('Audio Extraction...')
                features = DL.y[0]

        if classifier:
            logger.info('Encoding using classifier...')
            features = torch.tensor(classifier.encode_batch(features, device = self.device))
            features = features[0][0].type(torch.float64)
            features = features.cpu().numpy()

        return features

    def load_model(self, model : str):

        labels = LABELS
        no_speakers = len(labels)
        classifier = None

        match model:

            case 'ECAPA_TDNN':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ECAPA_TDNN')
                datatype = 'MFCC'
                model = ECAPA_TDNN(157 ,512, no_speakers)

            case 'ECAPA_TDNN_pretrained':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ECAPA_TDNN_pretrained')
                datatype = 'Audio'
                model = FeedForwardNN(192, len(labels), 0.5)
                classifier = Encoder.from_hparams(
                    source="yangwang825/ecapa-tdnn-vox2"
                ).to(self.device)

        model.load_state_dict(torch.load(save_path))
        model.eval().double()

        return model, datatype, classifier

import os
import torch
import torchaudio
import pickle
from dotenv import load_dotenv
from utils.models.ECAPA_TDNN import ECAPA_TDNN
from utils.models.neural_net import FeedForwardNN, ConvNN
from utils.models.ResNet import ResBottleneckBlock, ResNet, ResBlock
from utils.audio_dataloader import dataLoader_extraction
from utils.models.encoder import Encoder
from utils.models.xvector import xvecTDNN, xvecExtraction
import numpy as np
from logzero import logger

LABELS = ['Lee Hsien Loong', 'Amy Khor Lean Suan', 'Leong Mun Wai', 'Indranee Rajah', 
            'Ong Ye Kung', 'K Shanmugam', 'Tan See Leng', 'Teo Chee Hean', 'Vivian Balakrishnan', 
            'Lawrence Wong', 'Sylvia Lim', 'He Ting Ru', 'Rahayu Mahzam', 'Tan Chuan-Jin', 
            'Desmond Lee', 'Josephine Teo', 'Janil Puthucheary', 'Gan Kim Yong', 'Murali Pillai', 'Tin Pei Ling']

load_dotenv()

class model_loader:

    """
    Class is used to load in models from their .pt files and features from their audio Torch.tensor
    """

    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_features(self, audio, audio_length, params):
        """
        Function to load in the features of the Torch.tensor file that is provided

        Parameters
        ----------
            audio : Torch.tensor
                Tensor that describes the audio provided

            audio_length : Integer
                Duration of resultant tensors from audio clip in ms

            params: 
                1. datatype: String describing what datatype is neeed for the model from the audio Tensor
                2. classifier: model that is used to get embeddings from pretrained model (None if no classifier)
                3. Wav2Vec2: Wav2Vec2 Model used to get embeddings (None is no Wav2Vec2 Model used)
                4. List of models: NO USE FOR THIS PARAM

        Returns
        ----------
            features: Numpy array
                Features extracted into Numpy array
        """

        datatype, classifier, Wav2Vec2, _ = params

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
            case 'Spectrogram':
                logger.info('Spectrogram Extraction...')
                features = DL.spectro_gram()
                features = np.array([features])

        if classifier:
            logger.info('Encoding using Pretrained ECAPA_TDNN...')
            features = torch.tensor(classifier.encode_batch(features, device = self.device))
            features = features[0][0].type(torch.float64)
            features = features.cpu().numpy()

        if Wav2Vec2:
            logger.info('Encoding using Wav2Vec2...')
            features = Wav2Vec2.extract_features(features.type(torch.float))
            features = features[0][0].type(torch.double)
            features = features[None,:].detach().numpy()

        return features

    def load_model(self, model : str):
        """
        Function to load in the model with the respective pretrained model

        Parameters
        ----------
            model : String
                String of the model to be used for inference
        Returns
        ----------
            model: Torch model
                Torch model used for inference
            
            classifier: Torch Model
                Any model used for embedding prior to inference by model
            
            Wav2Vec2: Torch Model
                Wav2Vec2 pretrained model used for embedding if needed
            
            plda_model: Python Class
                PLDA model used for inference after the model
            
            models_classifier: List of Strings
                List of models used for the Neural Network inference with embeddings from other models
        """

        labels = LABELS
        no_speakers = len(labels)
        classifier = None
        Wav2Vec2 = None
        plda_model = None
        models_classfier = None

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

            case 'Wav2Vec2_Embedding':
                save_path = os.getcwd() + os.environ.get('PATH_TO_CNN_Wav2Vec2')
                datatype = 'Audio'
                model = ConvNN(len(labels))
                bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
                Wav2Vec2 = bundle.get_model()

            case 'CNN':
                save_path = os.getcwd() + os.environ.get('PATH_TO_CNN')
                datatype = 'Spectrogram'
                model = ConvNN(len(labels))

            case 'ResNet34':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ResNet34')
                datatype = 'Spectrogram'
                model = ResNet(1, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=len(labels))

            case 'ResNet50':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ResNet50')
                datatype = 'Spectrogram'
                model = ResNet(1, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=len(labels))

            case 'ResNet101':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ResNet101')
                datatype = 'Spectrogram'
                model = ResNet(1, ResBottleneckBlock, [3, 4, 23, 3], useBottleneck=True, outputs=len(labels))
            
            case 'XvecTDNN':
                save_path = os.getcwd() + os.environ.get('PATH_TO_XvecTDNN')
                datatype = 'MFCC'
                model = xvecTDNN(len(labels), 0.25, 157)

            case 'XvecPLDA':
                save_path = os.getcwd() + os.environ.get('PATH_TO_XvecPLDA')
                datatype = 'MFCC'
                model = xvecExtraction(len(labels), 0.5, 157)
                plda_model = pickle.load(open(os.getcwd() + os.environ.get('PATH_TO_PLDA'),"rb"))

            case 'ResNet_DNN_classifier':
                save_path = os.getcwd() + os.environ.get('PATH_TO_ResNet_DNN')
                datatype = 'Embeddings'
                model = FeedForwardNN(60, len(labels), 0.5)
                models_classfier = ['ResNet34', 'ResNet50', 'ResNet101']

            case 'All_DNN_classifier':
                save_path = os.getcwd() + os.environ.get('PATH_TO_All_DNN')
                datatype = 'Embeddings'
                model = FeedForwardNN(140, len(labels), 0.5)
                models_classfier = ['ResNet34', 'ResNet50', 'ResNet101', 'CNN', 'ECAPA_TDNN', 'Wav2Vec2_Embedding', 'XvecTDNN']

        logger.info(save_path)
        model.load_state_dict(torch.load(save_path))
        model.eval().double()

        return model, datatype, classifier, Wav2Vec2, plda_model, models_classfier

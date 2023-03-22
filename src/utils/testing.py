from utils.audio_dataloader import dataLoader_extraction
from utils.load_model import model_loader, LABELS
from utils.audio_splitter import splitter
import torch
import numpy as np
from logzero import logger
from collections import Counter
import operator
from logzero import logger

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

    def predict_one_portion(self, audio, audio_length = 5000):

        features = self.load_features(audio, audio_length)

        features = torch.from_numpy(features).type(torch.double).to(self.device)

        model = model_loader.load_model().to(self.device)

        predicted = torch.argmax(model(features))

        predicted = LABELS[predicted]

        return predicted

    
    def predict(self, audio_path, models = [], audio_length = 5000):

        #y = torchaudio.load(audio_path, normalize = True)

        logger.info(models)

        sp = splitter()
        split = sp.split_audio_tensor(audio_path)

        predicted_list = []

        for i in split:

            predicted = self.predict_one_portion(i, audio_length)
            predicted_list.append(predicted)

        predicted_politician = max(set(predicted_list), key=predicted_list.count)

        key_list = list(Counter(predicted_list).keys())
        value_list = list(Counter(predicted_list).values())

        total = sum(value_list)
        final_dic = {}

        for x in range(len(key_list)):
            final_dic[key_list[x]] = value_list[x]
            
        final_dic = dict( sorted(final_dic.items(), key=operator.itemgetter(1),reverse=True))
        final_string = 'Predicted Politician : '+ str(predicted_politician) + '\n\nConfidence:\n'

        for x in final_dic:
            final_string += str(x) + ' : ' + str(round((final_dic[x]/total)*100,2)) +'%\n'

        logger.info(final_string)
        return final_string
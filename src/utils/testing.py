from utils.audio_dataloader import dataLoader_extraction
from utils.load_model import model_loader, LABELS
from utils.audio_splitter import splitter
import utils.messages as messages
import torch
import numpy as np
from logzero import logger
from collections import Counter
import operator
from logzero import logger
import os

class tester:

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
    
    def final_string_construction(self, predicted_list, expected):

        predicted_politician = max(set(predicted_list), key=predicted_list.count)

        key_list = list(Counter(predicted_list).keys())
        value_list = list(Counter(predicted_list).values())

        total = sum(value_list)
        final_dic = {}

        for x in range(len(key_list)):
            final_dic[key_list[x]] = value_list[x]
            
        final_dic = dict( sorted(final_dic.items(), key=operator.itemgetter(1),reverse=True))
        final_string = ''

        if len(expected) != 0:
            final_string += 'Expected : ' + expected + '\n'
        final_string += 'Predicted Politician : '+ str(predicted_politician) + '\n\nConfidence:\n'

        for x in final_dic:
            final_string += str(x) + ' : ' + str(round((final_dic[x]/total)*100,2)) +'%\n'

        logger.info(final_string)

        return final_string
    
    def predict_one_portion(self, audio, params, audio_length = 5000):

        model, datatype, classifier, Wav2Vec2, plda = params
        ML = model_loader(self.device)
        
        logger.info('Feature extraction starting...')
        features = ML.load_features(audio, audio_length, [datatype, classifier, Wav2Vec2, plda])

        logger.info('Feature consversion to tensor...')
        features = torch.from_numpy(features).type(torch.double).to(self.device)

        logger.info('Model inference starting...')
        model = model.to(self.device)

        predicted = model(features)

        if plda:
            predicted, _ = plda.predict(predicted.cpu().detach().numpy())
            predicted = torch.from_numpy(np.array(predicted))

        predicted = torch.argmax(predicted)

        predicted = LABELS[predicted]

        return predicted

    
    def predict(self, expected, audio_path, models = [], audio_length = 5000):
            
        if len(models) == 0:
            return messages.select_model_error_message

        predicted_list = []

        for x in models:

            logger.info('Performing inference using ' + x)

            logger.info('Initialising model for ' + x + '...')
            ML = model_loader(self.device)

            params = ML.load_model(x)

            sp = splitter()
            split = sp.split_audio_tensor(audio_path)

            for i in split:

                predicted = self.predict_one_portion(i, params, audio_length)
                predicted_list.append(predicted)

        final_string = self.final_string_construction(predicted_list, expected)

        return final_string
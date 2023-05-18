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
    """
    Class is used to bring all other classes together to test audio with models provided.
    """

    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def final_string_construction(self, predicted_list, expected='', models_classifier = None):
        """
        Function used to produce the final string with confidence percentages and predicted speaker.
        Helps to calculate the predicted speaker from the list of predicted speakers based on the model predictions.

        Parameters
        ----------
            predicted_list : List of Strings
                List of speaker names used to be sumed up and find the predicted speaker

            expected : String
                Speaker that is expected to be predicted, can be left as ''

            models_classifier : List of Strings (Default = None)
                List of models used if DNN classifer of multiple models is used.

        Returns
        ----------
            final_string: String
                Final String with all relevant information for display
        """
        final_string = ''
        if models_classifier:
            final_string += 'Models used in this classifier '+str(models_classifier)+'\n\n'

        predicted_politician = max(set(predicted_list), key=predicted_list.count)

        key_list = list(Counter(predicted_list).keys())
        value_list = list(Counter(predicted_list).values())

        total = sum(value_list)
        final_dic = {}

        for x in range(len(key_list)):
            final_dic[key_list[x]] = value_list[x]
            
        final_dic = dict( sorted(final_dic.items(), key=operator.itemgetter(1),reverse=True))
        
        if len(expected) != 0:
            final_string += 'Expected : ' + expected + '\n'
        final_string += 'Predicted Politician : '+ str(predicted_politician) + '\n\nPrediction:\n'

        for x in final_dic:
            final_string += str(x) + ' : ' + str(round((final_dic[x]/total)*100,2)) +'%\n'

        logger.info(final_string)

        return final_string
    
    def predict_one_portion(self, audio, params, audio_length = 5000):
        """
        Function used to predict the speaker for a period of audio given the model in params

        Parameters
        ----------
            audio : Torch.tensors
                Tensors that describe a 5 second audio clip

            params : List of objects
                1. Initialised Pretrained Model 
                2. String of what datatype the model uses or 'Embeddings' for DNN classifier
                3. Classifier used for embedding (None if no external Classifier)
                4. Wav2Vec2 used for embedding (None if no Wav2Vec2 Model used)
                5. plda used for classification (None if no plda used)
                6. models_classifier : List of strings with names of models used for DNN classifier (None if DNN Classifier not used) 

            audio_length : Integar (Default = 5000)
                Length of each audio clip in audio, in ms

        Returns
        ----------
            predicted: String
                String of the predicted label
            
            embeddings: torch.tensor
                Torch tensor of the results from the model given this audio clip
        """
        model, datatype, classifier, Wav2Vec2, plda, models_classifier = params
        model = model.to(self.device)
        ML = model_loader()

        if datatype == 'Embeddings':

            embeddings_list = torch.tensor([]).to(self.device)

            for x in models_classifier:

                params = ML.load_model(x)

                _, embeddings = self.predict_one_portion(audio, params, audio_length = audio_length)

                logger.info('Embedding stored...')
                embeddings_list = torch.cat((embeddings_list, embeddings[0]), 0)
            
            logger.info('Prediction with Embedding...')
            logger.warning(embeddings_list)
            predicted = model(embeddings_list)

        else:

            logger.info('Feature extraction starting...')
            features = ML.load_features(audio, audio_length, [datatype, classifier, Wav2Vec2, plda])

            logger.info('Feature consversion to tensor...')
            features = torch.from_numpy(features).type(torch.double).to(self.device)

            logger.info('Model inference starting...')
            predicted = model(features)
            embeddings = predicted

            if plda:
                predicted, _ = plda.predict(predicted.cpu().detach().numpy())
                predicted = torch.from_numpy(np.array(predicted))

        predicted = torch.argmax(predicted)

        predicted = LABELS[predicted]

        return predicted, embeddings
    
    def predict(self, expected, audio_path, models = [], audio_length = 5000):
        """
        Function used to predict the speaker for a period of audio given the model in params
        This function puts together all the parts for the full inference

        Parameters
        ----------
            expected : String
                String of expected speaker, if no speaker expected, can leave as ''

            audio_path: String
                The path to the audio in .wav format
            
            models : List of Strings
                models that are used to infer

            audio_length : Integar (Default = 5000)
                Length of each audio clip in audio to be split into and inferred, in ms

        Returns
        ----------
            final_string: String
                String of the final prediction and confidence scores
        """

        logger.warning(torch.cuda.is_available())
            
        if len(models) == 0:
            return messages.select_model_error_message

        predicted_list = []

        for x in models:

            logger.info('Performing inference using ' + x)

            logger.info('Initialising model for ' + x + '...')
            ML = model_loader()

            params = ML.load_model(x)

            sp = splitter()
            split = sp.split_audio_tensor(audio_path)

            for i in split:

                predicted, _ = self.predict_one_portion(i, params, audio_length)
                predicted_list.append(predicted)

        final_string = self.final_string_construction(predicted_list, expected, params[-1])

        return final_string
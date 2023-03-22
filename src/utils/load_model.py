import os
from utils.model import ECAPA_TDNN

class model_loader():

    def load_model():

        labels = os.environ.get('LABELS')

        model = ECAPA_TDNN(157 ,512, len(labels))

        return model

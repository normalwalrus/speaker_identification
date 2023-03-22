import os
import torch
from dotenv import load_dotenv
from utils.model import ECAPA_TDNN

LABELS = ['Lee Hsien Loong', 'Amy Khor Lean Suan', 'Leong Mun Wai', 'Indranee Rajah', 
            'Ong Ye Kung', 'K Shanmugam', 'Tan See Leng', 'Teo Chee Hean', 'Vivian Balakrishnan', 
            'Lawrence Wong', 'Sylvia Lim', 'He Ting Ru', 'Rahayu Mahzam', 'Tan Chuan-Jin', 
            'Desmond Lee', 'Josephine Teo', 'Janil Puthucheary', 'Gan Kim Yong', 'Murali Pillai', 'Tin Pei Ling']

load_dotenv()

class model_loader:

    def load_model():

        labels = LABELS
        no_speakers = len(labels)
        save_path = os.getcwd() + os.environ.get('PATH_TO_MODEL')

        model = ECAPA_TDNN(157 ,512, no_speakers)
        
        model.load_state_dict(torch.load(save_path))
        model.eval().double()

        return model

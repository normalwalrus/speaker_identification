import gradio as gr
from utils.testing import tester
from typing import Any, List, Optional, Union
import utils.messages as messages

MODELS = ['ECAPA_TDNN', 'ECAPA_TDNN_pretrained', 'Wav2Vec2_Embedding', 'CNN', 
          'ResNet34', 'ResNet50', 'ResNet101', 'XvecTDNN', 'XvecPLDA', 
          'ResNet_DNN_classifier', 'All_DNN_classifier', 'ECAPA_Wav2_DNN_classifier']

AUDIO_LIST = [['Lee Hsien Loong', 'audio/LeeHsienLoong.wav', []],
              ['Josephine Teo', 'audio/JosephineTeo.wav', []],
              ['Sylvia Lim', 'audio/SylviaLim.wav', []]]

inputs = [gr.Textbox(label= 'Expected'), gr.Audio(source='upload', type='filepath', label = 'Audio'),
        gr.CheckboxGroup(choices= MODELS, label = 'Models')]

outputs = ['text']

if __name__ == "__main__":

    tester_app = tester()

    app = gr.Interface(
        tester_app.predict,
        inputs=inputs,
        outputs=outputs,
        title=messages.title,
        description=messages.desciption,
        examples=AUDIO_LIST
    ).launch(server_name="0.0.0.0")
    
    
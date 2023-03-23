import gradio as gr
from utils.testing import tester
from typing import Any, List, Optional, Union
import utils.messages as messages

MODELS = ['ECAPA_TDNN', 'ECAPA_TDNN_pretrained', 'Wav2Vec2_Embedding', 'CNN', 
          'ResNet34', 'ResNet50', 'ResNet101', 'XvecTDNN', 'XvecPLDA']

AUDIO_LIST = [['Lee Hsien Loong', 'audio/LeeHsienLoong.wav', []],
              ['Josephine Teo', 'audio/JosephineTeo.wav', []],
              ['Tan Chuan-Jin', 'audio/TanChuanJin.wav', []]]

inputs = [gr.Textbox(), gr.Audio(source='upload', type='filepath'), gr.CheckboxGroup(choices= MODELS)]
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
    ).launch()
    
    
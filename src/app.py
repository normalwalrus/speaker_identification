import gradio as gr
from utils.testing import tester
from typing import Any, List, Optional, Union
import utils.messages as messages

MODELS = ['ECAPA_TDNN', 'ECAPA_TDNN_pretrained', 'Wav2Vec2_Embedding', 'CNN']

inputs = [gr.Audio(source='upload', type='filepath'), gr.CheckboxGroup(choices= MODELS)]
outputs = ['text']

if __name__ == "__main__":

    tester_app = tester()

    app = gr.Interface(
        tester_app.predict,
        inputs=inputs,
        outputs=outputs,
        title=messages.title,
        description=messages.desciption,
    ).launch()
    
    
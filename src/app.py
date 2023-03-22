import gradio as gr
from utils.testing import tester
from typing import Any, List, Optional, Union

inputs = gr.Audio(source='upload', type='filepath')
# inputs: Union[str, gr.inputs.Audio] = gr.inputs.Audio(source='upload', type='filepath')
outputs = ['text']

if __name__ == "__main__":

    tester_app = tester()

    app = gr.Interface(
        tester_app.predict,
        inputs=inputs,
        outputs=outputs,
        title="Speech Recognition ECAPA_TDNN",
        description="Please input an audio clip of your favourite Singapore Politician",
    ).launch()
    
    
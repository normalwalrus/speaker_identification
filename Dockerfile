# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.1.0-runtime-ubuntu18.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

ENV PATH_TO_ECAPA_TDNN="/models/ECAPA_TDNN_v1.0_5sec_80MFCC_30epoch.pt"
ENV PATH_TO_ECAPA_TDNN_pretrained="/models/ECAPA_TDNN_Pretrained_v1.0_5sec_10epoch.pt"
ENV PATH_TO_CNN_Wav2Vec2="/models/CNN_Wav2Vec2_v1.0_5sec_110epoch.pt"
ENV PATH_TO_CNN="/models/CNN_v1.1.pt"
ENV PATH_TO_ResNet34="/models/ResNet34_v1.0.pt"
ENV PATH_TO_ResNet50="/models/ResNet50_v1.1.pt"
ENV PATH_TO_ResNet101="/models/ResNet101_v1.1.pt"
ENV PATH_TO_XvecTDNN="/models/xvecTDNN_v1.0_5sec_80MFCC.pt"
ENV PATH_TO_XvecPLDA="/models/XvecTDNN_vector_extraction.pt"
ENV PATH_TO_PLDA="/models/plda.pkl"

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /src/app.py
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "src/app.py"]

FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
LABEL AUTHOR=khiemtv
RUN nvcc --version
RUN apt-get update -y && apt-get upgrade -y
# RUN apt-get install -y build-essential
# RUN apt-get install -y cmake wget nano xterm git-core unzip pkg-config
# COPY requirements.txt .
# RUN pip3 install --upgrade pip
RUN python -V
RUN pip install --upgrade setuptools pip
RUN pip install nvidia-pyindex
RUN pip install --upgrade nvidia-tensorrt
RUN nvcc -V
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python
RUN pip install faiss-cpu albumentations tqdm requests uvicorn fastapi python-multipart
RUN pip uninstall opencv-python-headless -y
RUN pip install --force-reinstall opencv-python
# RUN apt-get install ffmpeg
RUN apt-get update -y && apt-get upgrade -y
# RUN apt-get install ffmpeg libsm6 -y
RUN apt-get install qt5-default -y
RUN apt-get install net-tools -y

RUN mkdir -p /root/.cache/torch/checkpoints && mkdir -p /root/.cache/torch/hub/checkpoints
COPY model/Resnet50_Final.pth /root/.cache/torch/checkpoints/
COPY model/mobilenet0.25_Final.pth /root/.cache/torch/checkpoints/
COPY model/attr_resnet18_jit_best.pt /root/.cache/torch/checkpoints/
COPY model/attr_mbnet2_jit_best.pt /root/.cache/torch/checkpoints/
COPY model/iresnet34-5b0d0e90.pth /root/.cache/torch/hub/checkpoints/
COPY model/iresnet50-7f187506.pth /root/.cache/torch/hub/checkpoints/
COPY model/iresnet100-73e07ba7.pth /root/.cache/torch/hub/checkpoints/

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
ENV TZ=Asia/Ho_Chi_Minh
RUN echo $TZ > /etc/timezone
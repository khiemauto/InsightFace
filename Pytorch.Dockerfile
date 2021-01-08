FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
LABEL AUTHOR=khiemtv
# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN nvcc --version
RUN apt-get update -y && apt-get upgrade -y
# RUN apt-get install -y build-essential
# RUN apt-get install -y cmake wget nano xterm git-core unzip pkg-config
# COPY requirements.txt .
# RUN pip3 install --upgrade pip
# RUN python3 -V
# RUN pip3 install torch torchvision 
RUN pip install faiss-cpu albumentations tqdm requests uvicorn fastapi
RUN mkdir -p /root/.cache/torch/checkpoints && mkdir -p /root/.cache/torch/hub/checkpoints
COPY model/Resnet50_Final.pth /root/.cache/torch/checkpoints/
COPY model/iresnet100-73e07ba7.pth /root/.cache/torch/hub/checkpoints/
RUN pip install python-multipart
RUN pip uninstall opencv-python-headless -y
RUN pip install --force-reinstall opencv-python
# RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install qt5-default -y
COPY model/attr_resnet18_jit_best.pt /root/.cache/torch/checkpoints/
RUN apt-get install net-tools -y

COPY model/mobilenet0.25_Final.pth /root/.cache/torch/checkpoints/
COPY model/iresnet34-5b0d0e90.pth /root/.cache/torch/hub/checkpoints/
COPY model/iresnet50-7f187506.pth /root/.cache/torch/hub/checkpoints/
COPY model/attr_mbnet2_jit_best.pt /root/.cache/torch/checkpoints/
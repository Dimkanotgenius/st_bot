FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
#FROM nvcr.io/nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    tmux \
    curl \
    wget \
    build-essential \
    sudo \
    python3-pip \
    # python-is-python3 \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
#RUN python3 -m pip install --upgrade pip
#RUN python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install -r requirements.txt


# add user and his password
ARG USER=docker_beit
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN mkdir -p beit && mkdir Datasets && chown -R ${UID}:${GID} /home/${USER}
USER ${UID}:${GID}
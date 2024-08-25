FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing -y 

RUN apt-get install -y python3 python3-pip git

RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install lightning

# some standard packages
RUN pip3 install tqdm matplotlib pillow pandas argparse scikit-learn scikit-image numpy opencv-python-headless tensorboardX tabulate colorama 

# additional packages
RUN pip3 install h5py scipy pyyaml wandb pydicom nibabel seaborn tqdm 
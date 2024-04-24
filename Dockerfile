FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install the torchrun package.
RUN pip install torchrun lightning

# Copy your training script to use with torchrun.
COPY ./train.py ./
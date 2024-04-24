FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install lightning

# Copy your training script to use with torchrun.
COPY ./train.py ./
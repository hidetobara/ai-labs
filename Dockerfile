FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
	apt-get install -y curl python3-pip git vim less wget libgl1-mesa-dev libglib2.0-0 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install Pillow transformers diffusers accelerate opencv-python torchsde

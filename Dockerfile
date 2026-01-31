FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
	apt-get install -y curl python3-pip git vim less wget libgl1-mesa-dev libglib2.0-0 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install Pillow transformers accelerate opencv-python torchsde gradio \
	sentencepiece protobuf spacy
RUN pip3 install git+https://github.com/huggingface/diffusers
RUN python3 -m spacy download en_core_web_sm

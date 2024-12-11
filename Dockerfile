# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python build dependencies, Rust, and system tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    git \
    curl \
    && apt-get clean

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="$HOME/.cargo/bin:$PATH" \
    && rustc --version

# Install Python 3.10.4 from source
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install python3.10

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install --upgrade pip setuptools wheel gradio

RUN python3.10 -m pip install datasets scikit-learn seaborn matplotlib spaces fastapi uvicorn pypdf2

# Set the working directory
WORKDIR /app

COPY . /app

# Install the Binoculars package in editable mode
RUN python3.10 -m pip install -e .

# CMD ["python3.10", "app.py"]

EXPOSE 7860
EXPOSE 8080

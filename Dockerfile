# Start with Python 3.10.4 base image
FROM python:3.10.4-slim

ENV PORT=7860

# For HuggingFace Dev Mode
RUN apt-get update && apt-get install -y \
 build-essential \
 wget \
 git \
 wget \
 curl \
 procps \
 lsof \
 nano \
 && apt-get clean

# HuggingFace
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set up the working directory
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

# Copy your application files
COPY --chown=user . $HOME/app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose ports for Gradio and FastAPI
EXPOSE $PORT
EXPOSE 8080

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860" ]
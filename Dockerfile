FROM nvcr.io/nvidia/tensorrt:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    ffmpeg libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx \
    git wget unzip \
    && apt-get clean


# Install Python packages
RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt


# Set working directory
WORKDIR /app
COPY . /app

# Test libraries
# CMD find / -name "libnvinfer_plugin.so*" 2>/dev/null && \
#     find / -name "libnvinfer.so*" 2>/dev/null && \
#     python3 main.py

ENTRYPOINT ["python3", "main.py"]
CMD []

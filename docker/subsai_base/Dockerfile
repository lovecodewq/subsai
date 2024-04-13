FROM python:3.8-slim 

WORKDIR /subsai

COPY requirements.txt .

# RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing && \
#     libmp3lame0 libopus0 libvorbis0a git gcc mono-mcs && apt-get upgrade -y && \
#     pip install --no-cache-dir -r requirements.txt && pip install . && apt-get clean && rm -rf /var/lib/apt/lists/* \
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc make nasm yasm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/abdeladim-s/subsai
RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    make \
    nasm \
    yasm \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libvorbis-dev \
    fonts-wqy-zenhei \
    libvorbisenc2 \ 
    build-essential \
    libfreetype6-dev \
    libvpx5 \
    libass-dev \
    ca-certificates \
    pkg-config && \
    git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git . && \
    ./configure \
    --enable-gpl --enable-libass --enable-libfreetype \
    --enable-libmp3lame --enable-libopus --enable-libvorbis \
    --enable-libx264 --enable-libx265 --enable-nonfree --disable-debug --disable-doc \
    --enable-libvpx --enable-libfdk-aac &&\
    make -j$(nproc) && \
    make install && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./src ./src
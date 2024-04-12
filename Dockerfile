FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /subsai

COPY requirements.txt .

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y --no-install-recommends --fix-missing \
    libmp3lame0 libopus0 libvorbis0a git gcc mono-mcs
RUN apt-get upgrade -y

RUN pip install --no-cache-dir -r requirements.txt 

COPY pyproject.toml .
COPY ./src ./src
COPY ./assets ./assets

RUN pip install .
RUN pip install python-dotenv
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-wqy-zenhei libvorbisenc2 \
    build-essential pkg-config git nasm yasm \
    libass-dev libfreetype6-dev libmp3lame-dev libopus-dev \
    libvorbis-dev libx264-dev libx265-dev ca-certificates

RUN  git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git /FFmpeg && \
    cd /FFmpeg && \
    ./configure \
      --enable-gpl --enable-libass --enable-libfreetype \
      --enable-libmp3lame --enable-libopus --enable-libvorbis \
      --enable-libx264 --enable-libx265 --enable-nonfree --disable-debug --disable-doc && \
    make -j$(nproc) && make install && \
    make distclean
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# ENTRYPOINT or CMD as needed
# ENTRYPOINT ["python", "src/subsai/webui.py", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]

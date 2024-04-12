# Use Debian Buster as the base image
FROM debian:buster as ffmpeg-test

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config git nasm yasm \
    libass-dev libfreetype6-dev libmp3lame-dev libopus-dev \
    libvorbis-dev libx264-dev libx265-dev \
    ca-certificates wget

RUN apt-get update && apt-get install -y --no-install-recommends wget make libass-dev

RUN wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2 && \
    tar xvf nasm-2.15.05.tar.bz2 && \
    cd nasm-2.15.05 && \
    ./autogen.sh && \
    ./configure && \
    make -j$(nproc) && \
    make install

# Clone and build FFmpeg with subtitle support
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git /FFmpeg && \
    cd /FFmpeg && \
    ./configure \
      --enable-gpl --enable-libass --enable-libfreetype \
      --enable-libmp3lame --enable-libopus --enable-libvorbis \
      --enable-libx264 --enable-libx265 --enable-nonfree --disable-debug --disable-doc && \
    make -j$(nproc) && make install && \
    make distclean

# After building FFmpeg, check for subtitle filters
RUN ffmpeg -filters | grep subtitles

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import subprocess
from typing import List
import os
import pysubs2
import time

from pathlib import Path

from functools import wraps

Ffmpeg_bin="/usr/local/bin/ffmpeg"

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to complete.")
        return result
    return wrapper

@timeit
def add_subtitles_to_video(video_path, subtitles_file_path, output_video_path):
    """
    Adds an ASS subtitle file to an MP4 video file using FFmpeg, keeping the subtitles as a separate track.

    Args:
    video_path (str): Path to the input MP4 video file.
    subtitles_path (str): Path to the ASS subtitle file.
    output_video_path (str): Path for the output MP4 video file with added subtitles.
    """
    command = [
        Ffmpeg_bin,
        '-i', str(video_path),  # Input video file
        # Filter to burn subtitles
        '-vf', f"subtitles='{str(subtitles_file_path)}'",
        '-c:v', 'libx264',  # Specify video codec as H.264
        '-crf', '20',
        '-c:a', 'copy',                  # Copy the audio without re-encoding
        '-max_muxing_queue_size', '1024',
        str(output_video_path)  # Output video file
    ]

    try:
        # Execute the FFmpeg command
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video with subtitles added created at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    parser.add_argument('-s', '--subtitles', default=None, type=str,
                        help="subtitles file")
    parser.add_argument('-v', '--video', default=None, type=str,
                        help="video file")
    args = parser.parse_args()
    subtitiles_file_path=Path(args.subtitles)
    video_file_path = Path(args.video)
    output_video = video_file_path.parent / \
                    (subtitiles_file_path.stem + ".mp4")
    add_subtitles_to_video(video_file_path, subtitiles_file_path, output_video)


if __name__ == '__main__':
    main()

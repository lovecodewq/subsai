#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SubsAI Command Line Interface (cli)
"""
import argparse
import importlib.metadata
__license__ = "GPLv3"
__version__ = importlib.metadata.version('subsai')
import subprocess
import math
from typing import List
import os
import json
import torch
import pysubs2
import pickle
import time

from pathlib import Path
from subsai import SubsAI
from subsai import SubsAI, Tools
from subsai.utils import available_translation_models, available_subs_formats
from functools import wraps


subs_ai = SubsAI()
tools = Tools()


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to complete.")
        return result
    return wrapper


def _handle_media_file(media_file_arg: List[str]):
    res = []
    for file in media_file_arg:
        if file.endswith('.txt'):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line == '':
                        continue
                    res.append(Path(line.strip()).resolve())
        else:
            res.append(Path(file).resolve())
    return res


def _handle_configs(model_configs_arg: str):
    if model_configs_arg.endswith('.json'):
        with open(model_configs_arg, 'r') as file:
            data = file.read()
            return json.loads(data)
    return json.loads(model_configs_arg)


@timeit
def extract_audio(input_video_path, output_audio_path):
    """
    Extracts audio from an input video file and saves it as a mp3 file.

    Args:
    input_video_path (str): The path to the input MP4 video file.
    output_audio_path (str): The path where the output mp3 audio file will be saved.
    """
    # Command to extract audio using FFmpeg
    command = [
        'ffmpeg',
        '-i', input_video_path,  # Input video file
        '-vn',  # Exclude video
        '-acodec', 'libmp3lame',  # MP3 audio codec
        '-q:a', '2',
        output_audio_path  # Output audio file
    ]

    try:
        # Execute the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Audio extracted and saved to {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


@timeit
def split_video(video_path, segment_duration):
    folder = video_path.parent
    # Use FFmpeg to get the total duration of the video in seconds
    cmd = ['ffprobe', '-v', 'error', '-show_entries',
           'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
           video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    total_duration = float(result.stdout)

    # Calculate the number of segments
    num_segments = math.ceil(total_duration / segment_duration)
    video_split_paths = []
    for segment in range(num_segments):
        start_time = segment * segment_duration
        file_path = folder / (video_path.stem +
                              "_segment_" + str(segment + 1) + ".mp4")
        video_split_paths.append(file_path)
        # FFmpeg command to split the video
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c', 'copy',
            file_path
        ]
        subprocess.run(cmd)
        print(f'Segment {segment + 1} created.')
        return video_split_paths
    print('All segments created.')
    return video_split_paths


@timeit
def merge_subtitles_into_video(video_path, subtitle_path_english, subtitle_path_chinese, output_video_path):
    """
    Embeds two subtitle files (English and Chinese) into a video file using FFmpeg.

    Args:
    video_path (str): Path to the input video file.
    subtitle_path_english (str): Path to the English subtitle file.
    subtitle_path_chinese (str): Path to the Chinese subtitle file.
    output_video_path (str): Path for the output video file with embedded subtitles.
    """
    command = [
        'ffmpeg',
        '-i', video_path,  # Input video
        '-i', subtitle_path_english,  # English subtitle
        '-i', subtitle_path_chinese,  # Chinese subtitle
        '-map', '0:v',  # Map video stream
        '-map', '0:a',  # Map audio stream
        '-map', '1',  # Map English subtitle stream
        '-map', '2',  # Map Chinese subtitle stream
        '-c', 'copy',  # Copy video and audio streams
        '-c:s', 'mov_text',  # Use mov_text for subtitles (compatible with MP4)
        '-metadata:s:s:0', 'language=eng',  # Set English subtitle metadata
        '-metadata:s:s:1', 'language=chi',  # Set Chinese subtitle metadata
        output_video_path  # Output video file
    ]

    try:
        # Execute the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Video with subtitles created at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


@timeit
def translation(subs,  model_name, configs, source_lang, target_lang):

    print('Done translation!')
    return subs


def run(media_file_arg: List[str],
        model_name,
        model_configs,
        destination_folder,
        subs_format,
        translation_model_name,
        translation_configs,
        source_lang,
        target_lang,
        output_suffix
        ):
    model_configs = _handle_configs(model_configs)
    print(f"[-] Model name: {model_name}")
    print(
        f"[-] Model configs: {'defaults' if model_configs == {} else model_configs}")
    print(f"---")
    print(f"[+] Initializing the model")
    model = subs_ai.create_model(model_name, model_configs)

    print(f"[+] Creating translation model: {translation_model_name}")
    tr_model = tools.create_translation_model(translation_model_name)
    print(
        f"[+] Translating from: {source_lang} to {target_lang}")
    tr_configs = _handle_configs(translation_configs)

    files = _handle_media_file(media_file_arg)
    split_duration = 60  # second
    for file in files:
        print(f"[+] Processing file: {file}")
        if not file.exists():
            print(f"[*] Error: {file} does not exist -> continue")
            continue
        # split video file
        video_split_paths = split_video(file, split_duration)
        for path in video_split_paths:
            folder = path.parent
            output_audio_file_path = folder / (file.stem + ".mp3")
            # extract audio
            extract_audio(path, output_audio_file_path)
            subs = subs_ai.transcribe(output_audio_file_path, model)
            source_subs_file_path = folder / \
                (file.stem + "_"+source_lang + "."+subs_format)
            subs.save(source_subs_file_path)

            # translate
            translated_subs = tools.translate(subs=subs,
                                              source_language=source_lang,
                                              target_language=target_lang,
                                              model=tr_model,
                                              translation_configs=tr_configs)
            target_subs_file_path = folder / \
                (file.stem + "_"+target_lang + "."+subs_format+"srt")

            translated_subs.save(target_subs_file_path)
            # merge subs into vido file
            output_video = folder / \
                (file.stem + "_" + source_lang + "_" + "target_lang" + ".mp4")
            merge_subtitles_into_video(
                path, source_subs_file_path, target_subs_file_path, output_video)


def main():
    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('media_file', type=str, nargs='+', help="The path of the media file, a list of files, or a "
                                                                "text file containing paths for batch processing.")

    # Optional args
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument('-m', '--model', default=SubsAI.available_models()[0],
                        help=f'The transcription AI models. Available models: {SubsAI.available_models()}')
    parser.add_argument('-mc', '--model-configs', default="{}",
                        help="JSON configuration (path to a json file or a direct "
                             "string)")
    parser.add_argument('-f', '--format', '--subtitles-format', default='srt',
                        help=f"Output subtitles format, available "
                             f"formats {available_subs_formats(include_extensions=False)}")
    parser.add_argument('-df', '--destination-folder', default=None,
                        help='The directory where the subtitles will be stored, default to the same folder where '
                             'the media file(s) is stored.')
    parser.add_argument('-tm', '--translation-model-name', default=None,
                        help=f"Translate subtitles using AI models, available "
                             f"models: {available_translation_models()}", )
    parser.add_argument('-tsl', '--source-lang',
                        default=None, help="Source language of the subtitles")
    parser.add_argument('-ttl', '--target-lang',
                        default=None, help="Target language of the subtitles")
    parser.add_argument('-tc', '--translation-configs', default="{}",
                        help="JSON configuration (path to a json file or a direct "
                             "string)")
    parser.add_argument('-os', '--output-suffix', default=None,
                        help="Name of the subtitles output file, (In batch processing, this will be used as a suffix to the media filename)")

    args = parser.parse_args()

    run(media_file_arg=args.media_file,
        model_name=args.model,
        model_configs=args.model_configs,
        destination_folder=args.destination_folder,
        subs_format=args.format,
        translation_model_name=args.translation_model_name,
        translation_configs=args.translation_configs,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        output_suffix=args.output_suffix)


if __name__ == '__main__':
    main()

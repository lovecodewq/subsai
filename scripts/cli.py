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
    print(f"[+] Extract audio to {output_audio_path}..")

    command = [
        Ffmpeg_bin,
        '-i', input_video_path,  # Input video file
        '-vn',  # Exclude video
        '-acodec', 'libmp3lame',  # MP3 audio codec
        '-q:a', '2',
        output_audio_path  # Output audio file
    ]
    try:
        # Execute the FFmpeg command
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video with subtitles added created at {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}\nOutput:\n{e.stdout}\nErrors:\n{e.stderr}")


@timeit
def split_video(video_path, segment_duration, destination_folder):
    # Use FFmpeg to get the total duration of the video in seconds
    command = ['ffprobe', '-v', 'error', '-show_entries',
           'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
           video_path]
    try:
        # Execute the FFmpeg command
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}\nOutput:\n{e.stdout}\nErrors:\n{e.stderr}")
    total_duration = float(result.stdout)

    # Calculate the number of segments
    num_segments = math.ceil(total_duration / segment_duration)
    video_split_paths = []
    for segment in range(num_segments):
        start_time = segment * segment_duration
        destination_folder = destination_folder / ("part_"+str(segment))
        if not destination_folder.exists():
            print(f"[+] Creating folder: {destination_folder}")
            os.makedirs(destination_folder, exist_ok=True)
        file_path = destination_folder / (video_path.stem +
                                          "_p_" + str(segment) + ".mp4")
        video_split_paths.append(file_path)
        # FFmpeg command to split the video
        command = [
            Ffmpeg_bin, '-y', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c', 'copy',
            file_path
        ]
        try:
            result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}\nOutput:\n{e.stdout}\nErrors:\n{e.stderr}")
        print(f'Segment {segment + 1} created.')
        if len(video_split_paths) == 1:
            return video_split_paths
    print('All segments created.')
    return video_split_paths


@timeit
def add_subtitles_to_video(video_path, subtitles_path, output_video_path):
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
        '-vf', f"subtitles='{str(subtitles_path)}'",
        '-c:v', 'libx264',  # Specify video codec as H.264
        '-crf', '20',
        '-c:a', 'copy',                  # Copy the audio without re-encoding
        '-max_muxing_queue_size', '1024',
        str(output_video_path)  # Output video file
    ]

    try:
        # Execute the FFmpeg command
        result = subprocess.run(command, check=True)
        print(f"Video with subtitles added created at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


@timeit
def translation(subs,  model_name, configs, source_lang, target_lang):

    print('Done translation!')
    return subs


def merge_subtitles_file(english_file, chinese_file):
    # Load the subtitle files
    english_subs = pysubs2.load(english_file)
    chinese_subs = pysubs2.load(chinese_file)
    return merge_subtitles(english_subs, chinese_subs)


def merge_subtitles(english_subs, chinese_subs):
    # for line in english_subs:
    #     line.MarginV = 10
    #     line.fontsize = 10
    # for line in chinese_subs:
    #     line.MarginV = 0
    #     line.fontsize = 10

    # Merge Chinese subtitles into English subtitles
    english_subs.events.extend(chinese_subs.events)

    # Sort the merged events by their start time
    english_subs.events.sort(key=lambda x: x.start)

    return english_subs


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

    print(f"-----------------------------------------------------------")
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
        if not destination_folder:
            destination_folder = file.parent / (file.stem + "_processed")
        else:
            destination_folder = Path(destination_folder).absolute()
        if not destination_folder.exists():
            print(f"[+] Creating folder: {destination_folder}")
            os.makedirs(destination_folder, exist_ok=True)
        video_split_paths = split_video(
            file, split_duration, destination_folder)
        for part_id in range(len(video_split_paths)):
            path = video_split_paths[part_id]
            file_name = path.stem
            destination_folder = destination_folder / ("part_"+str(part_id))
            if not destination_folder.exists():
                print(f"[+] Creating folder: {destination_folder}")
                os.makedirs(destination_folder, exist_ok=True)
            output_audio_file_path = destination_folder / (file_name + ".mp3")
            # extract audio
            print(f"-----------------------------------------------------------")
            print(f"[+] Etract video ..")
            extract_audio(path, output_audio_file_path)

            print(f"-----------------------------------------------------------")
            print(f"[+] Transcribe ..")
            subtitle_folder = destination_folder
            if not subtitle_folder.exists():
                print(f"[+] Creating folder: {subtitle_folder}")
                os.makedirs(subtitle_folder, exist_ok=True)
            source_subs_file_path = subtitle_folder / \
                (file_name + "_" + source_lang[:2].lower() + "."+subs_format)
            subs = subs_ai.transcribe(output_audio_file_path, model)
            subs.save(source_subs_file_path)

            # translate
            print(f"-----------------------------------------------------------")
            print(f"[+] Translate subtitles..")
            translated_subs_file_path = subtitle_folder / \
                (file_name + "_"+target_lang[:2].lower() + "."+subs_format)
            translated_subs = tools.translate(subs=subs,
                                              source_language=source_lang,
                                              target_language=target_lang,
                                              model=tr_model,
                                              translation_configs=tr_configs)
            # translated_subs.save(translated_subs_file_path)

            print(f"-----------------------------------------------------------")
            print(f"[+] Merge subtitile files..")
            # merged_subs = merge_subtitles_file(
            #     source_subs_file_path, translated_subs_file_path)
            merged_subs = merge_subtitles(subs, translated_subs)
            merge_subs_path = subtitle_folder / \
                (file_name + "_" + source_lang[:2].lower() +
                 "_" + target_lang[:2].lower() + "." + subs_format)
            merged_subs.save(merge_subs_path)
            # merge subs into vido file
            print(f"-----------------------------------------------------------")
            print(f"[+] Merge into video..")
            output_video = destination_folder.parent / \
                (file_name + "_" +
                 source_lang[:2].lower() + "_" + target_lang[:2].lower() + ".mp4")
            add_subtitles_to_video(path, merge_subs_path, output_video)
            print(f"-----------------------------------------------------------")
            print(f"Subtitle file path: {merge_subs_path}")
            print(f"Done!")


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

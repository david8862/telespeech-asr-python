#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to convert excel format annotation to one txt per person.
Reference from:
https://github.com/wiseman/py-webrtcvad

py-webrtcvad could be installed with following cmd:
pip install webrtcvad
"""
import os, sys, argparse
from tqdm import tqdm
#import numpy as np
#import pandas as pd
import soundfile as sf
import webrtcvad
import wave

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    wf = wave.open(path, 'rb')
    num_channels = wf.getnchannels()
    #assert num_channels == 1
    sample_width = wf.getsampwidth()
    #assert sample_width == 2
    sample_rate = wf.getframerate()
    assert sample_rate in (8000, 16000, 32000, 48000)
    pcm_data = wf.readframes(wf.getnframes())
    wf.close()

    return pcm_data, sample_rate



def vad_check(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_path):
        audio_list = [input_path]
    else:
        audio_list = glob.glob(os.path.join(input_path, '*.wav'))

    vad = webrtcvad.Vad()
    vad.set_mode(1)
    frame_duration = 10  # ms

    # Run the VAD on 10 ms of silence. The result should be False.
    #sample_rate = 16000
    #frame_duration = 10  # ms
    #frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
    #print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))

    pbar = tqdm(total=len(audio_list), desc='VAD check')
    for audio_file in audio_list:
        #audio_data, sample_rate = sf.read(audio_file)
        audio_data, sample_rate = read_wave(audio_file)

        print('type(audio_data):', type(audio_data))
        #print('audio_data.shape:', audio_data.shape)
        #print('audio_data.dtype:', audio_data.dtype)
        print('sample_rate:', sample_rate)

        frame_len = int(sample_rate * frame_duration / 1000)
        frame = audio_data[:frame_len]
        is_speech = vad.is_speech(frame, sample_rate)
        print('is_speech:', is_speech)

        pbar.update(1)
    pbar.close()



def main():
    parser = argparse.ArgumentParser(description='tool to convert excel format annotation to txt')
    parser.add_argument('--input_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save voice duration info in txt files. default=%(default)s')
    #parser.add_argument('--content_only', default=False, action="store_true",
                        #help='only convert speech content and igore other info')
    args = parser.parse_args()

    vad_check(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

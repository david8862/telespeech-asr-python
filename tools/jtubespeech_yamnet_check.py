#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" tool to check if audio segment file from jtubespeech  is speech, using Tensorflow YAMNet sound classification model
Reference from:
https://www.tensorflow.org/hub/tutorials/yamnet?hl=zh-cn
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import shutil
from scipy.io import wavfile
import csv

import tensorflow as tf
import tensorflow_hub as hub


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


def jtubespeech_yamnet_check(wav_16k_path, subtitle_path, output_path):
    # check input path
    assert (os.path.isdir(wav_16k_path)), 'wav_16k_path is not directory'
    assert (os.path.isdir(subtitle_path)), 'subtitle_path is not directory'

    # get .wav audio file list or single .wav audio file
    if os.path.isfile(wav_16k_path):
        wav_list = [wav_16k_path]
    else:
        wav_list = glob.glob(os.path.join(wav_16k_path, '*.wav'))

    # Load YAMNet model
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    # prepare picked wav audio segments path & subtitle text file
    output_audio_path = os.path.join(output_path, 'wav_segments')
    output_subtitle_path = os.path.join(output_path, 'subtitles')
    os.makedirs(output_audio_path, exist_ok=True)
    os.makedirs(output_subtitle_path, exist_ok=True)

    pbar = tqdm(total=len(wav_list), desc='Whisper ASR inference')
    for wav_file in wav_list:
        sample_rate, wav_data = wavfile.read(wav_file, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

        # Show some basic information about the audio.
        #duration = len(wav_data)/sample_rate
        #print(f'Sample rate: {sample_rate} Hz')
        #print(f'Total duration: {duration:.2f}s')
        #print(f'Size of the input: {len(wav_data)}')

        # 需要将 wav_data 归一化为 [-1.0, 1.0] 中的值（如模型文档中所述）。
        waveform = wav_data / tf.int16.max

        # Run the model, check the output.
        scores, embeddings, spectrogram = model(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        audio_type = class_names[scores_np.mean(axis=0).argmax()]
        #print(f'The audio type is: {audio_type}')

        # pick 'Speech' type audio segments and copy to target path
        if audio_type == 'Speech':
            wav_file_basename = os.path.splitext(os.path.split(wav_file)[-1])[0]
            target_wav_file = os.path.join(output_audio_path, wav_file_basename + '.wav')
            shutil.copyfile(wav_file, target_wav_file)

            source_subtitle_file = os.path.join(subtitle_path, wav_file_basename + '.txt')
            target_subtitle_file = os.path.join(output_subtitle_path, wav_file_basename + '.txt')
            shutil.copyfile(source_subtitle_file, target_subtitle_file)

        pbar.update(1)
    pbar.close()
    return


def main():
    parser = argparse.ArgumentParser(description='Tool to check if audio segment files from jtubespeech is speech with YAMNet model')
    parser.add_argument('--wav_16k_path', type=str, required=True,
                        help='directory for jtubespeech 16k .wav audio files')
    parser.add_argument('--subtitle_path', type=str, required=True,
                        help='directory for jtubespeech .txt subtitle files')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save split audio segment files. default=%(default)s')
    args = parser.parse_args()

    jtubespeech_yamnet_check(args.wav_16k_path, args.subtitle_path, args.output_path)


if __name__ == "__main__":
    main()

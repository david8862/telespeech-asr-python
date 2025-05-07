#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to convert excel format annotation to one txt per person.
Reference from:
https://github.com/wiseman/py-webrtcvad
https://github.com/wiseman/py-webrtcvad/blob/master/example.py

py-webrtcvad could be installed with following cmd:
pip install webrtcvad
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import wave
import webrtcvad


def read_wav(wav_file):
    """
    Reads wav file and check if it matches webrtcvad's requirement.
    return (PCM audio data, sample rate) if success.
    """
    wf = wave.open(wav_file, 'rb')
    # check channel number
    num_channels = wf.getnchannels()
    assert (num_channels == 1), 'unsupported audio channel number: %d' % num_channels

    # check sample width
    sample_width = wf.getsampwidth()
    assert (sample_width == 2), 'unsupported audio sample width: %d' % sample_width

    # check sample rate
    sample_rate = wf.getframerate()
    assert (sample_rate in (8000, 16000, 32000, 48000)), 'unsupported audio sample rate: %d' % sample_rate

    # read audio data
    audio_data = wf.readframes(wf.getnframes())
    wf.close()

    return audio_data, sample_rate


def get_frames(audio, sample_rate, sample_width, frame_duration_ms):
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * sample_width)

    # cut the audio sequence into a list of frames for VAD check
    offset = 0
    frame_list = []
    while offset + frame_size < len(audio):
        frame_list.append(audio[offset:offset+frame_size])
        offset += frame_size

    return frame_list



def vad_check(input_audio_path, aggressiveness_mode, frame_duration, output_path):
    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_audio_path):
        audio_list = [input_audio_path]
    else:
        audio_list = glob.glob(os.path.join(input_audio_path, '*.wav'))

    os.makedirs(output_path, exist_ok=True)

    vad = webrtcvad.Vad(aggressiveness_mode)
    #vad.set_mode(aggressiveness_mode)

    pbar = tqdm(total=len(audio_list), desc='VAD check')
    for audio_file in audio_list:
        # read wav audio and parse to frames
        audio, sample_rate = read_wav(audio_file)
        sample_width = 2  # webrtcvad only support 16 bit (2 bytes) audio
        frames = get_frames(audio, sample_rate, sample_width, frame_duration)

        # init speech related status for this audio
        speech_detected = False
        speech_start_frame = 0
        speech_stop_frame = 0
        best_speech_start_frame = 0
        best_speech_stop_frame = 0
        best_speech_frame_num = 0
        frame_size = int(sample_rate * sample_width * (frame_duration / 1000.0))

        # go through all the frames to find out speech segmant
        for i in range(len(frames)):
            is_speech = vad.is_speech(frames[i], sample_rate)
            if not speech_detected and is_speech:
                #print('speech detected at frame %d' % i)
                speech_detected = True
                speech_start_frame = i
            elif speech_detected and not is_speech:
                #print('speech disappeared at frame %d' % i)
                speech_detected = False
                speech_stop_frame = i

                # only pick the longest speech segment as best, and ignore too short segment (< 10 frames)
                short_segment_threshold = 10
                if (speech_stop_frame - speech_start_frame) >= short_segment_threshold and (speech_stop_frame - speech_start_frame) > best_speech_frame_num:
                    best_speech_start_frame = speech_start_frame
                    best_speech_stop_frame = speech_stop_frame
                    best_speech_frame_num = best_speech_stop_frame - best_speech_start_frame

        # convert the start/stop frame id to start/stop time (in second), and save to txt output file
        best_speech_start_time = float(best_speech_start_frame) * float(frame_size) / float(sample_rate * sample_width)
        best_speech_stop_time = float(best_speech_stop_frame) * float(frame_size) / float(sample_rate * sample_width)

        txt_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        txt_file_name = os.path.join(output_path, txt_file_basename + '.txt')
        txt_file = open(txt_file_name, 'a+', encoding='utf-8')  # append the text in file, if needed

        # write speech start/stop time, 1 time per line, like:
        # cat output/00001.txt
        # 1.64
        # 3.06
        txt_file.write(str(best_speech_start_time))
        txt_file.write('\n')
        txt_file.write(str(best_speech_stop_time))
        txt_file.write('\n')
        txt_file.close()
        pbar.update(1)
    pbar.close()
    print('\nCheck finished. Speech start & stop time has been saved to %s' % output_path)



def main():
    parser = argparse.ArgumentParser(description='tool to check speech start/stop time for dry audio with webrtcvad')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--aggressiveness_mode', type=int, required=False, default=3, choices=[0, 1, 2, 3],
                        help = "aggressiveness mode of VAD (0/1/2/3). The higher the more aggressive. default=%(default)s")
    parser.add_argument('--frame_duration', type=int, required=False, default=10, choices=[10, 20, 30],
                        help = "frame duration time in ms(10/20/30). default=%(default)s")
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save speech duration info in txt files. default=%(default)s')
    args = parser.parse_args()

    vad_check(args.input_audio_path, args.aggressiveness_mode, args.frame_duration, args.output_path)


if __name__ == "__main__":
    main()


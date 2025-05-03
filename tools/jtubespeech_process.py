#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to split jtubespeech long wav audio to short segment, according to the subtitle timestamp
Reference from:
https://github.com/sarulab-speech/jtubespeech
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
#import time
import soundfile as sf
import librosa
import multiprocessing


def merge_subtitles(file_path):
    merged_subtitles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    previous_end_time = None
    current_start_time = None
    current_end_time = None
    current_sentence = ""

    for line in lines:
        start_time, end_time, sentence = line.strip().split('\t')
        sentence = sentence.replace("\"", "")
        start_time = float(start_time)
        end_time = float(end_time)

        if start_time == previous_end_time and current_sentence == sentence:
        # if previous_end_time is not None and (start_time - previous_end_time) < -0.05:
            # Merge with previous sentence
            current_end_time = end_time
            current_sentence = sentence
        else:
            # Store the previous sentence
            if current_sentence:
                merged_subtitles.append(f"{current_start_time:.3f}\t{current_end_time:.3f}\t{current_sentence}")

            # Start a new sentence
            current_start_time = start_time
            current_end_time = end_time
            current_sentence = sentence

        previous_end_time = end_time

    # Add the last sentence
    if current_sentence:
        merged_subtitles.append(f"{current_start_time:.3f}\t{current_end_time:.3f}\t{current_sentence}")

    return merged_subtitles


def process(queue, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path):
    # prepare wav audio segments path & subtitle text file
    output_audio_path = os.path.join(output_path, 'wav_segments')
    #subtitle_text_file = os.path.join(output_path, 'subtitle.txt')
    subtitle_text_path = os.path.join(output_path, 'subtitles')

    os.makedirs(output_audio_path, exist_ok=True)
    os.makedirs(subtitle_text_path, exist_ok=True)

    while not queue.empty():
        try:
            audio_id = queue.get_nowait()
            # load subtitle text and merge short items
            merged_subtitles = merge_subtitles(subtitle_txt_dict[audio_id])

            # load audio data and make sure to use 16k sample rate
            audio_data, sample_rate = sf.read(wav_16k_dict[audio_id])
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, sample_rate, 16000)
                sample_rate = 16000

            # if the audio length mismatch with subtitles, ignore it
            if len(audio_data) / sample_rate < float(merged_subtitles[-1].split("\t")[1]):
                continue

            # process one line of subtitle as an audio segment
            segment_index = 0
            for subtitle in merged_subtitles:
                # parse start time, end time & subtitle text from subtitle string
                [start_time, end_time, subtitle_text] = subtitle.split("\t")

                # if the duration time is too long or too short, ignore it
                speech_duration = float(end_time) - float(start_time)
                if speech_duration < min_duration or speech_duration > max_duration:
                    continue

                # get start sample & end sample index
                start_index = int(np.floor(float(start_time) * sample_rate))
                end_index = int(np.ceil(float(end_time) * sample_rate))

                # save the audio segment (from start_index to end_index) to wav file
                segment_id_str = f"%s_%s"%(audio_id, str(segment_index))
                sf.write(os.path.join(output_audio_path, segment_id_str+'.wav'), audio_data[start_index:end_index], sample_rate)
                #if not os.path.exists(os.path.join(output_audio_path, audio_id[:2])):
                    #os.mkdir(os.path.join(output_audio_path, audio_id[:2]))
                #sf.write(os.path.join(output_audio_path, audio_id[:2], segment_id_str+'.wav'), audio_data[start_index:end_index], sample_rate)

                # record segment id & subtitle text into subtitle text file, like:
                #
                # 7HZzGyALfuc_0 Bugün geçsin, belki çok değişirsin
                # 7HZzGyALfuc_1 Yarın, sen belki beni seversin
                # 7HZzGyALfuc_2 Bir gün daha sabredebilirsin
                # ......
                #
                #with open(subtitle_text_file, "a") as f:
                    #f.write(f"%s %s\n" % (segment_id_str, subtitle_text.lower()))

                # record subtitle text into subtitle text file, like:
                # cat output/subtitles/-3CSDniPZYY_0.txt
                # dil ne bilir şekeri şerbeti aldığın lezzeti baldan mı sandın?
                subtitle_text_file = os.path.join(subtitle_text_path, segment_id_str+'.txt')
                with open(subtitle_text_file, "w") as f:
                    f.write("%s\n" % subtitle_text.lower())

                segment_index = segment_index + 1
        except Exception as e:
            print('Exception: ', e)
            continue
    return



def multiprocess(audio_id_list, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path, num_thread):
    queue = multiprocessing.Queue()
    for audio_id in audio_id_list:
        queue.put(audio_id)

    print('Processing JTubeSpeech data, please wait...')
    process_list = list()
    for i in range(num_thread):
        process_list.append(multiprocessing.Process(target=process, args=(queue, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path)))
    try:
        for i in range(num_thread):
            process_list[i].start()
        # while not queue.empty():
        #     print("residue queue.qsize():", queue.qsize())
        #     time.sleep(10)
        for i in range(num_thread):
            process_list[i].join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers...")
        for i in range(num_thread):
            process_list[i].terminate()
            process_list[i].join()

    return



def singleprocess(audio_id_list, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path):
    # prepare wav audio segments path & subtitle text file
    output_audio_path = os.path.join(output_path, 'wav_segments')
    #subtitle_text_file = os.path.join(output_path, 'subtitle.txt')
    subtitle_text_path = os.path.join(output_path, 'subtitles')

    os.makedirs(output_audio_path, exist_ok=True)
    os.makedirs(subtitle_text_path, exist_ok=True)

    pbar = tqdm(total=len(audio_id_list), desc='JTubeSpeech data process')
    for audio_id in audio_id_list:
        try:
            # load subtitle text and merge short items
            merged_subtitles = merge_subtitles(subtitle_txt_dict[audio_id])

            # load audio data and make sure to use 16k sample rate
            audio_data, sample_rate = sf.read(wav_16k_dict[audio_id])
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, sample_rate, 16000)
                sample_rate = 16000

            # if the audio length mismatch with subtitles, ignore it
            if len(audio_data) / sample_rate < float(merged_subtitles[-1].split("\t")[1]):
                continue

            # process one line of subtitle as an audio segment
            segment_index = 0
            for subtitle in merged_subtitles:
                # parse start time, end time & subtitle text from subtitle string
                [start_time, end_time, subtitle_text] = subtitle.split("\t")

                # if the duration time is too long or too short, ignore it
                speech_duration = float(end_time) - float(start_time)
                if speech_duration < min_duration or speech_duration > max_duration:
                    continue

                # get start sample & end sample index
                start_index = int(np.floor(float(start_time) * sample_rate))
                end_index = int(np.ceil(float(end_time) * sample_rate))

                # save the audio segment (from start_index to end_index) to wav file
                segment_id_str = f"%s_%s"%(audio_id, str(segment_index))
                sf.write(os.path.join(output_audio_path, segment_id_str+'.wav'), audio_data[start_index:end_index], sample_rate)
                #if not os.path.exists(os.path.join(output_audio_path, audio_id[:2])):
                    #os.mkdir(os.path.join(output_audio_path, audio_id[:2]))
                #sf.write(os.path.join(output_audio_path, audio_id[:2], segment_id_str+'.wav'), audio_data[start_index:end_index], sample_rate)

                # record segment id & subtitle text into subtitle text file, like:
                #
                # 7HZzGyALfuc_0 Bugün geçsin, belki çok değişirsin
                # 7HZzGyALfuc_1 Yarın, sen belki beni seversin
                # 7HZzGyALfuc_2 Bir gün daha sabredebilirsin
                # ......
                #
                #with open(subtitle_text_file, "a") as f:
                    #f.write(f"%s %s\n" % (segment_id_str, subtitle_text.lower()))

                # record subtitle text into subtitle text file, like:
                # cat output/subtitles/-3CSDniPZYY_0.txt
                # dil ne bilir şekeri şerbeti aldığın lezzeti baldan mı sandın?
                subtitle_text_file = os.path.join(subtitle_text_path, segment_id_str+'.txt')
                with open(subtitle_text_file, "w") as f:
                    f.write("%s\n" % subtitle_text.lower())

                segment_index = segment_index + 1
        except Exception as e:
            print('Exception: ', e)
            continue
        pbar.update(1)
    pbar.close()
    return


def jtubespeech_process(wav_16k_path, subtitle_path, min_duration, max_duration, num_thread, output_path):
    # check input path
    assert (os.path.isdir(wav_16k_path)), 'wav_16k_path is not directory'
    assert (os.path.isdir(subtitle_path)), 'subtitle_path is not directory'

    # get 16k wav audio file list and convert to dict, format like:
    # {
    #    '7HJFa8Xct80': 'wav16k/7H/7HJFa8Xct80.wav',
    #    '7HZzGyALfuc': 'wav16k/7H/7HZzGyALfuc.wav',
    #    '80aWAOrE4GQ': 'wav16k/80/80aWAOrE4GQ.wav',
    #    ......
    # }
    wav_16k_list = glob.glob(os.path.join(wav_16k_path, '**/*.wav'), recursive=True)
    wav_16k_dict = {os.path.basename(wav_16k_file.strip()).split(".")[0]:wav_16k_file.strip() for wav_16k_file in wav_16k_list}

    # get subtitle text file list and convert to dict, format like:
    # {
    #    '7HJFa8Xct80': 'txt/7H/7HJFa8Xct80.txt',
    #    '7HZzGyALfuc': 'txt/7H/7HZzGyALfuc.txt',
    #    '80aWAOrE4GQ': 'txt/80/80aWAOrE4GQ.txt',
    #    ......
    # }
    subtitle_txt_list = glob.glob(os.path.join(subtitle_path, '**/*.txt'), recursive=True)
    subtitle_txt_dict = {os.path.basename(subtitle_txt_file.strip()).split(".")[0]:subtitle_txt_file.strip() for subtitle_txt_file in subtitle_txt_list}

    # find the intersection of wav & subtitle file with id, so we can get the valid audio id list like:
    # [
    #    '7HJFa8Xct80',
    #    '7HZzGyALfuc',
    #    '80aWAOrE4GQ',
    #    ......
    # ]
    audio_id_list = np.intersect1d(list(subtitle_txt_dict.keys()), list(wav_16k_dict.keys()))

    os.makedirs(output_path, exist_ok=True)
    if num_thread == 1:
        singleprocess(audio_id_list, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path)
    else:
        multiprocess(audio_id_list, subtitle_txt_dict, wav_16k_dict, min_duration, max_duration, output_path, num_thread)

    print('\nProcess done. audio segments & subtitle text file has been saved to %s' % output_path)
    return


def main():
    parser = argparse.ArgumentParser(description='Tool to split jtubespeech long wav audio to short segment according to the subtitle timestamp')
    parser.add_argument('--wav_16k_path', type=str, required=True,
                        help='directory for jtubespeech 16k .wav audio files')
    parser.add_argument('--subtitle_path', type=str, required=True,
                        help='directory for jtubespeech .txt subtitle files')
    parser.add_argument('--min_duration', type=float, required=False, default=1.0,
                        help='min speech duration (in seconds) to keep. default=%(default)s')
    parser.add_argument('--max_duration', type=float, required=False, default=11.0,
                        help='max speech duration (in seconds) to keep. default=%(default)s')
    parser.add_argument('--num_thread', type=int, required=False, default=1,
                        help='number of working thread. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save split audio segment files. default=%(default)s')
    args = parser.parse_args()

    jtubespeech_process(args.wav_16k_path, args.subtitle_path, args.min_duration, args.max_duration, args.num_thread, args.output_path)


if __name__ == "__main__":
    main()

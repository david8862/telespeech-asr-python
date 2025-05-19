#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to check ASR results with txt annotation
Reference from:
https://blog.51cto.com/u_16213332/8609370
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import shutil
import json

def count_same_chars_set(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    same_chars = set1.intersection(set2)
    return len(same_chars)

def count_same_chars_dict(str1,str2):
    char_dict = {}
    count = 0
    for char in str1:
        char_dict[char] = 1

    for char in str2:
        if char in char_dict:
            count += 1
    return count

def count_same_chars_list(str1, str2):
    char_list1 = [char for char in str1]
    char_list2 = [char for char in str2]
    same_chars = [char for char in char_list1 if char in char_list2]
    return len(same_chars)


def asr_result_check(audio_path, asr_result_path, annotation_path, keep_num, output_path):
    # get .txt result file list or single .txt result file
    if os.path.isfile(asr_result_path):
        asr_result_list = [asr_result_path]
    else:
        asr_result_list = glob.glob(os.path.join(asr_result_path, '*.txt'))

    # get .txt annotation file list or single .txt annotation file
    if os.path.isfile(annotation_path):
        annotation_list = [annotation_path]
    else:
        annotation_list = glob.glob(os.path.join(annotation_path, '*.txt'))
    # try to strip the directory and only keep the file name
    annotation_basename_list = [os.path.split(annotation_filename)[-1] for annotation_filename in annotation_list]

    # create a list to record the ASR result check output
    asr_result_check_list = []

    pbar = tqdm(total=len(asr_result_list), desc='ASR result check')
    for asr_result_filename in asr_result_list:
        # try to strip the directory and only keep the file name
        asr_result_basename = os.path.split(asr_result_filename)[-1]

        # check if there is corresponding annotation file
        if annotation_basename_list.count(asr_result_basename) <= 0:
            print("ASR result %s is not found in annotation!" % asr_result_filename)
            pbar.update(1)
            continue

        # just read 1st line of ASR result file (speech content)
        asr_result_file = open(asr_result_filename, 'r', encoding='utf-8')
        asr_result_str = asr_result_file.readline().strip()
        asr_result_file.close()

        annotation_index = annotation_basename_list.index(asr_result_basename)
        annotation_filename = annotation_list[annotation_index]
        annotation_file = open(annotation_filename, 'r', encoding='utf-8')
        # just read 1st line of annotation file (speech content)
        annotation_str = annotation_file.readline().strip()
        annotation_file.close()

        # compare annotation string & ASR result string, to get mismatch charactor number
        mismatch_char_num = len(annotation_str) - count_same_chars_set(annotation_str, asr_result_str)
        #mismatch_char_num = len(annotation_str) - count_same_chars_dict(annotation_str, asr_result_str)
        #mismatch_char_num = len(annotation_str) - count_same_chars_list(annotation_str, asr_result_str)

        assert (mismatch_char_num >= 0), 'invalid mismatch charactor number: %d' % mismatch_char_num

        # record the audio in list if there's any mismatch charactor number, with format:
        #  [
        #    ['audio file name', 'annotation text', 'mismatch char rate']
        #  ]
        # like:
        # [
        #   ['00001.wav', '开始清洁', 0.75],
        #   ['00002.wav', '吸力大点', 0.5],
        #   ......
        # ]
        if mismatch_char_num > 0:
            mismatch_char_rate = float(mismatch_char_num) / len(annotation_str)
            audio_filename = os.path.splitext(asr_result_basename)[0] + '.wav'
            asr_result_check_list.append([audio_filename, annotation_str, mismatch_char_rate])
        pbar.update(1)
    pbar.close()

    # descend sort the list with "mismatch_char_rate"
    sorted_asr_result_check_list = sorted(asr_result_check_list, key=lambda item:item[2], reverse=True)

    # only keep top N if needed
    if keep_num > 0:
        sorted_asr_result_check_list = sorted_asr_result_check_list[:keep_num]

    # save the issue audio & result check list to output path
    os.makedirs(output_path, exist_ok=True)

    pbar = tqdm(total=len(sorted_asr_result_check_list), desc='save check output')
    target_audio_path = os.path.join(output_path, 'audio')
    os.makedirs(target_audio_path, exist_ok=True)
    for asr_result_item in sorted_asr_result_check_list:
        audio_filename = asr_result_item[0]
        src_audio_file = os.path.join(audio_path, audio_filename)
        target_audio_file = os.path.join(target_audio_path, audio_filename)
        shutil.copyfile(src_audio_file, target_audio_file)
        pbar.update(1)
    pbar.close()

    # save the result list to json file, one line per audio
    output_json_file = os.path.join(output_path, 'result.json')
    json_file = open(output_json_file, 'w', encoding='utf-8')
    for asr_result in sorted_asr_result_check_list:
        json_file.write(str(asr_result) + '\n')
    json_file.close()
    print('\nDone. ASR result check output has been saved to %s' % output_path)


def main():
    parser = argparse.ArgumentParser(description='Tool to check ASR results with txt annotation files')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='file or directory for wav format audio file')
    parser.add_argument('--asr_result_path', type=str, required=True,
                        help='file or directory for speech content text generated by ASR model')
    parser.add_argument('--annotation_path', type=str, required=True,
                        help='file or directory for txt format annotation')
    parser.add_argument('--keep_num', type=int, required=False, default=-1,
                        help='only keep top N audios in result, -1 for keep all. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save the check result. default=%(default)s')
    args = parser.parse_args()

    asr_result_check(args.audio_path, args.asr_result_path, args.annotation_path, args.keep_num, args.output_path)


if __name__ == "__main__":
    main()

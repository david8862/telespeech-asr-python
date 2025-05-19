#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to convert single text content list file to
    a directory with one txt file per audio
"""
import os, sys, argparse
from tqdm import tqdm


def text_list_convert(text_list_file, output_path):
    # text_list_file contains a list of audio file basename & text content for
    # this audio. one line per audio, like:
    #
    # cat text_list.txt
    # norm_221205231 Haritalamaya başla
    # norm_221205024 Mutfağı temizle
    # norm_221205246 Oturma odasını temizle
    # ...
    #
    text_file = open(text_list_file, 'r', encoding='utf-8')
    text_list = text_file.readlines()

    os.makedirs(output_path, exist_ok=True)

    pbar = tqdm(total=len(text_list), desc='text list convert')
    for text_item in text_list:
        # parse audio file basename & text content
        text_segments = text_item.strip().split()
        audio_file_basename = text_segments[0]
        text_content = ' '.join(text_segments[1:])

        # save text content to single audio txt file, with same basename as
        # the corresponding audio, like:
        #
        # cat norm_221205231.txt
        # Haritalamaya başla
        #
        output_txt_file = os.path.join(output_path, audio_file_basename+'.txt')
        txt_file = open(output_txt_file, 'w', encoding='utf-8')
        txt_file.write(text_content)
        txt_file.close()
        pbar.update(1)
    pbar.close()
    print('\nDone. Single audio text files been saved to %s' % output_path)


def main():
    parser = argparse.ArgumentParser(description='Tool to convert single text content list file to a directory with one txt file per audio')
    parser.add_argument('--text_list_file', type=str, required=True,
                        help='text file with list of audio file basename & text content')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save single audio txt file. default=%(default)s')
    args = parser.parse_args()

    text_list_convert(args.text_list_file, args.output_path)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to convert excel format annotation to one txt per audio.
Reference from:
https://www.cnblogs.com/flyup/p/15264897.html
"""
import os, sys, argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

def excel_annotation_convert(input_excel, output_path, content_only):
    # excel sheet format:
    #
    # audio_name | speech content | speech rate | gender | age | area
    # 00001.wav  |    客厅扫拖    |    常速     |   男   | 25  | 成都
    # 00002.wav  |      集尘      |    常速     |   男   | 35  | 重庆
    # ......
    #
    df = pd.read_excel(input_excel)
    os.makedirs(output_path, exist_ok=True)

    audio_num = len(df.values)
    pbar = tqdm(total=audio_num, desc='excel annotation convert')
    for i in range(audio_num):
        # parse annotation content
        audio_file_name = df.values[i, 0]
        if not audio_file_name.endswith('.wav')
            audio_file_name = os.path.splitext(audio_file_name)[0] + '.wav'

        speech_content = df.values[i, 1]
        speech_rate = df.values[i, 2]
        gender = df.values[i, 3]
        age = df.values[i, 4]
        area = df.values[i, 5]

        # target txt annotation file
        txt_file_basename = os.path.splitext(os.path.split(audio_file_name)[-1])[0]
        txt_file_name = os.path.join(output_path, txt_file_basename+'.txt')
        txt_file = open(txt_file_name, 'w', encoding='utf-8')

        # write speech content
        txt_file.write(str(speech_content))
        txt_file.write('\n')

        # write other annotation info if needed
        if not content_only:
            txt_file.write(str(speech_rate))
            txt_file.write('\n')

            txt_file.write(str(gender))
            txt_file.write('\n')

            txt_file.write(str(age))
            txt_file.write('\n')

            txt_file.write(str(area))
            txt_file.write('\n')
        txt_file.close()
        pbar.update(1)
    pbar.close()
    print('\nConvert finished. Txt annotation files saved to %s' % output_path)


def main():
    parser = argparse.ArgumentParser(description='tool to convert excel format annotation to txt')
    parser.add_argument('--input_excel', type=str, required=True,
                        help='input excel (.xls/.xlsx) annotation file')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save txt annotation files. default=%(default)s')
    parser.add_argument('--content_only', default=False, action="store_true",
                        help='only convert speech content and igore other info')
    args = parser.parse_args()

    excel_annotation_convert(args.input_excel, args.output_path, args.content_only)


if __name__ == "__main__":
    main()

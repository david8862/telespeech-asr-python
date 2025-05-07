#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to run ASR inference with OpenAI Whisper model
Reference from:
https://github.com/IS2AI/TurkicASR
https://espnet.github.io/espnet/installation.html

Prepare environment:
apt install cmake sox flac
git clone https://github.com/espnet/espnet
cd espnet/tools
./setup_venv.sh $(command -v python3)
. ./activate_python.sh
make TH_VERSION=2.2.0 CUDA_VERSION=11.8  (here you need to follow your local CUDA version)

bash -c ". ./activate_python.sh; . ./extra_path.sh; python3 check_install.py"
cd ../egs2
git clone https://github.com/IS2AI/TurkicASR.git
cd TurkicASR

# download pretrained model package and extract to "espnet/egs2/TurkicASR" dir.
# See "https://github.com/IS2AI/TurkicASR"
turkic_languages_model.zip: https://drive.google.com/file/d/1GtK-OrH3ZRYz2Zc8vf-xndp7R9dic4rV/view?usp=sharing
all_languages_model.zip: https://drive.google.com/file/d/15Dc4Uwzqqrw3jkE5-zrgVAyNddGS7onw/view?usp=sharing

# copy this script to "espnet/egs2/TurkicASR" and run:
python turkish_batch_inference.py --input_audio_path=/path/audio/ --asr_model_path=exp/asr_train_asr_1410_raw_all_turkic_1610_char_sp --asr_model_file=exp/asr_train_asr_1410_raw_all_turkic_1610_char_sp/valid.acc.ave_10best.pth --lm_model_path=exp/lm_train_lm_1410_all_turkic_1610_char --lm_model_file=exp/lm_train_lm_1410_all_turkic_1610_char/valid.loss.ave_10best.pth --output_path=/path/txt_result/


NOTE: to have a better inference performance, you'd better to have a Nvidia GPU with >= 4GB Memory
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import soundfile
import torch
from espnet2.bin.asr_inference import Speech2Text


def turkish_batch_inference(input_audio_path, asr_model_path, asr_model_file, lm_model_path, lm_model_file, segment_duration, output_path):
    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_audio_path):
        audio_list = [input_audio_path]
    else:
        audio_list = glob.glob(os.path.join(input_audio_path, '*.wav'))

    os.makedirs(output_path, exist_ok=True)

    train_config = os.path.join(asr_model_path, 'config.yaml')
    lm_config = os.path.join(lm_model_path, 'config.yaml')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_model = Speech2Text(asr_train_config=train_config,
                            asr_model_file=asr_model_file,
                            lm_train_config=lm_config,
                            lm_file=lm_model_file,
                            token_type=None,
                            bpemodel=None,
                            maxlenratio=0.0,
                            minlenratio=0.0,
                            beam_size=10,
                            ctc_weight=0.5,
                            lm_weight=0.3,
                            penalty=0.0,
                            nbest=1,
                            device=device)

    pbar = tqdm(total=len(audio_list), desc='Whisper ASR inference')
    for audio_file in audio_list:
        audio_data, sample_rate = soundfile.read(audio_file)

        # cut the audio sequence into segments for inference
        offset = 0
        output_text = ""
        segment_size = int(sample_rate * segment_duration)
        while offset + segment_size < len(audio_data):
            # ASR inference
            output = stt_model(audio_data[offset:offset+segment_size])
            # output format:
            # [(text, token, token_int, hypothesis object), ...]
            text = output[0][0]
            #print(text)

            # top 4 char in output text would be language indication, like:
            # "[tr] ay kalbimi yalın"
            language = text[:4]
            if language != '[tr]':
                print('detect non-turkish language:', language)
            output_text += text[5:]
            offset += segment_size

        # save ASR result segments to txt file
        txt_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        txt_file_name = os.path.join(output_path, txt_file_basename + '.txt')
        txt_file = open(txt_file_name, 'w', encoding='utf-8')
        txt_file.write(output_text)

        txt_file.close()
        pbar.update(1)
    pbar.close()

    print('\nCheck finished. Speech start & stop time has been saved to %s' % output_path)
    return



def main():
    parser = argparse.ArgumentParser(description='tool to run ASR inference with OpenAI Whisper model')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--asr_model_path', type=str, required=False, default="exp/asr_train_ksc2_raw_ksc2_char_sp",
                        help="trained ASR model path. default=%(default)s")
    parser.add_argument('--asr_model_file', type=str, required=False, default="exp/asr_train_asr_1410_raw_all_turkic_1610_char_sp/valid.acc.ave_10best.pth",
                        help="trained ASR model file. default=%(default)s")
    parser.add_argument('--lm_model_path', type=str, required=False, default="exp/lm_train_lm_ksc2_char",
                        help="trained language model path. default=%(default)s")
    parser.add_argument('--lm_model_file', type=str, required=False, default="exp/lm_train_lm_ksc2_char/valid.loss.ave_10best.pth",
                        help="trained language model file. default=%(default)s")
    parser.add_argument('--segment_duration', type=int, required=False, default=5,
                        help="segment duration time in second for inference. default=%(default)s")
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save transcribe text into txt files. default=%(default)s')
    args = parser.parse_args()

    turkish_batch_inference(args.input_audio_path, args.asr_model_path, args.asr_model_file, args.lm_model_path, args.lm_model_file, args.segment_duration, args.output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to run sichuan dialect ASR inference with a Paraformer finetuning model from modelscope
Reference from:
https://www.modelscope.cn/models/lukeewin01/paraformer-large-sichuan-offline/

funasr could be installed with following cmd:
pip install funasr

NOTE: to have a better inference performance, you'd better to have a Nvidia GPU with >= 4GB Memory
"""
import os, sys, argparse
import glob
from tqdm import tqdm
#import torch

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def sichuan_batch_inference(input_audio_path, asr_model_path, device, output_path):
    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_audio_path):
        audio_list = [input_audio_path]
    else:
        audio_list = glob.glob(os.path.join(input_audio_path, '*.wav'))

    os.makedirs(output_path, exist_ok=True)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(model=asr_model_path, disable_update=True, device=device)

    pbar = tqdm(total=len(audio_list), desc='Sichuan ASR inference')
    for audio_file in audio_list:

        res = model.generate(input=audio_file, cache={}, use_itn=False, batch_size_s=1)
        output_text = rich_transcription_postprocess(res[0]["text"])

        # save ASR result segments to txt file
        txt_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        txt_file_name = os.path.join(output_path, txt_file_basename + '.txt')
        txt_file = open(txt_file_name, 'w', encoding='utf-8')
        txt_file.write(output_text.lower())
        txt_file.write('\n')
        txt_file.close()
        pbar.update(1)
    pbar.close()

    print('\nCheck finished. ASR result has been saved to %s' % output_path)
    return



def main():
    parser = argparse.ArgumentParser(description='tool to run ASR inference with OpenAI Whisper model')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--asr_model_path', type=str, required=False, default="lukeewin01/paraformer-large-sichuan-offline",
                        help="trained ASR model path. default=%(default)s")
    parser.add_argument('--device', type=str, required=False, default="cpu", choices=["cpu", "cuda"],
                        help="device for inference. default=%(default)s")
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save transcribe text into txt files. default=%(default)s')
    args = parser.parse_args()

    sichuan_batch_inference(args.input_audio_path, args.asr_model_path, args.device, args.output_path)


if __name__ == "__main__":
    main()

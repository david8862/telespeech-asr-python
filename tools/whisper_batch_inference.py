#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to run ASR inference with OpenAI Whisper model
Reference from:
https://blog.csdn.net/hhy321/article/details/134897967
https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish

openai-whisper could be installed with following cmd:
pip install openai-whisper
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install zhconv wheel

install PyTorch if needed:
pip install torch torchvision torchaudio

NOTE: to have a better inference performance with large model (like 'large-v3-turbo'), you'd
      better to have a Nvidia GPU with >= 8GB Memory
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import whisper
import zhconv


def whisper_batch_inference(input_audio_path, model_type, language, no_speech_threshold, with_timestamp, fp16, output_path):
    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_audio_path):
        audio_list = [input_audio_path]
    else:
        audio_list = glob.glob(os.path.join(input_audio_path, '*.wav'))

    os.makedirs(output_path, exist_ok=True)

    model = whisper.load_model(model_type)

    pbar = tqdm(total=len(audio_list), desc='Whisper ASR inference')
    for audio_file in audio_list:
        result = model.transcribe(audio_file,
                                  language=language,
                                  verbose=None,  # None/True/False
                                  compression_ratio_threshold=2.4,  # if gzip compression ratio above this value, treat as failed
                                  logprob_threshold=-1.0,  # if avg_logprob below this value, treat as failed
                                  # if no_speech_prob is higher than this value AND avg_logprob below
                                  # `logprob_threshold`, consider the segment as silent
                                  no_speech_threshold=no_speech_threshold,
                                  word_timestamps=False,
                                  clip_timestamps='0',
                                  fp16=fp16
                                 )

        # check if language is correct
        if language is not None:
            assert (result['language'] == language), 'language mismatch: %s' % result['language']
        else:
            print('Language:', str(result['language']))

        # save ASR result to txt file
        txt_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        txt_file_name = os.path.join(output_path, txt_file_basename + '.txt')
        txt_file = open(txt_file_name, 'w')

        if with_timestamp:
            # Whisper ASR segments format:
            # [{
            #    'id': 0,
            #    'seek': 0,
            #    'start': 0.0,
            #    'end': 1.0,
            #    'text': '清洁完成',
            #    'tokens': [50364, 21784, 35622, 242, 41509, 50414],
            #    'temperature': 0.0,
            #    'avg_logprob': -0.46264615740094867,
            #    'compression_ratio': 0.8873239436619719,
            #    'no_speech_prob': 0.01184895820915699
            # }]
            asr_segments = result['segments']

            # save start/stop time and text for each segment
            if len(asr_segments) > 0:
                for asr_segment in asr_segments:
                    # for Chinese, make sure it is simplified format
                    if language == 'zh':
                        text = zhconv.convert(asr_segment['text'], 'zh-cn')
                    else:
                        text = str(asr_segment['text'])

                    # write speech start/stop time and content text, 1 segment per line, like:
                    # cat output/00001.txt
                    # 0.000   1.000   清洁完成
                    # 1.160   2.000   开始回充
                    txt_file.write("%.3f"%asr_segment['start'] + '\t' + "%.3f"%asr_segment['end'] + '\t' + text.lower())
                    txt_file.write('\n')
        else:
            # just save text for the whole audio, like:
            # cat output/00001.txt
            # 清洁完成,开始回充
            asr_text = result['text']
            txt_file.write(asr_text.lower())
            txt_file.write('\n')
        txt_file.close()
        pbar.update(1)
    pbar.close()

    if with_timestamp:
        print('\nCheck finished. Speech content and start & stop time has been saved to %s' % output_path)
    else:
        print('\nCheck finished. Speech content has been saved to %s' % output_path)

    return



def main():
    parser = argparse.ArgumentParser(description='tool to run ASR inference with OpenAI Whisper model')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--model_type', type=str, required=False, default='tiny',
                        choices=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
                        help="Whisper model type to use. default=%(default)s")
    parser.add_argument('--language', type=str, required=False, default=None,
                        choices=[None, 'zh', 'en', 'fr', 'de', 'it', 'es', 'ja', 'ko', 'ru', 'tr', 'th'],
                        help = "Target language to transcribe, None for auto-detect. default=%(default)s")
    parser.add_argument('--no_speech_threshold', type=float, required=False, default=0.8,
                        help="threshold to judge if an audio segment contains speech. default=%(default)s")
    parser.add_argument('--with_timestamp', default=False, action="store_true",
                        help='Whether to record start/stop timestamp for each speech segment. default=%(default)s')
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='Whether to use fp16 inference. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save transcribe text into txt files. default=%(default)s')
    args = parser.parse_args()

    whisper_batch_inference(args.input_audio_path, args.model_type, args.language, args.no_speech_threshold, args.with_timestamp, args.fp16, args.output_path)


if __name__ == "__main__":
    main()

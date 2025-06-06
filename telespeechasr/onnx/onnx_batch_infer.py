# -*- coding:utf-8 -*-
# @FileName  :onnx_infer.py
# @Time      :2024/6/3 15:33
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import json
import logging
import os
import glob
from tqdm import tqdm
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import kaldi_native_fbank as knf
import numpy as np
import soundfile as sf
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

class OrtInferRuntimeSession:
    def __init__(self, model_file, device='cpu', device_id=-1, intra_op_num_threads=4):
        if device == 'cpu':
            EP_list = ['CPUExecutionProvider']
        elif device == 'cuda':
            EP_list = ['CUDAExecutionProvider']
        elif device == 'tensorrt':
            EP_list = ['TensorrtExecutionProvider']
        else:
            raise ValueError('Unsupported device: ', device)

        if isinstance(model_file, list):
            merged_model_file = b""
            for file in sorted(model_file):
                with open(file, "rb") as onnx_file:
                    merged_model_file += onnx_file.read()

            model_file = merged_model_file
        else:
            self._verify_model(model_file)
        self.session = InferenceSession(
            #model_file, sess_options=sess_opt, providers=EP_list
            model_file, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        #if device_id != "-1" and cuda_ep not in self.session.get_providers():
            #warnings.warn(
                #f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                #"Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                #"you can check their relations from the offical web site: "
                #"https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                #RuntimeWarning,
            #)

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content[None, ...]))
        try:
            result = self.session.run(self.get_output_names(), input_dict)
            return result
        except Exception as e:
            raise RuntimeError("ONNXRuntime inferece failed.") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class TeleSpeechAsrInferSession:
    def __init__(
        self, model_file, vocab_path=None, device='cpu', device_id=-1, intra_op_num_threads=4
    ):
        self.vocab_path = vocab_path or os.path.join(
            os.path.dirname(__file__), "data", "vocab.json"
        )

        with open(self.vocab_path, "r", encoding='utf-8') as f:
            self.vocab2id = json.load(f)
            self.id2vocab = {}
            for k, v in self.vocab2id.items():
                self.id2vocab[v] = k

        logging.info(f"Loading model from {model_file}")
        self.session = OrtInferRuntimeSession(
            model_file, device=device, device_id=device_id, intra_op_num_threads=intra_op_num_threads
        )

        self.eps = 1e-5

        self.blank_weight = 0.0
        self.blank_mode = "add"

    def postprocess(self, feats):
        m = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True)
        feats = (feats - m) / (std + self.eps)
        return feats

    def get_logits(self, logits):
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        return logits

    def viterbi_decode(
        self,
        emissions: np.ndarray,
    ) -> List[List[Dict[str, np.ndarray]]]:
        def get_pred(e):
            toks = e.argmax(-1)
            return toks[toks != 0]

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]

    def postprocess_sentence(self, tokens):
        text = ""
        for token in tokens:
            if token in self.id2vocab:
                token = self.id2vocab[token]
                text += token
        return text

    def load_audio(self, filename: str) -> Tuple[np.ndarray, int]:
        data, sample_rate = sf.read(
            filename,
            always_2d=True,
            dtype="float32",
        )
        data = data[:, 0]  # use only the first channel
        samples = np.ascontiguousarray(data)
        return samples, sample_rate

    def get_features(self, file_path: str) -> np.ndarray:
        samples, sample_rate = self.load_audio(file_path)

        if sample_rate != 16000:
            import librosa

            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        samples *= 32768

        opts = knf.MfccOptions()
        # See https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/mfcc_hires.conf
        opts.frame_opts.dither = 0

        opts.num_ceps = 40
        opts.use_energy = False

        opts.mel_opts.num_bins = 40
        opts.mel_opts.low_freq = 40
        opts.mel_opts.high_freq = -200
        mfcc = knf.OnlineMfcc(opts)
        mfcc.accept_waveform(16000, samples)
        frames = []
        for i in range(mfcc.num_frames_ready):
            frames.append(mfcc.get_frame(i))

        frames = np.stack(frames, axis=0)
        return frames

    def infer(self, audio_path):
        feats = self.get_features(audio_path)
        feats = self.postprocess(feats)[None, ...]

        #logging.info("Decoding ...")
        start_time = time.time()
        model_output = self.session(feats)
        emissions = self.get_logits(model_output)
        emissions = emissions[0].transpose((1, 0, 2))
        hypos = self.viterbi_decode(emissions)
        result = self.postprocess_sentence(hypos[0][0]["tokens"])
        #logging.info(f"Inference time: {time.time() - start_time:.4}s")

        return result


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--audio_path", type=str, required=True)
    args.add_argument("--vocab_path", type=str, default=None)
    args.add_argument('--output_path', type=str, required=False, default=None)
    args.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "tensorrt"]
    )

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = args.parse_args()

    # get audio file list or single audio file
    if os.path.isfile(args.audio_path):
        audio_list = [args.audio_path]
    else:
        audio_list = glob.glob(os.path.join(args.audio_path, '*.wav'))

    model = TeleSpeechAsrInferSession(args.model_path, args.vocab_path, args.device)

    if len(audio_list) == 1:
        audio_file = audio_list[0]
        asr_result = model.infer(audio_file)
        if args.output_path is None:
            logging.info(asr_result)
        else:
            os.makedirs(args.output_path, exist_ok=True)
            audio_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
            output_file_name = os.path.join(args.output_path, audio_file_basename+'.txt')
            output_file = open(output_file_name, 'w')
            output_file.write(asr_result)
            output_file.write('\n')
            output_file.close()
    else:
        assert (args.output_path is not None), 'need to provide output path for several audio files'

        pbar = tqdm(total=len(audio_list), desc='Telespeech-ASR ONNX inference')
        for audio_file in audio_list:
            asr_result = model.infer(audio_file)

            os.makedirs(args.output_path, exist_ok=True)
            audio_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
            output_file_name = os.path.join(args.output_path, audio_file_basename+'.txt')
            output_file = open(output_file_name, 'w')
            output_file.write(asr_result)
            output_file.write('\n')
            output_file.close()
            pbar.update(1)
        pbar.close()
    print('\nInference done, ASR result has been saved to %s' % args.output_path)

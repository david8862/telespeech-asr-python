<br/>
<h2 align="center">Telespeech-asr-python</h2>
<br/>


[TeleSpeech-ASR（星辰超多方言语音识别大模型）](https://github.com/Tele-AI/TeleSpeech-ASR)是由中国电信人工智能研究院（TeleAI）发布业内首个支持30种方言自由混说的语音识别大模型。

首先感谢电信团队的开源奉献，该模型是目前来看修改版的data2vec， 整个模型类似于wav2vec_ctc， 期待后续技术报告及论文的发布。

由于原项目依赖fairseq和kaldi预处理， 光跑起来就非常麻烦，本项目提供一个不依赖与fairseq和kaldi的**推理环境**方便模型测试。

模型使用官方在KeSpeech数据集8种方言微调的模型

现sherpa-onnx已支持telespeech的c++ runtime， 见[详情](https://github.com/k2-fsa/sherpa-onnx/pull/970)。

## 如何使用

### 1. 安装依赖

onnxruntime 只需要安装requirements-onnxruntime.txt里面的依赖即可
```bash
pip install -r requirements-onnxruntime.txt
```

### 2. 下载模型

从huggingface
```bash
wget https://huggingface.co/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt

# 或者使用镜像
wget https://hf-mirror.com/lovemefan/telespeech/resolve/main/finetune_large_kespeech.pt?download=true -O finetune_large_kespeech.pt
```

### 3. 模型导出

<font color='brown'>如果修改了词表，需要手动修改torchscript_export.py
或onnx_export.py中的词表大小</font>
```python
Data2VecMultiModel(vocab_size=7535)
```

1. onnx 导出

```bash
PYTHONPATH=$PWD python telespeechasr/onnx/onnx_export.py --model_path /path/torch_checkpoint.pt
--output_dir /path/output_dir
```

### 4. 模型推理（目前还不支持batch解码）

**以下模型都可在huggingface [下载](https://huggingface.co/lovemefan/telespeech/tree/main)**

1. onnx 推理, 支持gpu，cpu推理
```bash
PYTHONPATH=$PWD python telespeechasr/onnx/onnx_infer.py --model_path /path/model_export.onnx
--audio_path /path/audio.wav
```

2. onnx 批量推理, 可成批处理一个目录下的全部wav音频文件，并将识别结果文本保存为指定输出目录下的同名txt文件。支持gpu，cpu推理
```bash
PYTHONPATH=$PWD python telespeechasr/onnx/onnx_batch_infer.py --model_path /path/model_export.onnx
--audio_path /path/audio_path/ --output_path /path/output/ --device cuda
```

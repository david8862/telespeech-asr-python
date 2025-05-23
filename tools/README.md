## Step 1: Convert xls/xlsx format annotation to txt annotation
python excel_annotation_convert.py --input_excel=/path/annotation.xlsx --output_path=/path/txt_annotation/ [--content_only]

## Step 2: run Telespeech ASR model (onnx format) on audios to generate txt format ASR result
python onnx_batch_infer.py --model_path=../models/model_export.onnx --audio_path=/path/audio/ --vocab_path=../telespeechasr/onnx/data/vocab.json --output_path=/path/txt_result/ --device=cpu (cuda,tensorrt)

## Step 2.1 (Optional): run Openai Whisper on audios to generate txt format ASR result, but it may be only good for mandarin. Related packages need to be installed
pip install openai-whisper
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install zhconv wheel

python whisper_batch_inference.py --model_type=large-v3-turbo --input_audio_path=/path/audio/ --output_path=/path/txt_result/ --language=zh


## Step 2.5 (Optional): Check the speech start/stop time (in second) with webrtcvad, and save to ASR result (will merge into ASR result txt)
python vad_check.py --input_audio_path=/path/audio/ --frame_duration=10 --aggressiveness_mode=3 --output_path=/path/txt_result/

## Step 3: check the ASR result with txt annotation to find out issue audio
python asr_result_check.py --audio_path=/path/audio/ --asr_result_path=/path/txt_result/ --annotation_path=/path/txt_annotation/ --output_path=/path/result/ [--keep_num=1000]

Issue audio & .json summary file "result.json" will be placed at "/path/result/" dir. The format of issue audio record in json file will be:
```
['audio file name', 'annotation text', 'mismatch char number with ASR result']
```
like
```
["00001.wav", "开始清洁", 3],
["00002.wav", "吸力大点", 2],
```
and all records will be sorted in descending order based on the number of mismatch characters.

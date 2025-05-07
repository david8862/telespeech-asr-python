## Step 1: Split JTubeSpeech long Turkish audios & subtitle into short segments
python jtubespeech_process.py --wav_16k_path=/path/jtubespeech_wav16k_audio/ --subtitle_path=/path/jtubespeech_subtitle_txt/ --min_duration=1 --max_duration=11 --num_thread=1 --output_path=audio_segments_output

## Step 2: check & pick speech audios from all audio segments, using Tensorflow YAMNet sound classification model
python jtubespeech_yamnet_check.py --wav_16k_path=audio_segments_output/wav_segments/ --subtitle_path=audio_segments_output/subtitles/ --output_path=speech_segments_output


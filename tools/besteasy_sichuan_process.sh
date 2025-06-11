#!/bin/bash
if [[ "$#" -ne 1 ]] && [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <target_dir> [inference_device=cpu/cuda]"
    exit 1
fi

TARGET_DIR=$1
if [ "$#" -eq 2 ]; then
    INFERENCE_DEVICE=$2
else
    INFERENCE_DEVICE="cpu"
fi

SUB_DIR_NUM=$(ls -d $TARGET_DIR/* | wc -l)
i=0

for SUB_DIR in `ls -d $TARGET_DIR/*`
do
    let i=i+1
    if [ -d $SUB_DIR ]; then
        printf "Processing %s (%d/%d)\n" "$SUB_DIR" "$i" "$SUB_DIR_NUM"

        ## Step 0: adjust target dir
        mkdir -p $SUB_DIR/audio
        mv $SUB_DIR/*.wav $SUB_DIR/audio/

        # Step 1: Convert xls/xlsx format annotation to txt annotation
        EXCEL_BASENAME=$(basename $SUB_DIR)
        python excel_annotation_convert.py --input_excel=$SUB_DIR/$EXCEL_BASENAME.xlsx --output_path=$SUB_DIR/txt_annotation/

        # Step 2: run Paraformer finetuning sichuan dialect ASR model on audios to generate txt format ASR result
        python sichuan_batch_inference.py --input_audio_path=$SUB_DIR/audio/ --output_path=$SUB_DIR/txt_result/ --device=$INFERENCE_DEVICE

        # Step 3: check the ASR result with txt annotation to find out issue audio
        python asr_result_check.py --audio_path=$SUB_DIR/audio/ --asr_result_path=$SUB_DIR/txt_result/ --annotation_path=$SUB_DIR/txt_annotation/ --convert_to_pinyin --output_path=$SUB_DIR/result/

        # Step 4: clean temp files and only keep final result
        #rm -rf $SUB_DIR/audio $SUB_DIR/txt_annotation $SUB_DIR/txt_result
    fi
done
echo "Done"

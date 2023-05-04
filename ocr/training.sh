#!/bin/bash

# Simple training script for Tesseract 4/5
# This script assumes you have built Tesseract from source and have the
# tesstrain repository cloned.

# If not follow the documentation
# Or it is also very well explained in this video: https://www.youtube.com/watch?v=veJt3U44yqc (build)
# and https://www.youtube.com/watch?v=KE4xEzFGSU8 (training)

# Configure paths - change these to match your system
TESSTRAIN_FOLDER=~/dev/tesstrain
TESSERACT_FOLDER=~/dev/tesseract
TESSDATA=$TESSERACT_FOLDER/tessdata  # Path to tessdata
TESSDATA_PREFIX=$TESSDATA  # Required for the makefile
export TESSDATA_PREFIX

# Configure training parameters
LANG=frk   # Language code, for this we will use Fraktur
ITERATIONS=100000  # 100k
MODEL_NAME=CUSTOM_FRK

echo "Verifiying structure"
if [ ! -d ${TESSTRAIN_FOLDER}/data/${MODEL_NAME}-ground-truth ]; then
    echo "Ground truth folder missing, please copy your ground truth files to $TESSTRAIN_FOLDER/data/$MODEL_NAME-ground-truth"
    exit 1
fi

if [ ! -f ${TESSDATA}/${LANG}.traineddata ]; then
    echo "Language data missing, please copy your language data to $TESSDATA/$LANG.traineddata"
    exit 1
fi

# Rename .txt to .gt.txt
echo "Renaming ground truth files"
cd $TESSTRAIN_FOLDER/data/${MODEL_NAME}-ground-truth
for file in *.txt; do
    # Check if file already does not end with .gt.txt
    if [[ $file == *.gt.txt ]]; then
        continue
    fi

    mv -- "$file" "${file%.txt}.gt.txt"
done

echo "Starting training"
cd $TESSTRAIN_FOLDER
make training START_MODEL=$LANG \
    MODEL_NAME=$MODEL_NAME \
    TESSDATA=$TESSDATA_PREFIX \
    MAX_ITERATIONS=$ITERATIONS

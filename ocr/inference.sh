#!/bin/bash

# Configure paths - change these to match your system
TESSERACT_FOLDER=~/dev/tesseract
TESSDATA=$TESSERACT_FOLDER/tessdata  # Path to tessdata
TESSDATA_PREFIX=$TESSDATA  # Required for the makefile
export TESSDATA_PREFIX

tesseract $1 $2_$1 -l $2 --tessdata-dir $TESSDATA --oem 1

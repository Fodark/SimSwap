#!/bin/bash

#conda activate simswap

searchdir="/media/dvl1/SSD_DATA/wildtrack-dataset/test-frames"

i=0
for f in "${searchdir}"/*
do
    python test_wholeimage_swapmulti.py --isTrain false  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/yiming.png --pic_b_path "$f" --output_path ./output/
    cp "output/result_whole_swapmulti.jpg" "/media/dvl1/SSD_DATA/wildtrack-dataset/r-test-frames/${i}.jpg"
    i=$((i+1))
done
path=../../../../../../TransformKit

$path/magik-transform-tools \
--framework onnx \
--target_device T40 \
--outputpath yolov5s-person-4bit.mk.h \
--inputpath ../runs/train/yolov5s-person-4bit.onnx \
--mean 0,0,0 \
--var 255,255,255 \
--img_width 416 \
--img_height 416 \
--img_channel 3

cp yolov5s-person-4bit.bin ../venus_sample_yolov5s/

../../../../TransformKit/magik-transform-tools \
--framework onnx \
--target_device T40 \
--outputpath ./yolov5s_magik.mk.h \
--inputpath ./yolov5s.onnx \
--mean 0,0,0 \
--var 255.0,255.0,255.0 \
--img_width 640 \
--img_height 640 \
--img_channel 3 \
--input_nodes images \
--output_nodes 640,660,680 \
--post_training_quantization true \
--ptq_config_path ./ptq_yolov5_config.json \
--save_quantize_model true
cp yolov5s_magik.bin  venus_sample_ptq_yolov5s/


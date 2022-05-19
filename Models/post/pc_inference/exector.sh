##Convert yolov5s onnx to pc magik model.
../../../TransformKit/magik-transform-tools \
--framework onnx \
--target_device T40 \
--outputpath yolov5s_magik.mk.h \
--inputpath ../yolov5s/yolov5s.onnx \
--mean 0,0,0 \
--var 255.0,255.0,255.0 \
--img_width 640 \
--img_height 640 \
--img_channel 3 \
--input_nodes images \
--output_nodes 640,660,680 \
--post_training_quantization true \
--ptq_config_path ptq_yolov5_config.json \
--save_quantize_model true

##Compile pc inferece code, then execute pc magik model.
make clean
make -j12
./pc_inference_bin  ./save-magik/model_quant.mgk fall_1054_sys.jpg 640 384

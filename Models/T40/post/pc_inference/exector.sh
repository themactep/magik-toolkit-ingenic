cd ../yolov5s_demo/
sh run_debug.sh
cd -
make clean
make -j12
./pc_inference_bin  ../yolov5s_demo/save-magik/model_quant.mgk fall_1054_sys.jpg 640 384

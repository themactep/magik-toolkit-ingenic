yolov5s model is used  to detect objects in images.
It will transform model from onnx to magik, complete post-quantization and generate 't40_graph_yolov5s_ptq.bin'.

<!--
model: ./yolov5s.onnx
quant_img_path: ./yolov5-20/
input node name: images
output node name: 640,660,680
onnxModel : 53.07%
magikModel: 53.07%
PTQ(KL_ABSMAX,12): 52.53%
-->


***1. transform and post-quant***
===========================================================
```bash
sh run.sh

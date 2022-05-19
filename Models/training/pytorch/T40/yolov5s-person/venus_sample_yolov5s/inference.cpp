/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : model_run.cc
 * Authors     : klyu
 * Create Time : 2020-10-28 12:22:44 (CST)
 * Description :
 *
 */

#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#define TIME
#ifdef TIME
#include <sys/time.h>
#endif



#ifdef VENUS_PROFILE
#define RUN_CNT 10
#else
#define RUN_CNT 1
#endif

#ifdef VENUS_DEBUG
#include "img_input.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

using namespace std;
using namespace magik::venus;

struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};

void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

void trans_coords(std::vector<magik::venus::ObjBbox_t> &in_boxes, PixelOffset &pixel_offset,float scale){
    
    printf("pad_x:%d pad_y:%d scale:%f \n",pixel_offset.left,pixel_offset.top,scale);
    for(int i = 0; i < (int)in_boxes.size(); i++) {
        in_boxes[i].box.x0 = (in_boxes[i].box.x0 - pixel_offset.left) / scale;
        in_boxes[i].box.x1 = (in_boxes[i].box.x1 - pixel_offset.left) / scale;
        in_boxes[i].box.y0 = (in_boxes[i].box.y0 - pixel_offset.top) / scale;
        in_boxes[i].box.y1 = (in_boxes[i].box.y1 - pixel_offset.top) / scale;
    }
}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h);

int main(int argc, char **argv) {
    /*set*/
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
            printf("warning: could not set CPU affinity, continuing...\n");
    }

#ifdef VENUS_DEBUG
    int ret = 0;
    if (argc != 4)
    {
        printf("%s model_path w h\n", argv[0]);
        exit(0);
    }
	int in_w = atoi(argv[2]), in_h = atoi(argv[3]);

    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }

    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);
    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());

    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);
    int data_cnt = 1;
    for (auto i : input->shape()) 
    {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

    for (int j = 0; j < data_cnt; j++) {
        indata[j] = image[j];
    }
    test_net->run();

#else

    int ret = 0;
    if (argc != 3)
    {
        printf("%s model_path img_path\n", argv[0]);
        exit(0);
    }

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int in_w = 640, in_h = 384;

    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;
    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        exit(0);
    }
    std::unique_ptr<venus::BaseNet> test_net;
    test_net = venus::net_create(TensorFormat::NHWC);

    std::string model_path = argv[1];
    ret = test_net->load_model(model_path.c_str());
    cv::Mat image;
    image = cv::imread(argv[2]); // BGR format
    
    cv::Mat rgba_img;
    cv::cvtColor(image, rgba_img, cv::COLOR_BGR2RGBA);
    ori_img_w = image.cols;
    ori_img_h = image.rows;
    printf("w:%d h:%d\n", ori_img_w, ori_img_h);

    magik::venus::shape_t temp_inshape;
    temp_inshape.push_back(1);
    temp_inshape.push_back(ori_img_h);
    temp_inshape.push_back(ori_img_w);
    temp_inshape.push_back(4);
    venus::Tensor input_tensor(temp_inshape);
    uint8_t *temp_indata = input_tensor.mudata<uint8_t>();
    int src_size = int(ori_img_h * ori_img_w * 4);

    for (int i = 0; i < src_size-3; i+=4) { // rgba -> alpha channel set zero
        rgba_img.data[i+3] = 0;
    }

    magik::venus::memcopy((void*)temp_indata, (void*)(rgba_img.data), src_size * sizeof(uint8_t));

    printf("ori_image w,h: %d ,%d \n",ori_img_w,ori_img_h);
    input = test_net->get_input(0);
    magik::venus::shape_t rgba_input_shape = input->shape();
    printf("model-->%d ,%d %d \n",rgba_input_shape[1], rgba_input_shape[2], rgba_input_shape[3]);
    input->reshape({1, in_h, in_w , 4});
    uint8_t *indata = input->mudata<uint8_t>();
    std::cout << "input shape:" << std::endl;
    printf("-->%d %d \n",in_h, in_w);

    float scale_x = (float)in_w/(float)ori_img_w;
    float scale_y = (float)in_h/(float)ori_img_h;
    scale = scale_x < scale_y ? scale_x:scale_y;  //min scale
    printf("scale---> %f\n",scale);
    int valid_dst_w = (int)(scale*ori_img_w);
    if (valid_dst_w % 2 == 1)
        valid_dst_w = valid_dst_w + 1;
    int valid_dst_h = (int)(scale*ori_img_h);
    if (valid_dst_h % 2 == 1)
    {
        valid_dst_h = valid_dst_h + 1;
    }

    int dw = in_w - valid_dst_w;
    int dh = in_h - valid_dst_h;
    
    pixel_offset.top = int(round(float(dh)/2 - 0.1));
    pixel_offset.bottom = int(round(float(dh)/2 + 0.1));
    pixel_offset.left = int(round(float(dw)/2 - 0.1));
    pixel_offset.right = int(round(float(dw)/2 + 0.1));
    
    check_pixel_offset(pixel_offset);
    printf("resize padding over: \n");
    printf("resize valid_dst, w:%d h %d\n",valid_dst_w,valid_dst_h);
    printf("padding info top :%d bottom %d left:%d right:%d \n",pixel_offset.top,pixel_offset.bottom,pixel_offset.left,pixel_offset.right);


    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.in_layout = magik::venus::ChannelLayout::RGBA;
    param.out_layout = magik::venus::ChannelLayout::RGBA;
    magik::venus::TensorFormat src_format = magik::venus::TensorFormat::NHWC;
    magik::venus::TensorFormat dst_format = magik::venus::TensorFormat::NHWC;
    warp_resize(input_tensor, *input, &param);

#ifdef TIME
    struct timeval tv; 
    uint64_t time_last;
    double time_ms;
#endif

#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif
    for(int i = 0 ; i < RUN_CNT; i++)
    {
        test_net->run();
    }
#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    time_ms = time_last*1.0/1000;
    printf("test_net run time_ms:%fms\n", time_ms);
#endif


#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec;
#endif

    std::unique_ptr<const venus::Tensor> out0 = test_net->get_output(0);
    std::unique_ptr<const venus::Tensor> out1 = test_net->get_output(1);
    std::unique_ptr<const venus::Tensor> out2 = test_net->get_output(2);

    auto shape0 = out0->shape();
    auto shape1 = out1->shape();
    auto shape2 = out2->shape();

    int shape_size0 = shape0[0] * shape0[1] * shape0[2] * shape0[3];
    int shape_size1 = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    int shape_size2 = shape2[0] * shape2[1] * shape2[2] * shape2[3];

    venus::Tensor temp0(shape0);
    venus::Tensor temp1(shape1);
    venus::Tensor temp2(shape2);

    float* p0 = temp0.mudata<float>();
    float* p1 = temp1.mudata<float>();
    float* p2 = temp2.mudata<float>();

    memcopy((void*)p0, (void*)out0->data<float>(), shape_size0 * sizeof(float));
    memcopy((void*)p1, (void*)out1->data<float>(), shape_size1 * sizeof(float));
    memcopy((void*)p2, (void*)out2->data<float>(), shape_size2 * sizeof(float));
   
    std::vector<venus::Tensor> out_res;
    out_res.push_back(temp0);
    out_res.push_back(temp1);
    out_res.push_back(temp2);

    std::vector<magik::venus::ObjBbox_t>  output_boxes;
    output_boxes.clear();
    generateBBox(out_res, output_boxes, in_w, in_h);
    trans_coords(output_boxes, pixel_offset, scale);
#ifdef TIME
    gettimeofday(&tv, NULL);
    time_last = tv.tv_sec*1000000 + tv.tv_usec - time_last;
    time_ms = time_last*1.0/1000;
	printf("post net time_ms:%fms\n", time_ms);
#endif

    for (int i = 0; i < int(output_boxes.size()); i++) 
    {
        auto person = output_boxes[i];
        printf("box:   ");
        printf("%d ",(int)person.box.x0);
        printf("%d ",(int)person.box.y0);
        printf("%d ",(int)person.box.x1);
        printf("%d ",(int)person.box.y1);
        printf("%.2f ",person.score);
        cv::rectangle(image,cvPoint((int)person.box.x0,(int)person.box.y0),cvPoint((int)person.box.x1, (int)person.box.y1),cv::Scalar(255,0,0),3,1,0);
        printf("\n");
    }
    cv::imwrite("result.jpg", image);

    ret = venus::venus_deinit();
    if (0 != ret) 
    {
        fprintf(stderr, "venus deinit failed.\n");
        return ret;
    }
    return 0;
#endif

}

void generateBBox(std::vector<venus::Tensor> out_res, std::vector<magik::venus::ObjBbox_t>& candidate_boxes, int img_w, int img_h)
{
  float person_threshold = 0.3;
  int classes = 1;
  float nms_threshold = 0.6;
  std::vector<float> strides = {8.0, 16.0, 32.0};
  int box_num = 3;
  std::vector<float> anchor = {10,13,  16,30,  33,23, 30,61,  62,45,  59,119, 116,90,  156,198,  373,326};

  std::vector<magik::venus::ObjBbox_t>  temp_boxes;
  venus::generate_box(out_res, strides, anchor, temp_boxes, img_w, img_h, classes, box_num, person_threshold, magik::venus::DetectorType::YOLOV5);
  venus::nms(temp_boxes, candidate_boxes, nms_threshold); 
}

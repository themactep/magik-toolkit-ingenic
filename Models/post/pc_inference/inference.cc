/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : magikExecutor.cc
 * Authors    : lqwang
 * Create Time: 2021-04-08:09:16:52
 * Description:
 *
 */

#include "args_parser.h"
#include "graph_executor.h"
#include "common.h"
#include "tensor.h"
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};
using namespace cv;

int resize_uniform(Mat &src, Mat &dst, Size dst_size, object_rect &effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = Mat(Size(dst_w, dst_h), CV_8UC3, Scalar(0));

    float ratio_src = w*1.0 / h;
    float ratio_dst = dst_w*1.0 / dst_h;

    int tmp_w=0;
    int tmp_h=0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w*1.0 / w) * h);
    } else if (ratio_src < ratio_dst){
        tmp_h = dst_h;
        tmp_w = floor((dst_h*1.0 / h) * w);
    } else {
        resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    Mat tmp;
    resize(src, tmp, Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) { //高对齐，宽没对齐
        int index_w = floor((dst_w - tmp_w) / 2.0);
        std::cout << "index_w: " << index_w << std::endl;
        for (int i=0; i<dst_h; i++) {
            memcpy(dst.data+i*dst_w*3 + index_w*3, tmp.data+i*tmp_w*3, tmp_w*3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else if (tmp_h != dst_h) { //宽对齐， 高没有对齐
        int index_h = floor((dst_h - tmp_h) / 2.0);
        std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data+index_h*dst_w*3, tmp.data, tmp_w*tmp_h*3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    } else {
        printf("error\n");
    }
    return 0;
}


using namespace magik::transformkit::magikexecutor;

int main(int argc, char** argv) {

    if (argc != 5){
        printf("%s model_path img_path in_w in_h\n", argv[0]);
        exit(0);
    }
    std::string model_path = argv[1];
    std::string img_path = argv[2];
    int in_w = atoi(argv[3]);
    int in_h = atoi(argv[4]);
    cv::Mat image_src;
    cv::Mat image;
    image_src = cv::imread(img_path);
    object_rect res_area;
    (void)resize_uniform(image_src, image, Size(in_w, in_h), res_area);
    int ori_img_w = image.cols;
    int ori_img_h = image.rows;
    printf("ori_w:%d ori_h:%d\n", ori_img_w, ori_img_h);
    cv::resize(image, image, cv::Size(in_w, in_h));
    //imwrite("out.jpg", image);
    printf("resize_w:%d resize_h:%d\n", image.cols, image.rows);

    magik::transformkit::magikexecutor::GraphExecutor graphExecutor(model_path);
    graphExecutor.set_inplace(false);
    
    std::vector<std::string> input_names = graphExecutor.get_input_names();
    std::vector<std::string> output_names = graphExecutor.get_output_names();
    Tensor* tensor = new Tensor({1, in_h, in_w, 3}, Tensor::DataType::DT_FLOAT);
    tensor->set_name(input_names[0]);

    for (int i = 0; i < tensor->total(); ++i) {
        tensor->mutable_data<float>(i) = (float)image.data[i];
    }
    graphExecutor.set_input(tensor);
    for (int i = 0; i < 1; ++i) {
        std::cout<<"start inference ....................................................."<<std::endl;
        graphExecutor.work();
        const Tensor* tensor_res = graphExecutor.get_node_tensor(output_names[0]);
        if (tensor_res == NULL) {
            exit(0);
        }
        //tensor_res->total()
        for(int j = 0 ; j < 10; ++j)
            printf("%f ", tensor_res->data<float>(j));
        printf("\n");
        std::cout<<"end inference....................................................."<<std::endl;

    }
    delete tensor;
    return 0;
}

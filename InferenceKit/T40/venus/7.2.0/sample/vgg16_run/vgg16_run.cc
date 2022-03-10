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

//#define MEMORY_MODEL
#ifdef MEMORY_MODEL
#include "magik_model_t40_graph_vgg16.mk.h"
#endif
#include "class12_32x32.h"
#include "venus.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

#ifdef VENUS_PROFILE
#ifdef VENUS_PMON
#define RUN_CNT 1
#else
#define RUN_CNT 10 // run at least 10 times, 5 of which are used for warm up
#endif
#else
#define RUN_CNT 1
#endif
int main(int argc, char **argv) {
    int ret = 0;

    /* set cpu affinity */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
        fprintf(stderr, "set cpu affinity failed, %s\n", strerror(errno));
        return -1;
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        return ret;
    }

    std::string model_path = "./t40_graph_vgg16.bin";
    std::unique_ptr<venus::BaseNet> test_net = venus::net_create();
#ifdef MEMORY_MODEL
    {
        MagikModelT40GraphVgg16Mk model;
        ret = test_net->load_model((const char *)model.data, true);
    }
#else
    ret = test_net->load_model(model_path.c_str());
#endif

    printf("Load model over.\n");
    size_t mem_size;
    ret = test_net->get_forward_memory_size(mem_size);
    std::cout << "Forward memory size: " << mem_size << std::endl;
    std::unique_ptr<venus::Tensor> input = test_net->get_input(0);
    std::cout << "input shape:" << std::endl;
    uint8_t *indata = input->mudata<uint8_t>();
    if (IS_ALIGN_64(indata) != 0) {
        fprintf(stderr, "input addr not align to 64 bytes.\n");
        return -1;
    }
    int data_cnt = 1;
    for (auto i : input->shape()) {
        std::cout << i << ",";
        data_cnt *= i;
    }
    std::cout << std::endl;

    for (int j = 0; j < data_cnt; j++) {
        indata[j] = image[j];
    }
    for (int i = 0; i < RUN_CNT; i++) {
        test_net->run();
    }
    auto out = test_net->get_output(0);
    std::cout << "output Shape: " << std::endl;
    int out_size = 1;
    for (auto i : out->shape()) {
        std::cout << i << ",";
        out_size *= i;
    }
    std::cout << std::endl;
    const float *out_ptr = out->data<float>();
    printf("out_ptr=%p\n", out_ptr);
    std::string out_name = "vgg16_out.bin";
    std::ofstream owput;
    owput.open(out_name, std::ios::binary);
    if (!owput || !owput.is_open() || !owput.good()) {
        owput.close();
        return -1;
    }
    owput.write((char *)out_ptr, out->shape()[0] * out->shape()[1] * out->shape()[2] *
                                     out->shape()[3] * sizeof(float));
    owput.close();
    // ret = venus::venus_deinit();
    // if (0 != ret) {
    //     fprintf(stderr, "venus deinit failed.\n");
    //     return ret;
    // }

    return 0;
}

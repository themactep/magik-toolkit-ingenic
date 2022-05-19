/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : common_type.h
 * Authors     : lzwang
 * Create Time : 2021-10-14 18:14:03 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_UTILS_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_UTILS_H__

#include "common_type.h"
#include <stdint.h>
#include <vector>

namespace magik {
namespace venus {

enum class VENUS_API BsPadType : int {
    NONE = -1,        /*no pad value*/
    BOTTOM_RIGHT = 0, /*fill pad value in bottom and right of output image*/
    SYMMETRY = 1,     /*fill pad value around the output image symmetrically*/
};

struct VENUS_API BsBaseParam {
    ChannelLayout in_layout;  /*input format:NV12 BGRA RGBA*/
    ChannelLayout out_layout; /*output format:NV12 BGRA RGBA*/
    /*coef_off_enable: enable or disable coef and offset in color format convert
     *true: calculate by coef and offset
     *false: calculate by default parameters*/
    bool coef_off_enable = false;
    uint32_t coef[9];   /*nv12 to bgra param*/
    uint32_t offset[2]; /*nv12 to bgra pixel offset*/
};

struct VENUS_API BsExtendParam : public BsBaseParam {
    BsPadType pad_type; /*BOTTOM_RIGHT or SYMMETRY*/
    uint8_t pad_val;    /*value of pad*/
};

struct VENUS_API BsCommonParam : public BsExtendParam {
    int input_height;      /*input image height*/
    int input_width;       /*input image width*/
    int input_line_stride; /*input line stride*/
};

struct VENUS_API GetZoneParam {
    std::vector<int32_t> input_shape;  /*HWC*/
    std::vector<int32_t> output_shape; /*HWC*/
    float threshold;
};
} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_UTILS_H__ */

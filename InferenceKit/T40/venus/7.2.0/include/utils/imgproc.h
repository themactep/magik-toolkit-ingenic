/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : imgproc.h
 * Authors     : klyu
 * Create Time : 2020-12-24 14:59:36 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__
#include "common/common_utils.h"
#include "core/tensor.h"
#include "core/type.h"
#include <vector>
namespace magik {
namespace venus {

enum class PaddingType : int {
    /*no padding*/
    NONE = -1,
    /*
     * (1). BGR0, eg: src_shape=[1, 3, 5, 4], dst_shape=[1, 5, 7, 4]:
     * padval = 0;
     * +--------------------+       +----------------------------+
     * |BGR0BGR0BGR0BGR0BGR0|       |BGR0BGR0BGR0BGR0BGR000000000|
     * |BGR0BGR0BGR0BGR0BGR0| ====> |BGR0BGR0BGR0BGR0BGR000000000|
     * |BGR0BGR0BGR0BGR0BGR0|       |BGR0BGR0BGR0BGR0BGR000000000|
     * +--------------------+       |0000000000000000000000000000|
     *                              |0000000000000000000000000000|
     *                              +----------------------------+
     *
     * (2). NV12, eg: src_shape=[1, 4, 6, 1], dst_shape=[1, 6, 8, 1]:
     * padval_y = 16;
     * padval_uv = 128;
     * +------+                     +-------------------------------+
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|       =====>        |Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  16  16  16  16  16  16 |
     * |UVUVUV|                     |16  16  16  16  16  16  16  16 |
     * +------+                     |U   V   U   V   U   V   128 128|
     *                              |U   V   U   V   U   V   128 128|
     *                              |128 128 128 128 128 128 128 128|
     *                              +-------------------------------+
     */
    BOTTOM_RIGHT = 0,

    /*
     * (1). BGR0, eg: src_shape=[1, 3, 5, 4], dst_shape=[1, 5, 7, 4]:
     * padval = 0;
     * +--------------------+        +----------------------------+
     * |BGR0BGR0BGR0BGR0BGR0|        |0000000000000000000000000000|
     * |BGR0BGR0BGR0BGR0BGR0| =====> |0000BGR0BGR0BGR0BGR0BGR00000|
     * |BGR0BGR0BGR0BGR0BGR0|        |0000BGR0BGR0BGR0BGR0BGR00000|
     * +--------------------+        |0000BGR0BGR0BGR0BGR0BGR00000|
     *                               |0000000000000000000000000000|
     *                               +----------------------------+
     *
     * (2). NV12, eg: src_shape=[1, 4, 6, 1], dst_shape=[1, 8, 10, 1]:
     * padval_y = 16;
     * padval_uv = 128;
     * +------+                     +---------------------------------------+
     * |YYYYYY|                     |16  16  16  16  16  16  16  16  16  16 |
     * |YYYYYY|                     |16  16  16  16  16  16  16  16  16  16 |
     * |YYYYYY|       =====>        |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |YYYYYY|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * |UVUVUV|                     |16  16  Y   Y   Y   Y   Y   Y   16  16 |
     * +------+                     |16  16  16  16  16  16  16  16  16  16 |
     *                              |16  16  16  16  16  16  16  16  16  16 |
     *                              |128 128 128 128 128 128 128 128 128 128|
     *                              |128 128 U   V   U   V   U   V   128 128|
     *                              |128 128 U   V   U   V   U   V   128 128|
     *                              |128 128 128 128 128 128 128 128 128 128|
     *                              +---------------------------------------+
     */
    SYMMETRY = 1
};
enum class AddressLocate : int {
    NMEM_VIRTUAL = 0,  // virtual address in nmem
    RMEM_PHYSICAL = 1, // physical address in rmem
};
/*
 * color space conversion
 * input: input tensor, format nv12
 * output: output tensor, format bgra
 */
VENUS_API int warp_covert_nv2bgr(const Tensor &input, Tensor &output);
/*
 * resize tensor
 * input: input tensor, format bgra
 * output: output tensor, format bgra
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int warp_resize_bgra(const Tensor &input, Tensor &output, PaddingType padtype,
                               uint8_t padval);
/*
 * resize tensor
 * input: input tensor, format nv12
 * output: output tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int warp_resize_nv12(const Tensor &input, Tensor &output, bool cvtbgra,
                               PaddingType padtype, uint8_t padval);
/*
 * crop and resize tensor
 * input: input tensor, format bgra
 * output: output tensor
 * boxes: boxes for input tensor crop
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int crop_resize_bgra(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Bbox_t> &boxes, PaddingType padtype, uint8_t padval);
/*
 * crop and resize tensor
 * input: input tensor, format nv12
 * output: output tensor
 * boxes: boxes for input tensor crop
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 */
VENUS_API int crop_resize_nv12(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Bbox_t> &boxes, bool cvtbgra, PaddingType padtype,
                               uint8_t padval);
/*
 * resize input nv12 format data
 * input: address of input nv12 format data
 * output: output tensor
 * img_h: input height
 * img_w: input width
 * line_stride: input line stride(byte)
 * cvtbgra: if true, ouput format is bgra, else nv12
 * padtype: BOTTOM_RIGHT or SYMMETRY
 * padval: value of pad
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 */
VENUS_API int common_resize_nv12(const void *input, Tensor &output, int img_h, int img_w,
                                 int line_stride, bool cvtbgra, PaddingType padtype, uint8_t padval,
                                 AddressLocate input_locate);
/*
 * affine tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: affine matrix tensor
 */
VENUS_API int warp_affine_bgra(const Tensor &input, Tensor &output, Tensor &matrix);

/*
 * affine tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: tensor of affine matrix
 * cvtbgra: if true, ouput format is bgra, else nv12
 */
VENUS_API int warp_affine_nv12(const Tensor &input, Tensor &output, Tensor &matrix, bool cvtbgra);

/*
 * crop and affine tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: affine matrix tensor
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_affine_bgra(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes);

/*
 * crop and affine tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: tensor of affine matrix
 * cvtbgra: if true, ouput format is bgra, else nv12
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_affine_nv12(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, bool cvtbgra,
                               std::vector<Bbox_t> &boxes);

/*
 * perspective tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: perspective matrix tensor
 */
VENUS_API int warp_perspective_bgra(const Tensor &input, Tensor &output, Tensor &matrix);

/*
 * perspective tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: perspective matrix tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 */
VENUS_API int warp_perspective_nv12(const Tensor &input, Tensor &output, Tensor &matrix,
                                    bool cvtbgra);
/*
 * crop and perspective tensor
 * input: input tensor, format bgra
 * output: output tensor,format bgra
 * matrix: perspective matrix tensor
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_perspective_bgra(const Tensor &input, std::vector<Tensor> &output,
                                    std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes);

/*
 * crop and perspective tensor
 * input: input tensor, format nv12
 * output: output tensor
 * matrix: perspective matrix tensor
 * cvtbgra: if true, ouput format is bgra, else nv12
 * boxes: boxes for input tensor crop
 */
VENUS_API int crop_perspective_nv12(const Tensor &input, std::vector<Tensor> &output,
                                    std::vector<Tensor> &matrix, bool cvtbgra,
                                    std::vector<Bbox_t> &boxes);

/*
 * similar transform
 * input_src : input tensor, source perspective matrix
 * input_dst : input tensor, dst perspective matrix
 * output : output tensor, similar transform result matrix
 * retval : 0, success, <0 failed
 */
VENUS_API int similar_transform(Tensor &input_src, Tensor &input_dst, Tensor &output);

/*
 * affine transform
 * input_src : input tensor, source affine matrix
 * input_dst : input tensor, dst affine matrix
 * output : output tensor, affine transform result matrix
 * retval : 0, success, <0 failed
 */
VENUS_API int get_affine_transform(Tensor &input_src, Tensor &input_dst, Tensor &output);

/*********new version**************/
/*if input chn equal 4, output format must equal input format*/
/*
 * resize tensor
 * input: input tensor
 * output: output tensor
 * param: resize param
 */
VENUS_API int warp_resize(const Tensor &input, Tensor &output, BsExtendParam *param);

/*
 * crop and resize tensor
 * input: input tensor
 * output: output tensor
 * boxes: boxes for input tensor crop
 * param: resize param
 */
VENUS_API int crop_resize(const Tensor &input, std::vector<Tensor> &output,
                          std::vector<Bbox_t> &boxes, BsExtendParam *param);
/*
 * affine tensor
 * input: input tensor
 * output: output tensor
 * matrix: affine matrix tensor
 * param: affine param
 */
VENUS_API int warp_affine(const Tensor &input, Tensor &output, Tensor &matrix,
                          BsExtendParam *param);
/*
 * crop and affine tensor
 * input: input tensor
 * output: output tensor
 * matrix: affine matrix tensor
 * boxes: boxes for input tensor crop
 * param: affine param
 */
VENUS_API int crop_affine(const Tensor &input, std::vector<Tensor> &output,
                          std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes,
                          BsExtendParam *param);
/*
 * perspective tensor
 * input: input tensor
 * output: output tensor
 * matrix: perspective matrix tensor
 * param: perspective param
 */
VENUS_API int warp_perspective(const Tensor &input, Tensor &output, Tensor &matrix,
                               BsExtendParam *param);
/*
 * perspective tensor
 * input: input tensor
 * output: output tensor
 * matrix: perspective matrix tensor
 * param: perspective param
 */
VENUS_API int crop_perspective(const Tensor &input, std::vector<Tensor> &output,
                               std::vector<Tensor> &matrix, std::vector<Bbox_t> &boxes,
                               BsExtendParam *param);
/*
 * resize input nv12 format data
 * input: address of input nv12 format data
 * output: output tensor
 * input_locate: NMEM_VIRTUAL or RMEM_PHYSICAL
 * param: resize param
 */
VENUS_API int common_resize(const void *input, Tensor &output, AddressLocate input_locate,
                            BsCommonParam *param);

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_IMGPROC_H__ */

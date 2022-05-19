/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : type.h
 * Authors     : klyu
 * Create Time : 2020-10-28 11:57:38 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TYPE_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TYPE_H__
#include "common/common_type.h"

namespace magik {
namespace venus {
enum class VENUS_API ShareMemoryMode : int {
    DEFAULT = 0,
    SHARE_ONE_THREAD = 1,
    SET_FROM_EXTERNAL = 2
};

enum class VENUS_API TensorFormat : int {
    /*
     * 1. image format.
     * (1). BGR0, eg: shape=[1, 3, 6, 4]:
     * +------------------------+
     * |BGR0BGR0BGR0BGR0BGR0BGR0|
     * |BGR0BGR0BGR0BGR0BGR0BGR0|
     * |BGR0BGR0BGR0BGR0BGR0BGR0|
     * +------------------------+
     * (2). RGB0, eg: shape=[1, 3, 6, 4]:
     * +------------------------+
     * |RGB0RGB0RGB0RGB0RGB0RGB0|
     * |RGB0RGB0RGB0RGB0RGB0RGB0|
     * |RGB0BGR0BGR0RGB0RGB0RGB0|
     * +------------------------+
     * (3). GRAY, eg: shape=[1, 3, 24, 1]:
     * +------------------------+
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * +------------------------+
     * 2. feature map and others.
     */
    NHWC = 1,
    /*
     * image format for nv12.
     * eg: shape=[1, 4, 24, 1]:
     * +------------------------+
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * |YYYYYYYYYYYYYYYYYYYYYYYY|
     * +------------------------+
     * |UVUVUVUVUVUVUVUVUVUVUVUV|
     * |UVUVUVUVUVUVUVUVUVUVUVUV|
     * +------------------------+
     */
    NV12 = 5
};

typedef struct {
    float x0;
    float y0;
    float x1;
    float y1;
} Bbox_t;

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_CORE_TYPE_H__ */

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

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__

#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define VENUS_API __attribute__((dllexport))
#else
#define VENUS_API __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define VENUS_API __attribute__((dllimport))
#else
#define VENUS_API __declspec(dllimport)
#endif
#endif
#else
#if __GNUC__ >= 4
#define VENUS_API __attribute__((visibility("default")))
#else
#define VENUS_API
#endif
#endif
#include <stdint.h>

namespace magik {
namespace venus {

enum class VENUS_API ChannelLayout : int {
    NONE = -1,
    NV12 = 0,
    BGRA = 1,
    RGBA = 2,
    ARGB = 3,
    RGB = 4,
    GRAY = 5
};
enum class VENUS_API TransformType : int {
    NONE = -1,
    NV12_NV12 = 0,
};

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_COMMON_COMMON_TYPE_H__ */

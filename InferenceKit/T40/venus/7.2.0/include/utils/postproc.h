/*
 *         (C) COPYRIGHT Ingenic Limited
 *              ALL RIGHT RESERVED
 *
 * File        : postproc.h
 * Authors     : klyu
 * Create Time : 2020-12-24 15:02:22 (CST)
 * Description :
 *
 */

#ifndef __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__
#define __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__
#include "core/type.h"
#include "venus.h"

namespace magik {
namespace venus {
typedef struct {
    Bbox_t box;
    float score;
    int class_id;
} ObjBbox_t;

enum class NmsType { HARD_NMS = 0, SOFT_NMS = 1 };
enum class DetectorType { YOLOV3 = 0, YOLOV5 = 1 };

/*
 * post-process NMS
 * type=0: hard nms, type=1: soft nms
 */
VENUS_API void nms(std::vector<ObjBbox_t> &input, std::vector<ObjBbox_t> &output,
                   float nms_threshold = 0.3, NmsType type = NmsType::HARD_NMS);
/*
 * post-process Generate Candidate Boxes
 * features : input feature
 * candidate_boxes : output candidate boxes
 * img_w,img_h : image width,height
 * classes : class number
 * box_num : the boxes number of each point
 * box_score_threshold : filter boxes based on box_score
 */
VENUS_API void generate_box(std::vector<Tensor> &features, std::vector<float> &strides,
                            std::vector<float> &anchor, std::vector<ObjBbox_t> &candidate_boxes,
                            int img_w, int img_h, int classes, int box_num,
                            float box_score_threshold,
                            DetectorType detector_type = DetectorType::YOLOV3);

} // namespace venus
} // namespace magik
#endif /* __MAGIK_INFERENCEKIT_VENUS_INCLUDE_UTILS_POSTPROC_H__ */

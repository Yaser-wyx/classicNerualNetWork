import torch
import torch.nn as nn
from torchvision.ops import nms

from detection.Faster_RCNN.utils.bbox_tools import amend_bbox, generate_anchor_template, generate_raw_image_anchor
from detection.Faster_RCNN.utils.config import cfg


class ProposalLayer(nn.Module):
    # 用于生成预测框
    def __init__(self, feat_stride, anchor_scales=None, anchor_ratios=None, config=None):
        super().__init__()
        if config is None:
            config = cfg
        self.feat_stride = feat_stride
        self.config = config
        # 生成anchor模板
        self._anchors_template = \
            torch.from_numpy(generate_anchor_template(ratios=anchor_ratios, scales=anchor_scales)).float()
        self._anchors = None

    def forward(self, inputs, img_size, offset_scale_pred, bbox_f_pred, cfg_key):
        batch, channel, feature_height, feature_width = inputs.shape
        # 获取配置文件数据
        pre_nms_topN = self.config[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = self.config[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = self.config[cfg_key].RPN_NMS_THRESH
        min_size = self.config[cfg_key].RPN_MIN_SIZE
        # 生成原始图片上的anchor
        self._anchors = generate_raw_image_anchor(feature_height=feature_height, feature_width=feature_width,
                                                  anchors_template=self._anchors_template,
                                                  feat_stride=self.feat_stride)
        rois = []
        rois_idx = []
        h, w = img_size[0], img_size[1]
        for i in range(batch):
            bbox_f_pred_of_i = bbox_f_pred[i]
            # 1. 使用预测的比例和偏移量对在原始图像上生成的anchor进行修正
            roi = amend_bbox(self._anchors, offset_scale_pred[i])
            # roi shape: anchor_num * 4 {x_min,y_min,x_max,y_max}
            # 2. 裁剪预测用的bbox在图像的宽高范围内
            roi[:, slice(0, 4, 2)].clip_(min=0, max=w)  # 裁剪宽度
            roi[:, slice(1, 4, 2)].clip_(min=0, max=h)  # 裁剪高度
            # 3. 剔除掉过小的anchor
            keep_idx = self._filter_roi(roi, min_size)
            roi = roi[keep_idx, :]  # 筛选出roi
            bbox_f_pred_of_i = bbox_f_pred_of_i[keep_idx]  # 筛选出预测为前景框的评分
            # 4. 对前景预测的结果进行排序，选择pre_nms_topN个roi
            order = torch.ravel(bbox_f_pred_of_i).argsort(descending=True)  # 降序排序
            if 0 < pre_nms_topN < len(bbox_f_pred_of_i):
                order = order[:pre_nms_topN]
            roi = roi[order, :]
            bbox_f_pred_of_i = bbox_f_pred_of_i[order]
            # https://blog.csdn.net/lz867422770/article/details/100019587
            # 5. 进行极大值抑制处理，对于IoU大于nms_thresh的进行合并
            keep_idx = nms(roi, bbox_f_pred_of_i, nms_thresh)

            if 0 < post_nms_topN < len(keep_idx):
                keep_idx = keep_idx[:post_nms_topN]
            roi = roi[keep_idx]
            rois.append(roi)
            # 每一个roi对应的batch_idx
            # 因为后续需要将rois展平，而一旦展平后，就没法确定哪些roi属于一组的了，
            # 所以需要一个辅助数组rois_idx，帮助在1维的情况确定对应位置的roi所属的batch
            # 注：本版本暂不支持batch>1，后续会加入支持
            # TODO: support for batch>1
            rois_idx.append(i * torch.ones(len(roi), dtype=torch.float32))
        rois = torch.cat(rois)  # 将rois数组进行拼接，因为当前版本仅支持batch为1的情况，所以用处相当于将rois转为tensor格式
        rois_idx = torch.cat(rois_idx)
        return rois, rois_idx, self._anchors

    def _filter_roi(self, roi, min_size):
        hs = roi[:, 3] - roi[:, 1]
        ws = roi[:, 2] - roi[:, 0]
        keep_idx = torch.where((hs > min_size) & (ws > min_size))  # 注：torch.where返回只有一个元素的tuple
        return keep_idx[0]  # 取出tuple的第一个

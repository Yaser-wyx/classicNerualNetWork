import torch.nn as nn


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std

    def forward(self, inputs):
        img_size = inputs.shape[2:]  # h*w
        feature_map = self.extractor(inputs)
        rois, rois_idx, anchors, offset_scale_pred, rpn_bbox_bf_pred = self.rpn(feature_map, img_size=img_size)
        roi_cls_pred, roi_bbox_pred = self.head(feature_map, rois, rois_idx)
        return roi_bbox_pred, roi_cls_pred, rois, rois_idx

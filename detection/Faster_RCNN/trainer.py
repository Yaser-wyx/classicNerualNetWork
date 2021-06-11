from collections import namedtuple

import torch.nn as nn
import torch
import torch.nn.functional as F
from detection.Faster_RCNN.target_layer import AnchorTargetLayer, ProposalTargetLayer
from detection.Faster_RCNN.utils.config import cfg
from detection.Faster_RCNN.utils.model_helper import cal_reg_loss

LossTuple = namedtuple('LossTuple',
                       ['rpn_reg_loss',
                        'rpn_cls_loss',
                        'roi_reg_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn, optimizer, device):
        super().__init__()
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = cfg.INIT.RPN_SIGMA
        self.roi_sigma = cfg.INIT.ROI_SIGMA
        self.device = device
        assert optimizer is not None, "optimizer should not be None"
        self.optimizer = optimizer
        # AnchorTargetLayer用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，
        # 也就是为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
        self.anchor_target_layer = AnchorTargetLayer()  # 服务于RPN部分
        # ProposalCreator是RPN为Fast R-CNN生成RoIs，在训练和测试阶段都会用到。
        # 所以测试阶段直接输进来300个RoIs，而训练阶段会有AnchorTargetCreator的再次干预。
        self.proposal_target_layer = ProposalTargetLayer()  # 服务于RoIHead部分
        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

    def step(self, img, bbox, label):
        loss = self.forward(img, bbox, label)
        loss.total_loss.backward()
        self.optimizer.step()
        return loss

    def forward(self, images, gt_bboxes, gt_labels):
        _, _, H, W = images.shape
        image_size = (H, W)
        # TODO: WHEN SUPPORT FOR BATCH>1, REMOVE IT
        # now is only support for batch=1
        gt_bbox = gt_bboxes[0]
        gt_label = gt_labels[0]
        # 经过VGG16提取特征
        features = self.faster_rcnn.extractor(images)
        # 经过rpn生成roi
        rois, rois_idx, anchors, offset_scale_pred, rpn_bbox_bf_pred = self.faster_rcnn.rpn(features, image_size)
        offset_scale_pred = offset_scale_pred[0]
        rpn_bbox_bf_pred = rpn_bbox_bf_pred[0]
        # 从roi中提取训练样本
        roi_samples, roi_gt_bbox, roi_gt_labels = self.proposal_target_layer(rois, gt_bbox, gt_label,
                                                                             self.loc_normalize_mean,
                                                                             self.loc_normalize_std)
        sample_roi_index = torch.zeros(len(roi_samples), device=self.device)  # TODO REMOVE WHEN SUPPORT FOR BATCH>1
        # 获取每一个预测roi的类别信息以及回归边界框
        # cls_pred shape: sample_num * (k*2) 是sample_num个样例，每一个样例对应K个类别是前景还是后景的概率
        # bbox_pred shape: sample_num * (k*4) 是sample_num个样例，每一个样例对应K个类别的位置
        roi_label_pred, roi_bbox_pred = self.faster_rcnn.head(features, roi_samples, sample_roi_index)
        # --------------------------RPN LOSSES
        rpn_offset_scale_gt, rpn_label_gt = self.anchor_target_layer(gt_bbox, anchors, image_size)
        rpn_label_gt = rpn_label_gt.long()
        # 计算rpn网络对anchor回归预测的loss
        print("计算rpn回归损失")
        rpn_reg_loss = cal_reg_loss(offset_scale_pred, rpn_offset_scale_gt, rpn_label_gt, self.rpn_sigma)
        # 计算rpn网络对于anchor分类(前景还是后景)预测的loss
        rpn_cls_loss = F.cross_entropy(rpn_bbox_bf_pred, rpn_label_gt, ignore_index=-1)
        # --------------------------ROI LOSSES
        # 计算RoI预测的位置损失
        sample_num = roi_bbox_pred.shape[0]
        roi_bbox_pred = roi_bbox_pred.view(sample_num, -1, 4)
        roi_gt_labels = roi_gt_labels.long()
        roi_bbox_pred = roi_bbox_pred[torch.arange(0, sample_num).long(), roi_gt_labels]  # 获取每一个预测roi对应真值的坐标
        print("计算RoI回归损失")

        roi_reg_loss = cal_reg_loss(roi_bbox_pred.contiguous(), roi_gt_bbox, roi_gt_labels, self.roi_sigma)
        # 计算RoI预测的label损失
        roi_cls_loss = F.cross_entropy(roi_label_pred, roi_gt_labels)

        losses = [rpn_reg_loss, rpn_cls_loss, roi_reg_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

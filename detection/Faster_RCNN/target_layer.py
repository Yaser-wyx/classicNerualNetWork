import torch.nn as nn
import torch
import numpy as np

from detection.Faster_RCNN.utils.bbox_tools import cal_IoU, cal_offset_scale, map_data2anchor
from detection.Faster_RCNN.utils.config import cfg


class AnchorTargetLayer(nn.Module):
    # 主要用于训练RPN网络的Anchor生成
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        super().__init__()
        device = cfg.INIT.DEVICE
        self.device = device
        self.n_sample = torch.tensor(n_sample).to(device)
        self.neg_iou_thresh = torch.tensor(neg_iou_thresh).to(device)
        self.pos_iou_thresh = torch.tensor(pos_iou_thresh).to(device)
        self.pos_ratio = torch.tensor(pos_ratio).to(device)

    def forward(self, gt_bbox, anchors, img_size):
        img_h, img_w = img_size
        anchor_num = len(anchors)
        # 计算在图片范围内的anchor
        img_inner_idx = torch.where((anchors[:, 0] >= 0) &
                                    (anchors[:, 1] >= 0) &
                                    (anchors[:, 2] <= img_w) &
                                    (anchors[:, 3] <= img_h))[0]
        anchors = anchors[img_inner_idx]  # 筛选出在图片内的anchor
        # 筛选出符合条件的正例与负例，一共256个，每个最多128个，并给它们附上相应的label
        argmax_iou, label = self._create_label(anchors, gt_bbox)  # shape：(num_of_anchor,)
        # 对于在图片内的anchor计算与gt_bbox的偏移量
        offset_scale = cal_offset_scale(anchors, gt_bbox[argmax_iou])  # shape：(num_of_anchor, 4)
        # 将筛选出的样例数据（label以及offset_scale数据）映射回输入的原始anchor
        label = map_data2anchor(data=label, anchor_idx=img_inner_idx, anchor_num=anchor_num, default_value=-1)
        offset_scale = map_data2anchor(data=offset_scale, anchor_idx=img_inner_idx, anchor_num=anchor_num,
                                       default_value=0)
        return offset_scale, label

    def _create_label(self, anchors, gt_bbox):
        # 给符合条件的anchor打上标签，前景为1，后景为0，-1是默认值
        label = torch.zeros(len(anchors), dtype=torch.int32,device = self.device)
        label.fill_(-1)  # 全部填充为默认值
        argmax_iou, per_anchor_max_iou, max_iou_anchor_idx = self._cal_iou(anchors, gt_bbox)
        # 填充负样本
        label[per_anchor_max_iou <= self.neg_iou_thresh] = 0
        # 填充正样本
        label[max_iou_anchor_idx] = 1  # 对于iou值最大的anchor作为正样本
        label[per_anchor_max_iou >= self.pos_iou_thresh] = 1

        def remove_redundant_label(label, expect_num, target_label):
            # 去除掉多余的标签
            idx = torch.where(label == target_label)[0].float()  # 获取多余标签的索引
            print("idx: ", torch.sum(label[idx.long()] == 1))
            # 判断数量
            if len(idx) > expect_num.item():
                # 随机抽取一些去掉
                num_sample = len(idx) - expect_num
                if torch.is_tensor(num_sample):
                    num_sample = num_sample.item()
                remove_idx_idx = torch.multinomial(idx, num_sample)
                remove_idx = idx[remove_idx_idx].long()
                label[remove_idx] = -1

            return label

        # 去掉多余的正样本
        positive_num = torch.round(self.n_sample * self.pos_ratio)
        label = remove_redundant_label(label, positive_num, 1)

        # 去掉多余的负样本
        negative_num = self.n_sample - torch.sum(label == 1)
        label = remove_redundant_label(label, negative_num, 0)
        num_pos = torch.sum(label == 1)
        num_neg = torch.sum(label == 0)
        num_None = torch.sum(label == -1)

        print("正样本数：", num_pos)
        print("负样本数：", num_neg)
        return argmax_iou, label

    def _cal_iou(self, anchors, gt_bbox):
        # 目标：获取每一个gt_bbox对应的iou最大的anchor
        # 计算每一个anchor与所有的gt_bbox的IOU值
        iou = cal_IoU(anchors, gt_bbox)  # iou shape: num_of_anchors*num_of_gt_bbox
        # 计算每一个gt_bbox与这些anchor最大的IOU是多少
        gt_max_iou = iou.max(dim=0).values
        per_anchor_max_iou = iou.max(dim=1).values  # 获得每一个anchor对应的最大iou值
        argmax_iou = iou.argmax(dim=1)  # 计算每一个anchor对应iou的最大的gt_bbox
        max_iou_anchor_idx = torch.where(iou == gt_max_iou)[0]  # 获得与gt_bbox的iou最大的anchor
        return argmax_iou, per_anchor_max_iou, max_iou_anchor_idx


class ProposalTargetLayer(nn.Module):
    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        super().__init__()
        device = cfg.INIT.DEVICE
        self.device = device
        self.n_sample = torch.tensor(n_sample).to(device)
        self.pos_ratio = torch.tensor(pos_ratio).to(device)
        self.pos_iou_thresh = torch.tensor(pos_iou_thresh).to(device)
        self.neg_iou_thresh_hi = torch.tensor(neg_iou_thresh_hi).to(device)
        self.neg_iou_thresh_lo = torch.tensor(neg_iou_thresh_lo).to(device)

        self.positive_num = torch.round(self.n_sample * self.pos_ratio).to(device)  # 正样本个数

    def forward(self, roi, gt_bbox, gt_label, loc_normalize_mean=(0., 0., 0., 0.),
                loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        # 从2000个roi中抽取128个用于训练
        # RoIs和GT box的IOU大于0.5的，选择一些如32个。
        # RoIs和gt_bboxes的IoU小于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本。
        # 将生成的roi区域与真实的bbox拼接，用于训练样本的生成
        roi = torch.cat((roi, gt_bbox), dim=0)
        # 计算roi与gt_bbox的IoU值
        iou = cal_IoU(roi, gt_bbox)  # shape: num_of_(roi+gt_bbox)*num_of_gt_bbox
        iou_max_idx = iou.argmax(dim=1)  # 获取每一个roi与gt_bbox之间iou最大的索引
        iou_max = iou.max(dim=1).values  # 最大的iou值

        iou_max_label = gt_label[iou_max_idx] + 1  # 获取对应的label，注：0是背景

        def sample_select(where, min_sample_num):
            roi_idx = torch.where(where)[0].float()
            sample_num = min(len(roi_idx), min_sample_num)
            if torch.is_tensor(sample_num):
                sample_num = sample_num.item()
            roi_idx = roi_idx
            if sample_num < len(roi_idx):
                # 从positive_roi_idx中选择positive_num个正样本
                roi_idx_idx = torch.multinomial(roi_idx, sample_num)
                roi_idx = roi_idx[roi_idx_idx]
            return roi_idx.long()

        # 选择正样本：RoIs和gt_box的IoU大于0.5的
        positive_roi_idx = sample_select(iou_max >= self.pos_iou_thresh, self.positive_num)
        # 选择负样本：RoIs和gt_box的IoU小于0.5大于0的
        positive_num = len(positive_roi_idx)
        negative_num = self.n_sample - positive_num
        negative_roi_idx = sample_select((self.neg_iou_thresh_lo <= iou_max) & (iou_max <= self.neg_iou_thresh_hi),
                                         negative_num)
        keep_idx = torch.cat((positive_roi_idx, negative_roi_idx))
        roi_samples = roi[keep_idx]  # 用于训练的样本
        roi_sample_labels = iou_max_label[keep_idx]  # 样本标签
        roi_sample_labels[positive_num:] = 0  # 将负样本标签全部设为背景

        # 计算用于训练的的roi样本与gt_bbox之间的偏移量
        roi_gt_offset_scale = cal_offset_scale(roi_samples, gt_bbox[iou_max_idx[keep_idx]])
        roi_gt_offset_scale = \
            (roi_gt_offset_scale - torch.tensor(loc_normalize_mean, dtype=torch.float32,
                                                device=self.device)) / torch.tensor(
                loc_normalize_std, dtype=torch.float32, device=self.device)
        return roi_samples, roi_gt_offset_scale, roi_sample_labels

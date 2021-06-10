import torch.nn as nn
import torch
from torchvision.ops import RoIPool

from detection.Faster_RCNN.utils.model_helper import normal_init


class VGGRoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier):
        super().__init__()
        # 将感兴趣区域统一池化到7*7的大小，spatial_scale表示被池化区域与原图的比例
        self.roi_pool = RoIPool((roi_size, roi_size), spatial_scale)
        # 4096是VGG中classifier的维度
        self.bbox_pred = nn.Linear(4096, n_classes * 4)  # 对每一个类别目标预测框回归位置
        self.cls_pred = nn.Linear(4096, n_classes)  # 预测类别
        self.classifier = classifier
        normal_init(self.cls_pred, 0, 0.001)
        normal_init(self.bbox_pred, 0, 0.01)

    def forward(self, inputs, rois, rois_idx):
        idx_rois = torch.cat([rois_idx[:, None], rois],
                             dim=1).contiguous()  # 将每一个roi所属的batch与roi拼接，
        # idx_rois shape：num{len(roi)}*5{idx_of_batch,x_min,y_min,x_max,y_max}
        pool = self.roi_pool(inputs, idx_rois)  # shape：num{len(roi)}*512{VGG最后一层conv的通道数}*7{roi_size}*7
        pool = pool.view(pool.size(0), -1)  # shape: num*25088(512*7*7)，对每一个roi进行roi_pool的结果在所有通道上进行展平处理
        classifier_res = self.classifier(pool)  # VGG16的分类器
        bbox_pred = self.bbox_pred(classifier_res)  # 对每一个类别的边界框进行回归
        cls_pred = self.cls_pred(classifier_res)  # 对每一个类别的概率进行预测
        return cls_pred, bbox_pred

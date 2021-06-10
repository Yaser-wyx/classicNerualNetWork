import torch
import torch.nn as nn
import numpy as np
from detection.Faster_RCNN.proposal_layer import ProposalLayer
from detection.Faster_RCNN.utils.bbox_tools import generate_anchor_template, generate_raw_image_anchor
import torch.nn.functional as F

from detection.Faster_RCNN.utils.model_helper import normal_init


class RegionProposalNetWork(nn.Module):

    def __init__(self, in_channels=512, mid_channels=512, anchor_ratios=None, anchor_scales=None, feat_stride=16):
        """
        :param in_channels(int): The channel size of input.
        :param mid_channels(int): The channel size of middle layer.
        :param anchor_ratios(list of floats): The different ratio of anchor for width to height.
        :param anchor_scales(list of numbers): The different zoom ratio of anchor areas.
                                    Such as the base anchor area is 16*16, then if the anchor_scales is [8,16,32],
                                    the zoomed anchors area will be [16*16*8, 16*16*16, 16*16*32].
        :param feat_stride(int): The down sampling stride, the product of stride of each conv layers and pooling layers
                                in extractor. In VGG, the value is 16.
        """
        super().__init__()
        if anchor_scales is None:
            anchor_scales = np.array([8, 16, 32])
        if anchor_ratios is None:
            anchor_ratios = np.array([0.5, 1, 2])
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalLayer(feat_stride, anchor_scales, anchor_ratios)
        # generate the anchors template
        self.anchor_num = anchor_ratios.shape[0] * anchor_scales.shape[0]
        # 对backbone提取的特征再一次进行特征提取，保持维度不变
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)
        # 预测每一个anchor中内容是后景还是前景的概率。
        # 注：后景在前，前景在后
        self.bf_pred_conv = nn.Conv2d(mid_channels, self.anchor_num * 2, 1, 1, 0)
        # 预测每一个anchor需要修正的比例和偏移量
        self.pred_offset_scale_conv = nn.Conv2d(mid_channels, self.anchor_num * 4, 1, 1, 0)
        normal_init(self.conv, 0, 0.01)
        normal_init(self.bf_pred_conv, 0, 0.01)
        normal_init(self.pred_offset_scale_conv, 0, 0.01)

    def forward(self, inputs, img_size):
        # 1. RPN head 部分
        batch, channel, feature_height, feature_width = inputs.shape
        rpn_feature = F.relu(self.conv(inputs), inplace=True)
        # 注：经过多层特征提取后, rpn_feature的每一个点都是对应到原图的一个anchor
        # 计算每一个anchor需要修正的比例和偏移量，每一个channel对应一个anchor的偏移量以及缩放比例
        # 注：每一个channel对应的是缩放比例还是偏移量是无所谓的，但一旦确定了，就不能随意改变
        # 本例中, channel分别代表:
        # offset_x_ratio(x偏移量的比率), offset_y_ratio(y偏移量的比率), scale_h_ratio(高度缩放比率), scale_w_ratio(宽度缩放比率)
        offset_scale_pred = self.pred_offset_scale_conv(rpn_feature)  # shape: n*(4*k)*h*w
        # 将预测需要修正的比例和偏移量放到最后一个维度上
        offset_scale_pred = offset_scale_pred.permute(0, 2, 3, 1).contiguous().view(batch, -1, 4)
        # 计算每一个anchor是景和前景的概率
        rpn_bbox_bf = self.bf_pred_conv(rpn_feature)  # n*(2*k)*h*w
        rpn_bbox_bf = rpn_bbox_bf.permute(0, 2, 3, 1).contiguous()  # n*h*w*(2*k)，每一组anchor（9个）的景和前景的概率

        rpn_bbox_bf_pred = rpn_bbox_bf.view(batch, feature_height, feature_width, self.anchor_num,
                                            2)  # 每一个anchor是前景和后景的概率
        rpn_bbox_bf_pred = F.softmax(rpn_bbox_bf_pred, dim=4)  # 对后景和前景计算softmax
        rpn_bbox_f_pred = rpn_bbox_bf_pred[:, :, :, :, 1].contiguous()  # 只要每一个anchor是前景的概率
        rpn_bbox_f_pred = rpn_bbox_f_pred.view(batch, -1)  # 将所有anchor在一个维度里，最后一维只保留前景概率
        rpn_bbox_bf_pred = rpn_bbox_bf_pred.view(batch, -1, 2)
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # 2. 获取rois
        rois, rois_idx, anchors = self.proposal_layer(inputs=inputs, img_size=img_size, cfg_key=cfg_key,
                                                      offset_scale_pred=offset_scale_pred, bbox_f_pred=rpn_bbox_f_pred)

        return rois, rois_idx, anchors, offset_scale_pred, rpn_bbox_bf_pred

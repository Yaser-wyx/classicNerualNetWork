import numpy as np
import numpy as xp

from torch.nn import functional as F
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa

from detection.Faster_RCNN.utils._bbox_tools import generate_anchor_template
from detection.dataset.voc_dataset import VocDataset


class RegionProposalNetwork(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, feat_stride=16):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_template()
        self.feat_stride = feat_stride
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # 再次提取特征
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  #
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

    def forward(self, x, img_size=None, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))
        # n*(4*k)*hh*ww
        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # 获得每一个anchor各个坐标的预测值 shape：n*(h*w*k)*4
        # n*(2*k)*h*w
        rpn_scores = self.score(h)  # 前景与背景的概率
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()  # n*h*w*(2*k)

        tmp = rpn_scores.view(n, hh, ww, n_anchor, 2)  # n*h*w*9*2
        rpn_softmax_scores = F.softmax(tmp, dim=4)  # 对前景与后景的概率进行softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 获取每一个anchor是前景的概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1)  # 每一个anchor是前景的概率
        rpn_scores = rpn_scores.view(n, -1, 2)  # 每一个anchor 前景与后景的概率

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


from torchvision.models import vgg16


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    model = vgg16(pretrained=False)
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]

    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


if __name__ == '__main__':
    extractor, classifier = decom_vgg16()
    root = "D:\\classicNerualNetWork\\detection\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    voc_dataset = VocDataset(root_dir=root, transforms=seq)
    voc_dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1)
    image, bbox_list, label = next(iter(voc_dataloader))
    print(image.shape)
    rpn = RegionProposalNetwork()
    image_feature = extractor(image)
    print(image_feature.shape)
    # C, H, W = image.shape
    rpn(image_feature)

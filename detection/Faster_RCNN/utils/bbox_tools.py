import numpy as np
import torch

from detection.Faster_RCNN.utils.config import cfg


def generate_anchor_template(base_size=16, scales=None, ratios=None):
    """
    :param base_size: 基本的anchor大小
    :param scales: 用于缩放anchor的宽度和高度，如果anchor尺寸为16*16，尺度为8，则anchor面积为(16*8)*(16*8)。
    :param ratios: 不同的anchor宽高之比，如anchor为16*16(H*W)，比值为0.5(W:H=1:2=0.5)，则anchor将转换为22*11(11/22=0.5)，
    面积与原anchor相似。
    注意图像是H*W，比例是W:H。
    :return: anchor列表。每个元素都是一个边界框的坐标列表。第二个轴值是[x_min, y_min, x_max, y_max]。
    """
    if scales is None:
        scales = np.array([8, 16, 32])
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    base_anchor = np.array([1, 1, base_size, base_size]) - 1  # x_min,y_min,x_max,y_max
    ratio_anchor_boxes = _ratio_anchor(base_anchor, ratios)  # 生成不同比例的anchor
    anchors = np.vstack([_scale_anchor(ratio_anchor_boxes[i, :], scales)
                         for i in range(ratio_anchor_boxes.shape[0])])
    x_ctr, y_ctr, _, __ = _cal_anchor_ctr(base_anchor)
    return np.round(anchors - [x_ctr, y_ctr, x_ctr, y_ctr])  # 平移到(0,0)


def _scale_anchor(anchor_box, scales):
    x_ctr, y_ctr, w, h = _cal_anchor_ctr(anchor_box)
    ws = scales * w
    hs = scales * h
    return _mk_anchors(ws, hs, x_ctr, y_ctr)


def _ratio_anchor(anchor_box, ratios):
    """
    计算公式: W/H=R->W=R*H->S=H*W->S=R*H*H->H=Sqrt(S/R)
    """
    x_ctr, y_ctr, w, h = _cal_anchor_ctr(anchor_box)
    area = w * h
    hs = np.round(np.sqrt(area / ratios))
    ws = np.round(hs * ratios)
    return _mk_anchors(ws, hs, x_ctr, y_ctr)


def _mk_anchors(ws, hs, x_ctr, y_ctr):
    # return a set of [x_min,y_min,x_max,y_max]
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _cal_anchor_ctr(anchor_box):
    """
    用来计算anchor的中心坐标以及宽高，兼容同时计算多个anchor的
    :param anchor_box: [x_min,y_min,x_max,y_max]
    :return: anchor center [x_ctr,y_ctr,w,h]
    """
    shape = anchor_box.shape
    if len(shape) == 1:
        w = anchor_box[2] - anchor_box[0] + 1  # As the value start with 0, so value should plus 1.
        h = anchor_box[3] - anchor_box[1] + 1
        x_ctr = anchor_box[0] + 0.5 * (w - 1)
        y_ctr = anchor_box[1] + 0.5 * (h - 1)
        return x_ctr, y_ctr, w, h
    else:
        ws = anchor_box[:, 2] - anchor_box[:, 0] + 1  # As the value start with 0, so value should plus 1.
        hs = anchor_box[:, 3] - anchor_box[:, 1] + 1
        x_ctrs = anchor_box[:, 0] + 0.5 * (ws - 1)
        y_ctrs = anchor_box[:, 1] + 0.5 * (hs - 1)
        return x_ctrs, y_ctrs, ws, hs


def generate_raw_image_anchor(feature_height, feature_width, feat_stride, anchors_template):
    # 生成原始图片上的anchor
    img_height, img_width = feature_height * feat_stride, feature_width * feat_stride
    shift_x = np.arange(0, img_width, feat_stride)
    shift_y = np.arange(0, img_height, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = torch.from_numpy(np.stack(  # x_min, y_min, x_max, y_max
        (np.ravel(shift_x), np.ravel(shift_y), np.ravel(shift_x), np.ravel(shift_y)),
        axis=1))  # shape: 2420,4 每一行就是一个bbox各坐标的偏移量
    # 对每一个点生成9个anchor
    shift_xys = shift.unsqueeze(0).permute((1, 0, 2))
    anchors = anchors_template.unsqueeze(0)
    anchors = shift_xys + anchors  # 将anchor模板加上偏移量
    anchors = anchors.reshape((anchors.shape[0] * anchors.shape[1], 4))
    device = cfg.INIT.DEVICE
    if device == torch.device("cuda"):
        anchors = anchors.cuda()
    return anchors


def amend_bbox(pred_bbox, offset_scale_pred):
    # 修正预测的bbox，已知预测的bbox和位置偏差dx,dy,dh,dw，求目标框G

    # scale_offset: offset_x_ratio(x偏移量率), offset_y_ratio(y偏移量率), scale_h_ratio(高度缩放率), scale_w_ratio(宽度缩放率)
    w = pred_bbox[:, 2] - pred_bbox[:, 0]
    h = pred_bbox[:, 3] - pred_bbox[:, 1]
    x_ctr = pred_bbox[:, 0] + 0.5 * w
    y_ctr = pred_bbox[:, 1] + 0.5 * h
    # 每4个为一组
    offset_x_ratio = offset_scale_pred[:, 0::4]  # 选 0,4,8。。。。
    offset_y_ratio = offset_scale_pred[:, 1::4]
    scale_h_ratio = offset_scale_pred[:, 2::4]
    scale_w_ratio = offset_scale_pred[:, 3::4]
    # 给w,h,x_ctr,y_ctr新增一个维度
    w.unsqueeze_(1)
    h.unsqueeze_(1)
    x_ctr.unsqueeze_(1)
    y_ctr.unsqueeze_(1)
    # \delta{x} = offset_x_ratio * w 预测框在x轴上偏移的量，
    # offset_x_ratio是一个与预测框宽度相关的偏移率
    ctr_x_pred = offset_x_ratio * w + x_ctr
    ctr_y_pred = offset_y_ratio * h + y_ctr
    w_pred = w * torch.exp(scale_w_ratio)
    h_pred = h * torch.exp(scale_h_ratio)
    # 对预测框进行修正后的bbox，输出值为四个顶点坐标
    pred_gt_bbox = torch.zeros_like(offset_scale_pred)
    pred_gt_bbox[:, 0::4] = ctr_x_pred - 0.5 * w_pred  # x_min
    pred_gt_bbox[:, 1::4] = ctr_y_pred - 0.5 * h_pred  # y_min
    pred_gt_bbox[:, 2::4] = ctr_x_pred + 0.5 * w_pred  # x_max
    pred_gt_bbox[:, 3::4] = ctr_y_pred + 0.5 * h_pred  # y_max

    return pred_gt_bbox


def cal_IoU(roi, gt_bbox):
    # 计算IoU值
    # roi shape: num_of_roi*4
    # gt_box shape: num_of_gt*4
    # Note: 两者维度不统一，给roi中间添加一个维度，变为num_of_roi*1*4
    # 计算每一个roi与gt_bbox的交集区域（左上角与右下角）
    x_y_min = torch.maximum(roi[:, None, :2], gt_bbox[:, :2])  # 计算每一个roi与每一个gt_box左上角的交点 （画图即可明白）
    x_y_max = torch.minimum(roi[:, None, 2:], gt_bbox[:, 2:])  # 计算每一个roi与每一个gt_box右下角的交点 （画图即可明白）
    # x_y_min与x_y_max的shape：num_of_roi*num_of_gt*2
    # 计算交集宽和高
    w = x_y_max[:, :, 0] - x_y_min[:, :, 0]
    h = x_y_max[:, :, 1] - x_y_min[:, :, 1]
    # 去除掉非法值
    w[w < 0] = 0
    h[h < 0] = 0
    # 计算区域的面积
    overlaps_area = w * h  # 交集面积 shape：num_of_roi*1
    roi_area = (roi[:, 2] - roi[:, 0]) * (roi[:, 3] - roi[:, 1])
    gt_box_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])

    iou = overlaps_area / (roi_area[:, None] + gt_box_area - overlaps_area)  # iou计算
    return iou


def cal_offset_scale(pred_bbox, gt_bbox):
    # 根据pred_bbox与gt_bbox计算偏移量
    # 计算pred_bbox的中心坐标以及高宽
    pred_x_ctrs, pred_y_ctrs, pred_ws, pred_hs = _cal_anchor_ctr(pred_bbox)
    # 计算gt_bbox的中心坐标以及高宽
    gt_x_ctrs, gt_y_ctrs, gt_ws, gt_hs = _cal_anchor_ctr(gt_bbox)
    device = cfg.INIT.DEVICE

    eps = torch.tensor(torch.finfo(torch.float32).eps, device=device)
    # Note：pred_ws与pred_hs要用于分母，防止出现0
    pred_ws = torch.maximum(eps, pred_ws)
    pred_hs = torch.maximum(eps, pred_hs)

    offset_x_ratios = (gt_x_ctrs - pred_x_ctrs) / pred_ws
    offset_y_ratios = (gt_y_ctrs - pred_y_ctrs) / pred_hs
    scale_w = torch.log(gt_ws / pred_ws)
    scale_h = torch.log(gt_hs / pred_hs)
    pred_gt_offset_scale = torch.vstack((offset_x_ratios, offset_y_ratios, scale_h, scale_w))
    return pred_gt_offset_scale.T


def map_data2anchor(data, anchor_idx, anchor_num, default_value):
    device = cfg.INIT.DEVICE

    if len(data.shape) == 1:
        anchor_data = torch.empty((anchor_num,), dtype=data.dtype, device=device)
        anchor_data.fill_(default_value)
        anchor_data[anchor_idx] = data
    else:
        anchor_data = torch.empty((anchor_num,) + data.shape[1:], dtype=data.dtype, device=device)
        anchor_data.fill_(default_value)
        anchor_data[anchor_idx, :] = data
    return anchor_data

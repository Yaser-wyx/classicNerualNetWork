from torchvision.models import vgg16
import torch
import torch.nn as nn


def init_VGG16(pretrain=False, model_path=None):
    model = vgg16(pretrain)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    extractor = list(model.features)[:30]  # 去掉最后一个池化层
    classifier = list(model.classifier)
    del classifier[6]  # 去掉最后一个分类层
    # 冻结前十层的参数
    for layer in extractor[:10]:
        for param in layer.parameters():
            param.requires_grad = False

    return nn.Sequential(*extractor), nn.Sequential(*classifier)


def cal_reg_loss(pred_offset_scale, gt_offset_scale, gt_label, sigma):
    # 只需要对正样本计算回归损失
    weight = torch.zeros_like(gt_offset_scale)
    weight[gt_label > 0] = 1
    print("weight:",torch.sum(weight == 1))
    reg_loss = cal_smooth1_loss(pred_offset_scale, gt_offset_scale, weight.detach(), sigma)
    reg_loss /= (gt_label >= 0).sum().float()
    return reg_loss


def cal_smooth1_loss(input, target, weight, sigma):
    sigma = sigma ** 2
    x = weight * (input - target)
    x_abs = torch.abs(x)
    flag = (x < 1 / sigma).float()
    loss = (flag * (sigma / 2.) * (x ** 2) + (1 - flag) * (x_abs - 0.5 / sigma))
    return loss.sum()


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

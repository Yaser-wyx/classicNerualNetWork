from detection.Faster_RCNN.faster_rcnn import FasterRCNN
from detection.Faster_RCNN.roi import VGGRoIHead
from detection.Faster_RCNN.rpn import RegionProposalNetWork
from detection.Faster_RCNN.utils.config import cfg
from detection.Faster_RCNN.utils.model_helper import init_VGG16


class VGGFasterRCNN(FasterRCNN):
    def __init__(self):
        # 加载配置参数
        pretrain_model_path = cfg.INIT.PRETRAIN_MODEL_PATH
        pretrain = cfg.INIT.PRETRAIN
        n_classes = cfg.INIT.N_CLASSES
        ratios = cfg.INIT.RATIOS
        anchor_scales = cfg.INIT.ANCHOR_SCALES
        rpn_in_channel = cfg.INIT.RPN_IN_CHANNEL
        rpn_mid_channel = cfg.INIT.RPN_MID_CHANNEL
        feat_stride = cfg.INIT.FEAT_STRIDE
        roi_size = cfg.INIT.ROI_SIZE

        extractor, classifier = init_VGG16(pretrain, pretrain_model_path)

        rpn = RegionProposalNetWork(in_channels=rpn_in_channel, mid_channels=rpn_mid_channel, anchor_ratios=ratios,
                                    anchor_scales=anchor_scales, feat_stride=feat_stride)

        roi_head = VGGRoIHead(n_classes=n_classes + 1, roi_size=roi_size, spatial_scale=1. / feat_stride,
                              classifier=classifier)

        super().__init__(extractor, rpn, roi_head)

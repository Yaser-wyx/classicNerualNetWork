import torch
from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

__C.INIT = edict()
__C.INIT.DATA_ROOT = "D:\\classicNerualNetWork\\detection\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"
__C.INIT.EPOCHS = 10
__C.INIT.PRETRAIN = True
__C.INIT.PRETRAIN_MODEL_PATH = None
__C.INIT.N_CLASSES = 20
__C.INIT.RATIOS = np.array([0.5, 1, 2])
__C.INIT.RPN_SIGMA = 3.
__C.INIT.ROI_SIGMA = 1.
__C.INIT.ANCHOR_SCALES = np.array([8, 16, 32])
__C.INIT.RPN_IN_CHANNEL = 512
__C.INIT.RPN_MID_CHANNEL = 512
__C.INIT.FEAT_STRIDE = 16
__C.INIT.ROI_SIZE = 7
__C.INIT.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__C.TRAIN = edict()

__C.TRAIN.LEARNING_RATE = 0.002
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.MAX_SIZE = 1000
__C.TRAIN.RPN_NMS_THRESH = 0.7
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
__C.TRAIN.RPN_MIN_SIZE = 8

__C.TEST = edict()

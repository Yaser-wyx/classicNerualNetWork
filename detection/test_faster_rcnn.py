from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torch
import numpy as np
import random
from detection.Faster_RCNN.trainer import FasterRCNNTrainer
from detection.Faster_RCNN.vgg_faseter_rcnn import VGGFasterRCNN
from detection.dataset.voc_dataset import VocDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    vgg_faster_rcnn = VGGFasterRCNN()

    optimizer = torch.optim.SGD(vgg_faster_rcnn.parameters(), lr=0.001, momentum=0.5)
    print(type(optimizer))
    # setup_seed(20)
    #
    # root = "D:\\classicNerualNetWork\\detection\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"
    # seq = iaa.Sequential([
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5)
    # ])
    # voc_dataset = VocDataset(root_dir=root, transforms=seq)
    # voc_dataloader = DataLoader(voc_dataset, shuffle=False, batch_size=1)
    # image, bbox_list, label = next(iter(voc_dataloader))
    # print(bbox_list, label)
    # vgg_faster_rcnn = VGGFasterRCNN()
    # trainer = FasterRCNNTrainer(vgg_faster_rcnn)
    # losses = trainer(image, bbox_list, label)
    # print(losses)

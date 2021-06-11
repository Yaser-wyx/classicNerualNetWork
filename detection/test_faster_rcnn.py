from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torch
import numpy as np
import random
from detection.Faster_RCNN.trainer import FasterRCNNTrainer
from detection.Faster_RCNN.utils.config import cfg
from detection.Faster_RCNN.vgg_faseter_rcnn import VGGFasterRCNN
from detection.dataset.voc_dataset import VocDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(20)
    device = cfg.INIT.DEVICE
    root = "D:\\classicNerualNetWork\\detection\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012"
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    voc_dataset = VocDataset(root_dir=root, transforms=seq)
    voc_dataloader = DataLoader(voc_dataset, shuffle=False, batch_size=1)
    image, bbox_list, label = next(iter(voc_dataloader))
    image, bbox_list, label = image.to(device), bbox_list.to(device), label.to(device)
    print(bbox_list, label)
    vgg_faster_rcnn = VGGFasterRCNN()
    vgg_faster_rcnn.to(device=device)
    optimizer = torch.optim.SGD(vgg_faster_rcnn.parameters(), lr=0.001, momentum=0.5)
    trainer = FasterRCNNTrainer(vgg_faster_rcnn, optimizer, device)
    losses = trainer.step(image, bbox_list, label)
    print(losses)

import copy

from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import torch
import numpy as np
import random
from detection.Faster_RCNN.trainer2 import FasterRCNNTrainer
from detection.Faster_RCNN.utils.config import cfg
from detection.Faster_RCNN.vgg_faseter_rcnn import VGGFasterRCNN
from detection.dataset.dataset import Dataset
from detection.dataset.voc_dataset import VocDataset
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(trainer, train_dataloader, valid_dataloader, device, epochs, get_best=True,
          log_dir=None):
    best_loss = 99999
    best_model = None
    for epoch in range(epochs):
        trainer.set_mode()
        for index, (image, bbox_list, label) in enumerate(train_dataloader):
            # print(bbox_list, label)
            image, bbox_list, label = image.to(device), bbox_list.to(device), label.to(device)
            losses = trainer.step(image, bbox_list, label)
            print(
                    "Epoch:{} iterations:{}, rpn_cls_loss: {}, rpn_reg_loss: {}, roi_cls_loss: {}, roi_reg_loss: {}, "
                    "total_loss :{}".format(
                        epoch, index,
                        losses.rpn_cls_loss.item(),
                        losses.rpn_reg_loss.item(),
                        losses.roi_cls_loss.item(),
                        losses.roi_reg_loss.item(),
                        losses.total_loss.item()))

            #
            # if index % 10 == 0:
            #     print(
            #         "Epoch:{} iterations:{}, rpn_cls_loss: {}, rpn_reg_loss: {}, roi_cls_loss: {}, rpn_reg_loss: {}, "
            #         "total_loss :{}".format(
            #             epoch, index,
            #             losses.rpn_cls_loss.item(),
            #             losses.rpn_reg_loss.item(),
            #             losses.roi_cls_loss.item(),
            #             losses.rpn_reg_loss.item(),
            #             losses.total_loss.item()))

        # trainer.set_mode("eval")
        # total_loss = 0.
        # for index, (image, bbox_list, label) in enumerate(valid_dataloader):
        #     image, bbox_list, label = image.to(device), bbox_list.to(device), label.to(device)
        #     losses = trainer.step(image, bbox_list, label)
        #     total_loss += losses.total_loss.item()
        # val_loss = total_loss / len(valid_dataloader.dataset)
        # print("Epoch:{} val_loss:{} ".format(epoch, val_loss))


if __name__ == '__main__':
    setup_seed(20)
    print("read config file")
    device = cfg.INIT.DEVICE
    root = cfg.INIT.DATA_ROOT
    epochs = cfg.INIT.EPOCHS
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ])
    print("build dataset")
    voc_train_dataset = Dataset(root)
    train_dataloader = DataLoader(voc_train_dataset, shuffle=False, batch_size=1)

    voc_val_dataset = VocDataset(root_dir=root, transforms=seq, purpose="val")
    val_dataloader = DataLoader(voc_val_dataset, shuffle=True, batch_size=1)
    print("build network")
    vgg_faster_rcnn = VGGFasterRCNN()
    vgg_faster_rcnn.to(device=device)
    optimizer = torch.optim.SGD(vgg_faster_rcnn.parameters(), lr=0.001, momentum=0.9)
    trainer = FasterRCNNTrainer(vgg_faster_rcnn, optimizer, device)
    print("start train")
    train(trainer=trainer, train_dataloader=train_dataloader, valid_dataloader=val_dataloader, device=device,
          epochs=epochs)

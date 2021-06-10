from torch.utils.data import Dataset
import os
import cv2


class ImageDataset(Dataset):
    def __init__(self,  data, transforms):
        super().__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        path = item[0]
        image = cv2.imread(path)
        res = self.transforms(image)

        return res, item[1]


def data_process(root):
    """get data path and labels"""
    # read labels
    # read train labels
    labels = []
    with open(os.path.join(root, "tiny-imagenet-200/tiny-imagenet-200/wnids.txt")) as file:
        for line in file:
            labels.append(line.strip())
    labels_map = {label: index for index, label in enumerate(labels)}
    idx2label = {index: label for index, label in enumerate(labels)}
    # read train data
    train_path = os.path.join(root, "tiny-imagenet-200/tiny-imagenet-200/train")
    train_data = []
    for label in labels:
        images_path = os.path.join(train_path, label, "images")
        images_list = list(os.listdir(images_path))
        for image in images_list:
            image_path = os.path.join(images_path, image)
            train_data.append((image_path, labels_map[label]))

    # read val data
    val_images_path = os.path.join(root, "tiny-imagenet-200/tiny-imagenet-200/val/images")
    val_data = []
    with open(os.path.join(root, "tiny-imagenet-200/tiny-imagenet-200/val/val_annotations.txt")) as file:
        for line in file:
            line_list = line.strip().split()
            picture_name = line_list[0]
            label = line_list[1]
            val_data.append((os.path.join(val_images_path, picture_name), labels_map[label]))

    return labels_map, idx2label, train_data, val_data

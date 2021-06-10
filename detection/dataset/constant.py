import os

ANNOTATIONS = lambda root: os.path.join(root, "Annotations")
TRAIN_PATH = lambda root: os.path.join(root, "ImageSets", "Main", "train.txt")
VAL_PATH = lambda root: os.path.join(root, "ImageSets", "Main", "val.txt")
TEST_PATH = lambda root: os.path.join(root, "ImageSets", "Main", "test.txt")
IMAGE_DIR_PATH = lambda root: os.path.join(root, "JPEGImages")

VOC_BBOX_LABEL_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor']
VOC_BBOX_LABEL_NAMES2IDX = {name: idx for idx, name in enumerate(VOC_BBOX_LABEL_NAMES)}

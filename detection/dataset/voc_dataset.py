from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from detection.dataset.constant import *
from detection.dataset.utils import analyse_xml
import numpy as np
import cv2


class VocDataset(Dataset):
    def __init__(self, root_dir, purpose="train", transforms=None):
        # 对传入的路径参数进行判断是否正确
        if not os.path.exists(root_dir):
            raise RuntimeError("Root dir: {} is not exists".format(root_dir))
        assert purpose in ["train", "val", "test"], "The field of purpose is invalid for {}".format(purpose)

        self.purpose = purpose
        self.root_dir = root_dir
        self.transforms = transforms
        self.dataset = None
        self.dataset_file_path = None

        if self.purpose is "train":
            self.dataset_file_path = TRAIN_PATH(self.root_dir)  # 获取训练验证集的路径
        elif self.purpose is "test":
            self.dataset_file_path = TEST_PATH(self.root_dir)  # 获取测试集的路径
        else:
            self.dataset_file_path = VAL_PATH(self.root_dir)  # 获取测试集的路径

        if not os.path.exists(self.dataset_file_path):  # 判断是否存在
            raise RuntimeError("Dataset dir: {} is not exists".format(self.dataset_file_path))
        # 初始化数据集
        self.init_dataset()

    def init_dataset(self):
        # 读取相关数据集列表
        filename_list = []
        with open(self.dataset_file_path, mode="r", encoding="utf-8") as file:
            for line in file:
                filename_list.append(line.strip())
        # 获取图片路径列表
        image_dir = IMAGE_DIR_PATH(self.root_dir)
        annotation_path = ANNOTATIONS(self.root_dir)
        self.dataset = []
        for filename in filename_list:
            image_path = os.path.join(image_dir, filename + ".jpg")
            # 解析xml文件
            bbox, label = analyse_xml(os.path.join(annotation_path, filename + ".xml"))
            self.dataset.append({
                "image_path": image_path,
                "bbox": bbox,
                "label": label
            })

    def __getitem__(self, index) -> T_co:
        item = self.dataset[index]
        image_path = item["image_path"]
        label = item["label"]
        image = cv2.imread(image_path)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print("raw image shape:{}".format(image.shape))
        bbox_list = item["bbox"]
        if self.transforms is not None:
            # 对数据进行转化，增强数据
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]) for bbox in bbox_list],
                shape=image.shape)  # bbox坐标转化
            image, bbx_aug = self.transforms(image=image, bounding_boxes=bbs)  # 对image进行转化
            print("transformed image shape:{}".format(image.shape))
            bbox_list = []
            # 提取坐标数据
            for index in range(len(bbx_aug.bounding_boxes)):
                bbox = bbx_aug.bounding_boxes[index]
                bbox_list.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            bbox_list = np.array(bbox_list, dtype=np.float32)
        """
        对图像进行缩放操作，因为原论文要求输入到网络中的图片大小要在600*600~1000*1000之间
        """
        H, W, _ = image.shape
        scale1 = 600 / min(H, W)
        scale2 = 1000 / max(H, W)
        scale = min(scale1, scale2)
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将image转为torch支持的tensor格式
            transforms.Resize([int(H * scale), int(W * scale)]),
        ])
        """
        注：此处image必须copy后才能进行转换，否则会报错。
        RuntimeError: some of the strides of a given numpy array are negative
        """
        image = transform(image.copy())
        print("scaled image shape:{}".format(image.shape))
        return image, bbox_list, label

    def __len__(self):
        return len(self.dataset)

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from detection.dataset.constant import *
from detection.dataset.utils import analyse_xml, read_image
import numpy as np
import cv2
import xml.etree.ElementTree as ET


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data.
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label

    __getitem__ = get_example


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
        bbox_list = item["bbox"]
        if self.transforms is not None:
            # 对数据进行转化，增强数据
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]) for bbox in bbox_list],
                shape=image.shape)  # bbox坐标转化
            image, bbx_aug = self.transforms(image=image, bounding_boxes=bbs)  # 对image进行转化
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
        return image, bbox_list, label

    def __len__(self):
        return len(self.dataset)

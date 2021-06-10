import xml.etree.ElementTree as ET
from .constant import *
import os
import numpy as np


def analyse_xml(xml_path):
    element_tree = ET.parse(xml_path)
    object_list = element_tree.findall("object")  # 找到所有的object
    label_list = []
    bbox_list = []
    for obj in object_list:
        label = obj.find("name").text.lower().strip()
        label_list.append(VOC_BBOX_LABEL_NAMES2IDX[label])
        bbox_node = obj.find("bndbox")
        # xmin, ymin, xmax, ymax
        bbox_list.append([x.text for x in bbox_node])
    label_list = np.array(label_list, dtype=np.float32)
    bbox_list = np.array(bbox_list, dtype=np.float32)
    return bbox_list, label_list


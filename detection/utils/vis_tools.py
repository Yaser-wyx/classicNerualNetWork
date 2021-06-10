import matplotlib.pyplot as plt
import cv2 as cv


def bbox_to_rect(bbox, color="red"):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_image_bbox(image, bbox_list, labels, label_dict):
    fig = plt.imshow(image)
    for i, label in enumerate(labels):
        rect = bbox_to_rect(bbox_list[i])
        fig.axes.add_patch(rect)
        fig.axes.text(rect.xy[0] + 24, rect.xy[1] + 10, label_dict[int(label)],
                      va='center', ha='center', fontsize=6, color='blue',
                      bbox=dict(facecolor='m', lw=0))
    plt.show()

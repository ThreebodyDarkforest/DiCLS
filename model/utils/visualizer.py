import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2

def data_load(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def vis_image(image):
    h, w = image.shape[0], image.shape[1]
    plt.subplots(figsize=(w * 0.01, h * 0.01))
    plt.imshow(image, alpha=1)

    return h, w

def vis_grid_attention(img_path, attention_map, cmap='jet'):
    """
    :param img_path:图像路径
    :param attention_map:注意力图
    :param cmap: cmap是图像的颜色类型，有很多预设的颜色类型
    :return:
    """
    # draw the img
    img = data_load(img_path)
    h, w = vis_image(img)

    # draw the attention
    map = cv2.resize(attention_map, (w, h))
    normed_map = map / map.max()
    normed_map = (normed_map * 255).astype('uint8')
    plt.imshow(normed_map, alpha=0.4, interpolation='nearest', cmap=cmap)  # alpha值决定图像的透明度,0为透明,1不透明

    # 去掉图片周边白边
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 调整图像与画布的边距(此时填充满)
    plt.margins(0, 0)

    # 保存图像,以300dpi
    img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
    plt.savefig(f'./{img_name}', dpi=300)
    print(f'[{img_name}] is generated')
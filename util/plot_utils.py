"""
Plotting utilities to visualize training logs.
"""

import matplotlib.pyplot as plt
import numpy as np

from util.box_ops import box_denormalize
from util.boxes import box_cxcywh_to_xywh


def plot_results(img, boxes, labels=None, probs=None, colors=None, linewidth=3, text_color="yellow", text_alpha=0.5,
                 fontsize=None, figsize=None, return_img=False):
    """
    Plots the bounding boxes and labels onto an image.

    :param img: RGB image with shape [c, h, w] with pixel values between 0 and 255.
    :param boxes: Bounding boxes with shape [n, 4] and format [x_center, y_center, w, h]. Boxes should be normalized to be between
        0 and 1.
    :param labels: List of labels
    :param probs: List of label probabilities
    :param colors: List of colors to cycle through
    :return:
    """
    img = img.permute(1, 2, 0)
    boxes = box_cxcywh_to_xywh(boxes)
    boxes = box_denormalize(boxes, img.shape[0], img.shape[1])

    img = np.array(img)
    boxes = np.array(boxes)
    if labels is not None:
        labels = np.array(labels)
    if probs is not None:
        probs = np.array(probs)
    if colors is None:
        colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    if figsize is None:
        figsize = (img.shape[1] / 100, img.shape[0] / 100)

    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    ax = plt.gca()

    for i in range(len(boxes)):
        x0, y0, w, h = boxes[i]
        color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, color=color, linewidth=linewidth))

        text = None
        if labels is not None:
            label = labels[i]
            text = f'{label}'
        if probs is not None:
            p = probs[i]
            text = text + f': {p:0.2f}'

        if text is not None:
            if fontsize is None:
                fontsize_ratio = 1.8e-05
                fontsize = (figsize[0] * figsize[1]) * 100 * fontsize_ratio
                fontsize = np.round(fontsize, 0).astype(int)
            ax.text(x0, y0 + fontsize, text, fontsize=fontsize, bbox=dict(facecolor=text_color, alpha=text_alpha))

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if return_img:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        out_img = tf.image.decode_png(buf.getvalue(), channels=4)
        return out_img
    else:
        plt.show()

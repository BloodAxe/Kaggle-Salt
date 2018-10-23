import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def create_overlay(image, pred_mask, true_mask=None):
    if true_mask is None:
        true_mask = np.zeros_like(image)

    overlay = np.dstack([image, image + true_mask * 0.5, image + pred_mask * 0.5]) * 255
    return overlay.astype(np.uint8)


def header_image(image, text, font_scale=1, font_thickness=1, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    header = np.zeros((int(1.3 * text_height), image.shape[1], 3), dtype=np.uint8)

    cv2.rectangle(header, (0, 0), (header.shape[1], header.shape[0]), bg_color, -1)
    cv2.putText(header, text, (0, 0 - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color,
                font_thickness, lineType=cv2.LINE_AA)

    return np.concatenate((header, image), axis=0)


def visualize_predictions(images, pred_masks, true_masks, columns=12):
    overlays = np.array([create_overlay(image, y_pred, y_true) for image, y_pred, y_true in zip(images, pred_masks, true_masks)])
    overlays = torch.from_numpy(np.moveaxis(overlays, -1, 1))
    grid = make_grid(overlays, nrow=columns).numpy()
    grid = np.moveaxis(grid, 0, -1)

    cell_size = 2
    x = cell_size * columns
    y = math.ceil(len(images) / columns) * cell_size

    plt.figure(figsize=(x, y))
    plt.imshow(grid)
    plt.tight_layout()

# SPDX-License-Identifier: AGPL-3.0-only
"""Shared 448 letterbox and exact inverse geometry."""
import cv2
import numpy as np

SIZE = 448


def preprocess(image, size=SIZE):
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        raise ValueError('Empty frame')
    ratio = min(size / h, size / w)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh))
    x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    x /= np.float32(255)
    return x, dict(width=w, height=h, resized_width=nw, resized_height=nh, left=left, top=top)


def restore(detections, geometry, threshold=.25):
    rows = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 4] >= threshold)
    valid &= (rows[:, 4] <= 1) & (rows[:, 5] == np.rint(rows[:, 5]))
    valid &= (rows[:, 5] >= 0) & (rows[:, 5] < 80)
    valid &= (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])
    rows = rows[valid].copy()
    for axis, offset, original, resized in (
        ((0, 2), geometry['left'], geometry['width'], geometry['resized_width']),
        ((1, 3), geometry['top'], geometry['height'], geometry['resized_height']),
    ):
        rows[:, axis] = np.clip((rows[:, axis] - offset) * original / resized, 0, original)
    return rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]


def quantize(x, scale, zero, scratch, output):
    np.divide(x, np.float32(scale), out=scratch)
    np.rint(scratch, out=scratch)
    np.add(scratch, np.float32(zero), out=scratch)
    np.clip(scratch, -32768, 32767, out=scratch)
    np.copyto(output, scratch, casting='unsafe')
    return output

# -*- coding: utf-8 -*-
"""画像の読み書き。

OpenCV の `imread` / `imwrite` は、Windows で非 ASCII（日本語など）を含むパスを
開けない。しかも **例外を投げずに None を返す**ので、「全ページが読めない」まま
静かに失敗する。`np.fromfile` + `imdecode` に置き換えれば、どの言語のパスでも動く。
"""
import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """どんなパスでも読める imread。読めなければ None。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img, params=None):
    """どんなパスでも書ける imwrite。"""
    ext = os.path.splitext(path)[1] or '.png'
    ok, enc = cv2.imencode(ext, img, params or [])
    if ok:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        enc.tofile(path)
    return bool(ok)


def downscale(img, long_side=1500):
    """長辺を揃えた縮小画像と縮小率を返す。

    枠の検出は縮小画像で行う。600dpi の原寸（8000px 級）のままでは
    1 ページの処理に何十秒もかかるうえ、紙の縁を探すのに解像度は要らない。
    """
    h, w = img.shape[:2]
    s = float(long_side) / max(h, w)
    if s >= 1.0:
        return img, 1.0
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA), s


IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def list_pages(folder):
    """フォルダ直下のページ画像を名前順に。"""
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXT)
                  and os.path.isfile(os.path.join(folder, f)))

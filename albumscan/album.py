# -*- coding: utf-8 -*-
"""アルバム 1 冊を通して処理する。

**1 冊まとめて見るのが要点**。ページ単位で完結させると、そのページに写真が
2 枚しか無いときに判型が決まらない。同じアルバムのプリントは同じ紙なので、
冊全体の検出結果を集めてから判型を決め、その判型で全ページの枠を確定する。

    pages  → 輪郭マップ（推論 or キャッシュ）
           → 判型の推定（sizes.build）
           → ページごとの枠（boxes.page_boxes）
           → 品質の記録（quality.page_report）
"""
import json
import os

import numpy as np

from . import boxes as B
from . import quality as Q
from . import sizes as S
from .imageio import downscale, imread, list_pages
from .segment import load_map, save_map

LONG_SIDE = 1500        # 検出に使う縮小画像の長辺
DPI = 600.0             # スキャン解像度の既定値


class Album(object):
    def __init__(self, folder, cache_dir=None, dpi=DPI, long_side=LONG_SIDE):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        self.cache_dir = cache_dir or os.path.join(folder, '_cache')
        self.dpi = dpi
        self.long_side = long_side
        self.pages = list_pages(folder)
        self.catalogue = []
        self._maps = {}

    # ---------------------------------------------------------------- 輪郭マップ
    def page_map(self, page, segmenter=None):
        """(inside 等 3 枚, 縮小率) を返す。キャッシュがあれば使う。"""
        if page in self._maps:
            return self._maps[page]
        img = imread(os.path.join(self.folder, page))
        if img is None:
            return None, None
        sm, s = downscale(img, self.long_side)
        m = load_map(self.cache_dir, page, shape=sm.shape)
        if m is None:
            if segmenter is None:
                return None, None
            p = segmenter(sm)
            save_map(self.cache_dir, page, [p[0], p[1], p[2]])
            m = (p[0], p[1], p[2])
        self._maps[page] = (m, s)
        return m, s

    def prepare(self, segmenter=None, progress=None):
        """全ページのマップを用意する（推論が要るならここで走る）。"""
        for i, p in enumerate(self.pages, 1):
            self.page_map(p, segmenter)
            if progress:
                progress(i, len(self.pages), p)
        return self

    # ---------------------------------------------------------------- 判型
    def fit_sizes(self, band=S.BAND_MM, passes=2):
        """このアルバムの判型を、このアルバムのスキャンだけから決める。"""
        pages = [(m[2], s) for m, s in
                 (self._maps.get(p, (None, None)) for p in self.pages)
                 if m is not None]
        self.catalogue, n = S.build(pages, band=band, passes=passes)
        return self.catalogue, n

    # ---------------------------------------------------------------- 枠
    def page_boxes(self, page, snap_max_mm=4.0):
        m, s = self._maps.get(page, (None, None))
        if m is None:
            return [], None
        px = S.to_px(self.catalogue, s) if self.catalogue else ()
        bs = B.page_boxes(m[2], s, px, shape=m[2].shape, snap_max_mm=snap_max_mm)
        return bs, s

    def detect(self, snap_max_mm=4.0):
        """全ページの枠。戻り値 {ページ: {'scale':…, 'boxes':…, 'quality':…}}"""
        out = {}
        for p in self.pages:
            bs, s = self.page_boxes(p, snap_max_mm)
            if s is None:
                continue
            m = self._maps[p][0]
            out[p] = dict(scale=s, boxes=[[int(v) for v in b] for b in bs],
                          mm=[[round(B.px_to_mm(b[2], s, self.dpi), 1),
                               round(B.px_to_mm(b[3], s, self.dpi), 1)] for b in bs],
                          quality=Q.page_report(bs, m[2], s, self.catalogue))
        return out

    # ---------------------------------------------------------------- 保存
    def save(self, result, path):
        d = dict(album=self.name, dpi=self.dpi, long_side=self.long_side,
                 catalogue=[[round(c[0], 1), round(c[1], 1), int(c[2])]
                            for c in self.catalogue],
                 pages={k: dict(scale=v['scale'], boxes=v['boxes'], mm=v['mm'],
                                fit_mean=round(v['quality']['fit_mean'], 3),
                                fit_min=round(v['quality']['fit_min'], 3),
                                uncovered=round(v['quality']['uncovered'], 3))
                        for k, v in result.items()})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        return path


def crop(img, box, scale):
    """縮小画像の座標の枠で、原寸から切り出す。"""
    f = 1.0 / scale
    H, W = img.shape[:2]
    x = max(0, int(round(box[0] * f)))
    y = max(0, int(round(box[1] * f)))
    w = min(W - x, int(round(box[2] * f)))
    h = min(H - y, int(round(box[3] * f)))
    if w <= 0 or h <= 0:
        return None
    return img[y:y + h, x:x + w]


def overlay(small, boxes, scale=None, color=(0, 220, 0), thickness=2, labels=None):
    """確認用に枠を描いた画像を返す。"""
    import cv2
    vis = small.copy()
    for i, b in enumerate(boxes):
        b = [int(v) for v in b]
        cv2.rectangle(vis, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(vis, str(labels[i]), (b[0] + 4, b[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif scale:
            cv2.putText(vis, '%.0fx%.0f' % (B.px_to_mm(b[2], scale),
                                            B.px_to_mm(b[3], scale)),
                        (b[0] + 4, b[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
    return vis

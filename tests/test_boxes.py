# -*- coding: utf-8 -*-
"""合成した内側マップで、枠を作る 4 段の振る舞いを確かめる。

実データが無くても回るように、マップは手で作る。標準ライブラリの unittest
だけで書いてあるので `python -m unittest discover tests` でも
`pytest tests` でも動く。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from albumscan import boxes as B      # noqa: E402
from albumscan import quality as Q    # noqa: E402
from albumscan import sizes as S      # noqa: E402

SCALE = 0.2                            # 600dpi を長辺 1500px 相当に落とした比率


def mm(v):
    return int(round(B.mm_to_px(v, SCALE)))


def blank(w_mm=210, h_mm=297):
    return np.zeros((mm(h_mm), mm(w_mm)), np.float32)


def put(page, x_mm, y_mm, w_mm, h_mm, v=1.0):
    page[mm(y_mm):mm(y_mm + h_mm), mm(x_mm):mm(x_mm + w_mm)] = v
    return page


class TestComponents(unittest.TestCase):
    def test_finds_each_print(self):
        p = blank()
        put(p, 20, 20, 82, 117)
        put(p, 110, 20, 82, 117)
        got = B.components(p, SCALE)
        self.assertEqual(len(got), 2)
        for b in got:
            self.assertLess(abs(B.px_to_mm(b[2], SCALE) - 82), 1.5)
            self.assertLess(abs(B.px_to_mm(b[3], SCALE) - 117), 1.5)

    def test_drops_small_and_page_wide(self):
        p = blank()
        put(p, 20, 20, 82, 117)
        put(p, 120, 20, 20, 20)        # 小さすぎる（題字の札など）
        put(p, 0, 250, 210, 40)        # ページを横断する帯
        self.assertEqual(len(B.components(p, SCALE)), 1)

    def test_page_wide_component_is_rescued(self):
        """台紙まで内側と判定されたページでも、閾値を上げて写真を取り直す。"""
        p = blank()
        p[:, :] = 0.85                 # ページ全体が「内側」寄り
        put(p, 40, 60, 100, 140, 1.0)  # その中で写真だけ確信度が高い
        got = B.components(p, SCALE)
        self.assertEqual(len(got), 1)
        self.assertLess(B.px_to_mm(got[0][2], SCALE), 200)


class TestSplit(unittest.TestCase):
    def test_touching_prints_are_split(self):
        """隙間なく貼られた 2 枚は 1 つの成分になる。判型を渡せば割れる。"""
        p = blank()
        put(p, 20, 20, 164, 117)       # 82mm が 2 枚ぶん繋がった塊
        sizes = S.to_px([(82.0, 117.0, 10)], SCALE)
        got = [q for b in B.components(p, SCALE) for q in B.split(b, sizes)]
        self.assertEqual(len(got), 2)

    def test_does_not_cut_a_single_print(self):
        p = blank()
        put(p, 20, 20, 82, 117)
        sizes = S.to_px([(82.0, 117.0, 10)], SCALE)
        got = [q for b in B.components(p, SCALE) for q in B.split(b, sizes)]
        self.assertEqual(len(got), 1)


class TestSnap(unittest.TestCase):
    def test_pulls_to_album_size_when_close(self):
        b = [0, 0, mm(80.5), mm(115.0)]
        got = B.snap(b, S.to_px([(82.0, 117.0, 20)], SCALE), SCALE, snap_max_mm=4.0)
        self.assertLess(abs(B.px_to_mm(got[2], SCALE) - 82.0), 0.5)
        self.assertLess(abs(B.px_to_mm(got[3], SCALE) - 117.0), 0.5)

    def test_leaves_far_sizes_alone(self):
        """判型から遠い枠は動かさない。1 枚だけ違う紙のプリントを守るため。"""
        b = [0, 0, mm(150.0), mm(200.0)]
        got = B.snap(b, S.to_px([(82.0, 117.0, 20)], SCALE), SCALE, snap_max_mm=4.0)
        self.assertEqual(got, b)


class TestSizes(unittest.TestCase):
    def test_estimates_two_formats_from_the_album(self):
        """判型 2 種類のアルバムから、その 2 つが出てくる。"""
        pages = []
        for i in range(6):
            p = blank()
            put(p, 20, 20, 82, 117)
            put(p, 110, 20, 82, 117)
            put(p, 20, 150, 88, 89)
            pages.append((p, SCALE))
        cat, n = S.build(pages)
        self.assertGreaterEqual(n, 12)
        got = sorted((round(c[0]), round(c[1])) for c in cat)
        self.assertIn((82, 117), got)
        self.assertIn((88, 89), got)


class TestQuality(unittest.TestCase):
    def test_prefers_the_correct_frame(self):
        p = blank()
        put(p, 20, 20, 82, 117)
        tight = [mm(20), mm(20), mm(82), mm(117)]
        loose = [mm(15), mm(15), mm(92), mm(127)]
        self.assertGreater(Q.edge_fit(tight, p, SCALE), Q.edge_fit(loose, p, SCALE))

    def test_uncovered_reports_missing_print(self):
        p = blank()
        put(p, 20, 20, 82, 117)
        put(p, 110, 20, 82, 117)
        one = [[mm(20), mm(20), mm(82), mm(117)]]
        self.assertAlmostEqual(Q.uncovered(one, p), 0.5, delta=0.02)
        self.assertLess(Q.uncovered(one + [[mm(110), mm(20), mm(82), mm(117)]], p), 0.02)


if __name__ == '__main__':
    unittest.main()

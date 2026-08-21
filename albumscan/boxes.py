# -*- coding: utf-8 -*-
"""セグメンテーションの「プリントの内側」マップから、写真の矩形を作る。

**外部のプリント寸法表は使わない。** 日本の定型サイズ（E判・L判…）に当てはめる
やり方は、そのアルバムに定型でない紙が使われていたとき（トリミングされた
スナップ、記念写真の大判、中判の正方形プリント）に枠を数 mm〜数 cm ずらす。
判型は `sizes.py` が **そのアルバム自身の検出結果から** 推定する。

処理は 4 段:

  1. 内側マップを閾値で二値化して連結成分の外接矩形を取る
  2. 成分がアルバムの判型の整数倍なら等分する（隣り合わせに貼られた写真は
     1 つの成分になる）
  3. 4 辺を内側マップの確率が 0.5 を横切る位置で決め直す
  4. アルバムの判型が十分近くにあれば、寸法だけそれに合わせる（中心は保つ）

4 段目は「同じ紙のプリントなら寸法は揃うはず」という事実を使って、
1 枚ごとの検出のばらつき（±1mm）を消すためのもの。**近い判型が無ければ
何もしない**（`snap_max_mm`）。ここを無条件にすると、判型が 1 枚ごとに違う
アルバムで枠が壊れる。
"""
import cv2
import numpy as np

DPI = 600.0

# 内側マップの二値化。0.5〜0.85 はほぼ平ら、0.95 で崩れる
THR = 0.8
# これより短い辺の成分は写真ではない（題字の札・飾り・スキャンの切れ端）
MIN_MM = 40.0
# ページの縦（横）のこの割合以上にわたる成分は写真ではない。スキャンの端に写る
# 白い帯や隣のページの縁が、内側マップで写真扱いになることがある。実在する
# プリントは最大でも八つ切 216mm で、A4 相当のページ（297mm）の 73%。
PAGE_FRAC = 0.92


def mm_to_px(mm, scale, dpi=DPI):
    return mm * scale * dpi / 25.4


def px_to_mm(px, scale, dpi=DPI):
    return px / (scale * dpi / 25.4)


# ページ全体に広がってしまった成分を取り直すときの、上げていく閾値
RESCUE_THR = (0.9, 0.95, 0.98)


def _raw_components(inside, thr, min_px):
    """閾値 thr で二値化した連結成分の外接矩形。"""
    b = (inside >= thr).astype(np.uint8)
    # 5x5 のオープニング。写真どうしが数 px で繋がるのを切り、点状のノイズを消す
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(b, 8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i][:4])
        if w >= min_px and h >= min_px:
            out.append([x, y, w, h])
    return out


def components(inside, scale, min_mm=MIN_MM, thr=THR, page_frac=PAGE_FRAC):
    """内側マップの連結成分の外接矩形 [[x, y, w, h], ...]。

    ページの縦（横）の `page_frac` 以上にわたる成分は写真ではない（スキャンの端の
    白い帯、隣のページの縁、台紙ごと写真と判定された領域）。実在するプリントは
    最大でも八つ切 216mm で、A4 相当のページ（297mm）の 73%。

    **ただしそのまま捨てると、写真 1 枚しか貼っていないページで枠がゼロになる。**
    台紙が白いアルバムでは内側マップが台紙まで拾い、ページ全体が 1 つの成分に
    なることがある（記念写真ばかりの冊で 21 頁中 12 頁）。その領域だけ閾値を
    上げて取り直すと、写真の芯が残って分離できる。辺の位置は後段の `refine` が
    0.5 の交差で決め直すので、**高い閾値で取っても枠は小さくならない**。
    """
    lim = mm_to_px(min_mm, scale)
    H, W = inside.shape[:2]
    out, big = [], []
    for b in _raw_components(inside, thr, lim):
        (big if (b[3] >= page_frac * H or b[2] >= page_frac * W) else out).append(b)
    grow = int(round(mm_to_px(2.0, scale)))
    for x, y, w, h in big:
        sub = inside[y:y + h, x:x + w]
        for t in RESCUE_THR:
            got = [q for q in _raw_components(sub, t, lim)
                   if q[3] < page_frac * H and q[2] < page_frac * W]
            if not got:
                continue
            # 高い閾値で取った分だけ内側に寄っているので、少し広げてから後段の
            # refine に渡す（refine は矩形の外へは出られない）
            for q in got:
                nb = [max(0, x + q[0] - grow), max(0, y + q[1] - grow),
                      q[2] + 2 * grow, q[3] + 2 * grow]
                nb[2] = min(nb[2], W - nb[0])
                nb[3] = min(nb[3], H - nb[1])
                if not any(_iou(nb, o) > 0.3 for o in out):
                    out.append(nb)
            break
    return out


def _iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    it = ix * iy
    return it / max(1.0, float(a[2] * a[3] + b[2] * b[3] - it))


def _rel_d(w, h, W, H):
    return abs(W - w) / max(w, 1.0) + abs(H - h) / max(h, 1.0)


def split(box, sizes_px, tol=0.20, margin=0.05):
    """成分が判型の整数倍なら等分する。

    隣り合わせに貼られた写真は白フチが繋がって 1 つの成分になる。**整数倍と
    単一の判型を相対距離で競わせ、単一のほうが近ければ割らない**のが要点。
    単純に「整数倍に入ったら割る」だけだと、見逃しは減るが余分な枠が 5 倍に増える。
    """
    x, y, w, h = box
    d1 = min([_rel_d(w, h, W, H) for (W, H) in sizes_px] or [1e9])
    best, d2 = None, 1e9
    for (W, H) in sizes_px:
        W, H = int(round(W)), int(round(H))
        for k in (2, 3):
            d = _rel_d(w, h, k * W, H)
            if d < d2:
                best, d2 = ('x', k), d
            d = _rel_d(w, h, W, k * H)
            if d < d2:
                best, d2 = ('y', k), d
    if best is None or d2 > tol or d2 + margin > d1:
        return [box]
    ax, k = best
    if ax == 'x':
        return [[x + i * (w // k), y, w // k, h] for i in range(k)]
    return [[x, y + i * (h // k), w, h // k] for i in range(k)]


def refine(box, inside):
    """4 辺を、内側マップの確率が 0.5 を横切る位置で決め直す。

    二値化の閾値だけで辺を決めると、境界でなだらかに落ちる分だけ外側を拾う。
    プロファイルの立ち上がりで決めると 4 辺誤差の中央値が 0.79→0.60mm になる。
    """
    H, W = inside.shape
    x = max(0, min(int(box[0]), W - 2))
    y = max(0, min(int(box[1]), H - 2))
    w = max(2, min(int(box[2]), W - x))
    h = max(2, min(int(box[3]), H - y))
    win = inside[y:y + h, x:x + w]
    px, py = win.mean(axis=0), win.mean(axis=1)

    def cross(p, from_left):
        rng = range(len(p)) if from_left else range(len(p) - 1, -1, -1)
        for i in rng:
            if p[i] >= 0.5:
                return i
        return 0 if from_left else len(p) - 1

    l, r = cross(px, True), cross(px, False)
    t, bo = cross(py, True), cross(py, False)
    return [x + l, y + t, max(2, r - l + 1), max(2, bo - t + 1)]


def snap(box, sizes_px, scale, snap_max_mm=4.0, area_lo=0.55, area_hi=1.6):
    """寸法だけアルバムの判型に合わせる（中心は保つ）。向きは成分に従う。

    **近い判型が無ければ何もしない。** これが定型サイズ表に当てはめる方式との
    決定的な違いで、判型が 1 枚ごとに違うアルバム（記念写真帳、トリミングされた
    スナップの束）では検出した寸法をそのまま使う。
    """
    x, y, w, h = box
    a0 = float(max(w * h, 1))
    lim = mm_to_px(snap_max_mm, scale)
    best, bd = None, 1e18
    for (W, H) in sizes_px:
        W, H = int(round(W)), int(round(H))
        if not (area_lo <= (W * H) / a0 <= area_hi):
            continue
        # 縦横は成分に従う（ほぼ正方形の中判は両方見る）
        if (W > H) != (w > h) and abs(w - h) > 0.15 * max(w, h):
            continue
        d = abs(W - w) + abs(H - h)
        if d < bd:
            best, bd = (W, H), d
    if best is None or bd > lim:
        return list(box)
    W, H = best
    return [int(round(x + w / 2.0 - W / 2.0)),
            int(round(y + h / 2.0 - H / 2.0)), W, H]


def page_boxes(inside, scale, sizes_px=(), forbid=None, shape=None,
               snap_max_mm=4.0):
    """1 ページ分の枠。

    inside      : 内側チャンネル（float 0..1、縮小画像と同じ大きさ）
    scale       : 縮小率（mm 換算に使う）
    sizes_px    : そのアルバムの判型 [(w, h), ...]（縦横は展開済み・画素）
    forbid      : 綴じ代など「枠を置いてはいけない列」の bool 配列
    snap_max_mm : 判型に合わせる上限。0 で「合わせない」
    """
    out = components(inside, scale)
    if sizes_px:
        out = [q for b in out for q in split(b, sizes_px)]
    out = [refine(b, inside) for b in out]
    if sizes_px and snap_max_mm > 0:
        out = [snap(b, sizes_px, scale, snap_max_mm) for b in out]
    if forbid is not None and len(forbid):
        keep = []
        for b in out:
            cx = int(np.clip(b[0] + b[2] // 2, 0, len(forbid) - 1))
            if not forbid[cx]:
                keep.append(b)
        out = keep
    if shape is not None:
        H, W = shape[:2]
        out = [b for b in out if b[2] >= 8 and b[3] >= 8 and b[0] < W and b[1] < H]
    return out

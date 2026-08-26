# -*- coding: utf-8 -*-
"""正解データを持たずに枠の良し悪しを測る。

人が引いた枠を「正解」として最適化すると、**その正解が規格サイズに引きずられて
いた場合、引きずられた枠を再現する方向に最適化が進む**。実際に、手で作った
1035 ページの登録枠を正解として測ると 87% が「そのまま使える枠」だったが、
その中には紙の外形から 5〜6mm 外れた枠が混ざっていた（枠を置くときに
定型サイズのドロップダウンから選んでいたため）。

ここでは正解を使わず、**枠が写真の輪郭に乗っているか**を画像から直接測る。

  edge_fit    枠の内側の帯と外側の帯の「内側らしさ」の差。1 に近いほど、
              枠が紙の縁にぴったり乗っている
  uncovered   どの枠にも入らなかった「内側」画素の割合（＝検出漏れ）
  overlap     枠どうしの重なり（正しい枠は重ならない）
  size_dev    アルバムの判型からの寸法のずれ mm（同じ紙なら揃うはず）
"""
import numpy as np

from . import boxes as B


def edge_fit(box, inside, scale, band_mm=1.5):
    """枠が輪郭に乗っている度合い（-1..1）。"""
    H, W = inside.shape[:2]
    x, y, w, h = [int(v) for v in box]
    t = max(1, int(round(B.mm_to_px(band_mm, scale))))
    if w <= 2 * t + 2 or h <= 2 * t + 2:
        return 0.0
    inner_all = inside[max(0, y):y + h, max(0, x):x + w]
    core = inside[y + t:y + h - t, x + t:x + w - t]
    n_all, n_core = inner_all.size, core.size
    if n_all <= n_core:
        return 0.0
    ring_in = (inner_all.sum() - core.sum()) / float(n_all - n_core)

    ox1, oy1 = max(0, x - t), max(0, y - t)
    ox2, oy2 = min(W, x + w + t), min(H, y + h + t)
    outer = inside[oy1:oy2, ox1:ox2]
    n_out = outer.size - n_all
    if n_out <= 0:
        return 0.0
    ring_out = (outer.sum() - inner_all.sum()) / float(n_out)
    return float(ring_in - ring_out)


def uncovered(boxes, inside, thr=B.THR):
    """どの枠にも入らない「内側」画素の割合。"""
    m = inside >= thr
    tot = float(m.sum())
    if tot < 1:
        return 0.0
    cov = np.zeros(inside.shape[:2], bool)
    for x, y, w, h in boxes:
        cov[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = True
    return float((m & ~cov).sum() / tot)


def overlap(boxes):
    """枠どうしの重なり面積 ÷ 枠の総面積。"""
    if len(boxes) < 2:
        return 0.0
    area = sum(max(0, b[2]) * max(0, b[3]) for b in boxes) or 1
    ov = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            ov += ix * iy
    return float(ov) / float(area)


def size_dev(box, catalogue, scale):
    """いちばん近い判型との |Δ短辺|+|Δ長辺| mm。判型が無ければ None。"""
    if not catalogue:
        return None
    w, h = B.px_to_mm(box[2], scale), B.px_to_mm(box[3], scale)
    sh, lo = min(w, h), max(w, h)
    return min(abs(sh - c[0]) + abs(lo - c[1]) for c in catalogue)


def page_report(boxes, inside, scale, catalogue=()):
    """1 ページ分のまとめ。"""
    fits = [edge_fit(b, inside, scale) for b in boxes]
    devs = [d for d in (size_dev(b, catalogue, scale) for b in boxes) if d is not None]
    return dict(
        n=len(boxes),
        fit_min=float(min(fits)) if fits else 0.0,
        fit_mean=float(np.mean(fits)) if fits else 0.0,
        uncovered=uncovered(boxes, inside),
        overlap=overlap(boxes),
        size_dev_max=float(max(devs)) if devs else 0.0,
        fits=fits, devs=devs)


# ------------------------------------------------ 人に見てもらうページを選ぶ
#
# 実データ 1035 ページでの分布（新方式）:
#   fit_min    p5 0.247  p10 0.316  p25 0.384  中央 0.445  p90 0.508
#   uncovered  中央 0.007  p75 0.016  p90 0.028
#   overlap    ほぼ 0（正しい枠は重ならない）
#
# 下のしきい値だと **全体の 8% が「要確認」** になる。人が確定させた枠と
# 枠の数が食い違うページのうち 46% がこの 8% に入っていた。**残り半分は
# 指標では拾えない**ので、「ここだけ見れば完璧」ではなく「ここから見ると
# 効率がよい」という道具として使うこと。
FIT_MIN = 0.30          # これを下回る枠が 1 つでもあれば要確認
UNCOVERED = 0.10        # 検出漏れの疑い
OVERLAP = 0.02          # 枠が重なっている
EMPTY_OK = True         # 枠が 0 個のページ（題字だけ・白紙）は要確認にしない


def needs_review(rep, fit_min=FIT_MIN, unc=UNCOVERED, ov=OVERLAP):
    """要確認なら理由の一覧を、問題なければ空リストを返す。"""
    why = []
    if rep['n'] == 0:
        if not EMPTY_OK or rep['uncovered'] > unc:
            why.append('枠がない')
        return why
    if rep['fit_min'] < fit_min:
        why.append('枠が輪郭に乗っていない (fit %.2f)' % rep['fit_min'])
    if rep['uncovered'] > unc:
        why.append('検出漏れの疑い (未被覆 %.0f%%)' % (100 * rep['uncovered']))
    if rep['overlap'] > ov:
        why.append('枠が重なっている (%.0f%%)' % (100 * rep['overlap']))
    return why


def priority(rep):
    """要確認ページの並び順。大きいほど先に見る。"""
    return (max(0.0, FIT_MIN - rep['fit_min']) * 3.0
            + max(0.0, rep['uncovered'] - UNCOVERED) * 2.0
            + max(0.0, rep['overlap'] - OVERLAP))

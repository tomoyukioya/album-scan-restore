# -*- coding: utf-8 -*-
"""褪せたプリント 1 枚の色を戻す。

**枠ごとに独立して補正する**のが要点。1 ページに貼られた写真は、撮った時期も
現像所も褪せ方も違う。ページ全体をまとめて補正すると、いちばん褪せた 1 枚に
引きずられて他が転ぶ。

手順は 4 段:

  1. チャンネルごとにパーセンタイルで伸張する（褪色で狭くなった分布を戻す）
  2. Lab の a/b の中央値だけ引いて、残った色かぶりを部分的に中和する
  3. 彩度と S 字コントラストを軽く掛ける
  4. ハイライトの色を白へ寄せる

**中和を「中央値」で行うことの弱点**は分かっていて、対処も入れてある:

  * 画面の 3〜4 割が「ほぼ黒」だと、そこは Lab で中立なので中央値が 0 に
    引かれ、中間調に残った本物のかぶりが直らない。→ L>15 の画素だけで測る。
  * 逆に、画面の大半を**実在する強い色**（芝生、緑のフェルト、赤い花）が
    占めると、その色をかぶりと誤認して打ち消す。これは統計だけでは判別
    できないので、`ai` / `bi` を外から与えて上書きできるようにしてある。
  * 雪面のように「大部分は中立で、影だけ強く残る」写真では、一律のシフトが
    効かない。→ `prop` を与えると「既存の色ズレを prop 倍に縮める」方式に
    切り替わる。
"""
import math

import cv2
import numpy as np

# 補正の強さ（3 段階）
LEVELS = dict(
    natural=dict(plo=0.6, phi=99.4, st=1.0, ab=0.72, abmax=19, sat=1.07, ctr=0.05),
    strong=dict(plo=0.3, phi=99.7, st=1.0, ab=0.85, abmax=22, sat=1.18, ctr=0.10),
    light=dict(plo=1.5, phi=98.5, st=0.75, ab=0.35, abmax=9, sat=1.02, ctr=0.02),
)

# 白黒プリントと分かっているときの彩度の残し方（ほぼ完全に落とす）
MONO_PROP = 0.05

# 「シアンに転んだ」とみなす a の中央値と、そのときのやり直しの強さ。
# ほぼ白まで褪せた写真をチャンネル別に強く伸張すると、死んでいるチャンネルを
# 持ち上げて**元のデータに無いシアン**を作る。出来上がりを見て 1 回だけ
# やり直す後判定なので、外しても「補正が弱まる」だけで色は失われない。
CYAN_A, CYAN_ST, CYAN_AB = -6.0, 0.5, 1.0

# かぶりの推定から外す暗部の明度（Lab の L、0..100）
DARK_L = 15.0


class Result(object):
    """補正結果と、そのとき使った値（GUI やログで見せるため）。"""

    def __init__(self, image, ai, bi, mono, cyan_fix):
        self.image = image
        self.ai, self.bi = ai, bi
        self.mono, self.cyan_fix = mono, cyan_fix


def restore_patch(patch, level='natural', valid=None, opts=None, _retry=False):
    """プリント 1 枚（BGR uint8）を補正して Result を返す。

    valid : 補正の統計に使う画素の真偽マップ（台紙を除きたいときなど）
    opts  : 1 枚ごとの上書き
            ab / abmax / sat / ctr / st … 効き方
            ai / bi                      … かぶりの量を直接指定する
            prop                         … 比例縮小方式に切り替える
            mono                         … 白黒プリントとして扱う
    """
    P = dict(LEVELS[level])
    o = opts or {}
    for k in ('ab', 'abmax', 'sat', 'ctr', 'st'):
        if o.get(k) is not None:
            P[k] = o[k]

    x = patch.astype(np.float32)
    h, w = x.shape[:2]
    iy, ix = int(h * 0.02), int(w * 0.02)
    v = np.ones((h, w), bool) if valid is None else valid
    vi = v[iy:h - iy, ix:w - ix]

    # ---- 1. チャンネルごとの伸張
    smp = x[iy:h - iy, ix:w - ix][vi]
    if len(smp) < 1000:
        smp = x[iy:h - iy, ix:w - ix].reshape(-1, 3)
    if len(smp) > 300000:
        smp = smp[np.random.RandomState(0).choice(len(smp), 300000, replace=False)]
    lo = np.percentile(smp, P['plo'], axis=0)
    hi = np.maximum(np.percentile(smp, P['phi'], axis=0), lo + 18)
    gain = np.clip((252.0 - 3.0) / (hi - lo), 0.55, 3.6)
    g = 1.0 + (gain - 1.0) * P['st']
    off = (3.0 - lo * gain) * P['st']
    out = x * g + off
    # 白飛びを作らないよう、上のほうだけ tanh で寝かせる
    k = 238.0
    m = out > k
    out[m] = k + (255.0 - k) * np.tanh((out[m] - k) / (255.0 - k))
    out = np.clip(out, 0, 255)

    # ---- 2. 残った色かぶりの中和
    lab = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    A, B = lab[..., 1] - 128, lab[..., 2] - 128
    Ai = A[iy:h - iy, ix:w - ix][vi] if vi.sum() > 1000 else A[iy:h - iy, ix:w - ix]
    Bi = B[iy:h - iy, ix:w - ix][vi] if vi.sum() > 1000 else B[iy:h - iy, ix:w - ix]
    # ほぼ黒の画素は Lab で中立なので、暗い写真では中央値を 0 へ引いてしまう
    not_black = (lab[..., 0] * 100.0 / 255.0) > DARK_L
    vi2 = (v & not_black)[iy:h - iy, ix:w - ix]
    if vi2.sum() > 500:
        Ai, Bi = A[iy:h - iy, ix:w - ix][vi2], B[iy:h - iy, ix:w - ix][vi2]
    ai = float(np.clip(np.median(Ai), -P['abmax'], P['abmax']))
    bi = float(np.clip(np.median(Bi), -P['abmax'], P['abmax']))
    if o.get('ai') is not None:
        ai = float(o['ai'])
    if o.get('bi') is not None:
        bi = float(o['bi'])

    prop = o.get('prop')
    if prop is None and o.get('mono'):
        prop = MONO_PROP
    mono_used = o.get('prop') is None and prop is not None
    if prop is not None:
        f = float(prop)
        lab[..., 1] = np.clip(128.0 + (lab[..., 1] - 128.0) * f, 0, 255)
        lab[..., 2] = np.clip(128.0 + (lab[..., 2] - 128.0) * f, 0, 255)
    else:
        lab[..., 1] = np.clip(lab[..., 1] - ai * P['ab'], 0, 255)
        lab[..., 2] = np.clip(lab[..., 2] - bi * P['ab'], 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

    # ---- 3. 彩度と S 字コントラスト
    if P['sat'] != 1.0:
        gray = out @ np.array([0.114, 0.587, 0.299], np.float32)
        out = gray[..., None] + (out - gray[..., None]) * P['sat']
    if P['ctr']:
        n = np.clip(out / 255.0, 0, 1)
        n = np.clip(n - P['ctr'] * np.sin(2 * math.pi * n) / (2 * math.pi), 0, 1)
        out = n * 255.0
    out = np.clip(out, 0, 255)

    # ---- 4. ハイライトの色を白へ寄せる
    # 褪せたプリントの明るい所には本物の色がほとんど残っていない。そこに乗る
    # 強い色は、死にかけたチャンネルを伸張した副産物。白フチも中立に保てる。
    lab2 = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    Lp = lab2[..., 0] * 100.0 / 255.0
    wgt = np.clip((Lp - 80.0) / 20.0, 0, 1) ** 1.2
    lab2[..., 1:] = 128.0 + (lab2[..., 1:] - 128.0) * (1.0 - 0.65 * wgt)[..., None]
    out = cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR).astype(np.float32)

    # ---- 出来上がりが青緑に転んでいたら、伸張を弱めて 1 度だけやり直す
    if not _retry and o.get('st') is None and o.get('ab') is None \
            and o.get('prop') is None:
        av = lab2[..., 1][v] if v.shape == lab2.shape[:2] and v.sum() > 500 \
            else lab2[..., 1]
        if float(np.median(av)) - 128.0 < CYAN_A:
            o2 = dict(o)
            o2['st'], o2['ab'] = CYAN_ST, CYAN_AB
            r = restore_patch(patch, level, valid, o2, _retry=True)
            r.cyan_fix = True
            return r
    return Result(np.clip(out, 0, 255).astype(np.uint8), ai, bi, mono_used, False)


def restore_page(image, boxes, scale, level='natural', box_opts=None, mono=False):
    """ページ画像の中の枠を 1 つずつ補正して、ページ全体を返す。

    枠の外（台紙）はそのまま残す。アルバムの見開きとして残したいときのため。
    """
    out = image.copy()
    f = 1.0 / scale
    H, W = image.shape[:2]
    used = []
    for i, b in enumerate(boxes):
        x = max(0, int(round(b[0] * f)))
        y = max(0, int(round(b[1] * f)))
        w = min(W - x, int(round(b[2] * f)))
        h = min(H - y, int(round(b[3] * f)))
        if w < 8 or h < 8:
            continue
        o = dict((box_opts or {}).get(i) or {})
        if mono and 'mono' not in o:
            o['mono'] = True
        r = restore_patch(image[y:y + h, x:x + w], level, opts=o)
        out[y:y + h, x:x + w] = r.image
        used.append(r)
    return out, used

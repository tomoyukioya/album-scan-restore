# -*- coding: utf-8 -*-
"""そのアルバムに実際に貼られているプリントの寸法（判型）を、
**アルバム自身のスキャンだけから** 推定する。

**なぜ定型サイズ表を使わないか**

「日本の定型プリント（E判 82.5x117、L判 89x127 …）のどれかに当てはめる」
という作り方は、当てはまるアルバムでは強力に働く一方、外れるアルバムでは
枠を数 mm から数 cm ずらす。実データ（36 冊）で確かめた外れ方は 3 種類:

  * **定型に無い紙** — 中判の正方形プリント、ラボの縁印字が入る 6x6 判、
    トリミングされたスナップ。近い定型に押し込むと台紙まで枠に入る。
  * **判型が 1 枚ごとに違うアルバム** — 記念写真の大判ばかりを集めた冊は、
    実測が 158x116 / 204x152 / 211x161 とすべて違い、共通の判型が無い。
    2 種類のクラスタに全部を吸着させると 24 枠中 20 枠が寸法違いになった。
  * **同じアルバムに近い判型が 2 つある** — 80x108 と 82.5x117 が混在すると、
    検出寸法がその中間に来たとき、どちらに寄せるかを間違える。

一方、**同じアルバムの中では紙は揃っている**（同じ時期に同じラボで焼いた）。
検出した寸法をアルバム単位で集めると、判型の数だけ密集した山ができる。
その山を数えて代表値を取れば、外部の表を持ち込まずに判型が決まる。

推定の手順:

  1. 各ページの成分から、**信用できる標本だけ**を選ぶ
     （縦横比・寸法・充填率・ページ端に接していないか）
  2. (短辺, 長辺) mm を平均シフトでクラスタリング（バンド幅 2mm）
  3. 他のクラスタのちょうど整数倍のクラスタは「繋がった 2〜3 枚の塊」として捨てる
  4. 支持の薄いクラスタを捨てる
  5. 1 巡目の判型で塊を割り直し、2 巡目のクラスタを最終結果にする
"""
import numpy as np

from . import boxes as B

# 標本として使う条件。**プリントの実在寸法には踏み込まない**（物理的にありえない
# ものを落とすだけ）。
MIN_SHORT_MM = 35.0     # これより小さい紙のプリントは無い
MAX_LONG_MM = 240.0     # 六つ切 254 まであるが、それ以上はページ全体
MAX_RATIO = 1.7         # 長辺÷短辺の上限。35mm 判で 1.5、パノラマは別扱い
MIN_FILL = 0.90         # 外接矩形のうち内側マップが占める割合。低いと L 字の塊
EDGE_MM = 2.0           # ページ端にこれだけ近い辺を持つ成分は切れている疑い

BAND_MM = 2.0           # 平均シフトのバンド幅
MIN_SUPPORT = 3         # クラスタに必要な標本数
MIN_FRAC = 0.02         # 全標本に占める割合の下限


def page_samples(inside, scale, sizes_px=()):
    """1 ページ分の (短辺mm, 長辺mm) の標本。sizes_px を渡すと分割も行う。"""
    H, W = inside.shape[:2]
    bs = B.components(inside, scale)
    if sizes_px:
        bs = [q for b in bs for q in B.split(b, sizes_px)]
    bs = [B.refine(b, inside) for b in bs]
    e = B.mm_to_px(EDGE_MM, scale)
    out = []
    for x, y, w, h in bs:
        wm, hm = B.px_to_mm(w, scale), B.px_to_mm(h, scale)
        sh, lo = min(wm, hm), max(wm, hm)
        if sh < MIN_SHORT_MM or lo > MAX_LONG_MM or lo / max(sh, 1e-6) > MAX_RATIO:
            continue
        if x < e or y < e or x + w > W - e or y + h > H - e:
            continue        # ページの端で切れている（スキャンのはみ出し）
        if float((inside[y:y + h, x:x + w] >= B.THR).mean()) < MIN_FILL:
            continue        # 外接矩形の中が埋まっていない＝L 字や斜めの塊
        out.append((sh, lo))
    return out


def cluster(obs, band=BAND_MM, iters=30):
    """(短辺, 長辺) の平均シフト。戻り値 [(短辺, 長辺, 標本数), ...] 多い順。

    k-means と違ってクラスタ数を先に決めなくてよい。判型が何種類あるかは
    アルバムごとに違い（1 種類の冊も 8 種類の冊もある）、決め打ちできない。
    """
    if not obs:
        return []
    P = np.asarray(obs, float)
    C = P.copy()
    for _ in range(iters):
        d = np.linalg.norm(C[:, None, :] - P[None, :, :], axis=2)
        w = np.exp(-(d / band) ** 2)
        C2 = (w[:, :, None] * P[None, :, :]).sum(1) / w.sum(1)[:, None]
        done = np.abs(C2 - C).max() < 1e-3
        C = C2
        if done:
            break
    cent, lab = [], []
    for c in C:
        for i, e in enumerate(cent):
            if np.linalg.norm(c - e) < band * 0.5:
                lab.append(i)
                break
        else:
            cent.append(c)
            lab.append(len(cent) - 1)
    lab = np.asarray(lab)
    out = []
    for i in range(len(cent)):
        pts = P[lab == i]
        if len(pts):
            # 代表値は中央値（外れ値に強い）
            out.append((float(np.median(pts[:, 0])), float(np.median(pts[:, 1])),
                        int(len(pts))))
    return sorted(out, key=lambda t: -t[2])


def drop_multiples(cl, tol=0.08):
    """他の判型のちょうど整数倍のクラスタを捨てる（繋がった 2〜3 枚の塊）。"""
    out = []
    for i, (s, l, n) in enumerate(cl):
        dup = False
        for j, (s2, l2, n2) in enumerate(cl):
            if i == j or n2 < n:
                continue
            for k in (2, 3):
                if abs(s - s2) / s2 < tol and abs(l - k * l2) / (k * l2) < tol:
                    dup = True
                if abs(l - k * s2) / (k * s2) < tol and abs(s - l2) / l2 < tol:
                    dup = True
        if not dup:
            out.append((s, l, n))
    return out


def keep(cl, min_support=MIN_SUPPORT, min_frac=MIN_FRAC):
    tot = sum(c[2] for c in cl) or 1
    return [c for c in cl if c[2] >= min_support and c[2] / tot >= min_frac]


def to_px(catalogue, scale):
    """判型 [(短辺mm, 長辺mm, 枚数), ...] を画素の [(w, h), ...] に（縦横とも）。"""
    out = []
    for c in catalogue:
        a, b = B.mm_to_px(c[0], scale), B.mm_to_px(c[1], scale)
        out.append((a, b))
        out.append((b, a))
    return out


def build(pages, band=BAND_MM, passes=2):
    """アルバムの判型を推定する。

    pages : [(inside, scale), ...]  そのアルバムの全ページ
    戻り値: ([(短辺mm, 長辺mm, 標本数), ...], 標本数)
    """
    obs = [v for ins, sc in pages for v in page_samples(ins, sc)]
    cl = keep(drop_multiples(cluster(obs, band)))
    for _ in range(passes - 1):
        if not cl:
            break
        # 1 巡目の判型で塊を割ってから測り直す。隣り合わせに貼られた写真は
        # 1 成分になるので、割らないと標本から落ちたままになる。
        obs2 = []
        for ins, sc in pages:
            obs2 += page_samples(ins, sc, to_px(cl, sc))
        cl2 = keep(drop_multiples(cluster(obs2, band)))
        if cl2:
            cl, obs = cl2, obs2
    return cl, len(obs)

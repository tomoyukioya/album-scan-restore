# -*- coding: utf-8 -*-
"""コマンドライン。

    python -m albumscan detect  <アルバムのフォルダ> --model models/boundary.pt
    python -m albumscan crop    <アルバムのフォルダ> --boxes boxes.json --out 切り出し
    python -m albumscan overlay <アルバムのフォルダ> --boxes boxes.json --out 枠確認

`detect` は 1 冊分のページをまとめて見て、そのアルバムの判型を推定してから
枠を決める。**プリント寸法の一覧は持っていない**（`sizes.py` の説明を参照）。
"""
import argparse
import json
import os
import sys

from .album import Album, crop, overlay
from .imageio import imread, imwrite


def _albums(root, recursive):
    """アルバムのフォルダを列挙する。画像が直下にあるフォルダをアルバムとみなす。"""
    from .imageio import list_pages
    if list_pages(root):
        return [root]
    if not recursive:
        return []
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and list_pages(p):
            out.append(p)
    return out


def cmd_detect(a):
    seg = None
    albums = _albums(a.folder, a.recursive)
    if not albums:
        print('画像のあるフォルダが見つかりません: %s' % a.folder)
        return 1
    for folder in albums:
        alb = Album(folder, cache_dir=a.cache, dpi=a.dpi, long_side=a.long_side)
        if not alb.pages:
            continue
        need = any(alb.page_map(p)[0] is None for p in alb.pages[:1])
        if need and seg is None:
            if not a.model:
                print('--model が要ります（輪郭マップのキャッシュがありません）')
                return 1
            from .segment import Segmenter
            seg = Segmenter(a.model, device=a.device)
            print('モデル: %s %s' % (a.model, seg.info))
        alb.prepare(seg, progress=(lambda i, n, p: (
            print('  %d/%d %s' % (i, n, p), flush=True) if i % 10 == 0 else None))
            if a.verbose else None)
        cat, nsamp = alb.fit_sizes()
        print('%s  %d 頁  標本 %d  判型: %s'
              % (alb.name, len(alb.pages), nsamp,
                 '  '.join('%.1fx%.1f×%d' % c for c in cat) or '（推定できず）'))
        res = alb.detect(snap_max_mm=a.snap_max_mm)
        out = a.out or os.path.join(folder, 'boxes.json')
        if len(albums) > 1 and a.out:
            out = os.path.join(a.out, alb.name + '.json')
        alb.save(res, out)
        nb = sum(len(v['boxes']) for v in res.values())
        fits = [f for v in res.values() for f in v['quality']['fits']]
        print('  枠 %d 個 → %s   輪郭との一致 平均%.3f 最小%.3f'
              % (nb, out, sum(fits) / max(len(fits), 1), min(fits) if fits else 0))
    return 0


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def cmd_crop(a):
    d = _load(a.boxes)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for page, v in sorted(d['pages'].items()):
        img = imread(os.path.join(a.folder, page))
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(page))[0]
        for i, b in enumerate(v['boxes'], 1):
            c = crop(img, b, v['scale'])
            if c is None:
                continue
            imwrite(os.path.join(a.out, '%s_%02d.jpg' % (stem, i)), c,
                    [__import__('cv2').IMWRITE_JPEG_QUALITY, a.quality])
            n += 1
    print('%d 枚を切り出しました → %s' % (n, a.out))
    return 0


def cmd_overlay(a):
    from .imageio import downscale
    d = _load(a.boxes)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for page, v in sorted(d['pages'].items()):
        img = imread(os.path.join(a.folder, page))
        if img is None:
            continue
        sm, _s = downscale(img, d.get('long_side', 1500))
        vis = overlay(sm, v['boxes'], v['scale'])
        imwrite(os.path.join(a.out, os.path.basename(page)), vis)
        n += 1
    print('%d 頁の確認画像を書きました → %s' % (n, a.out))
    return 0


def cmd_restore(a):
    from .restore import restore_page
    d = _load(a.boxes)
    os.makedirs(a.out, exist_ok=True)
    import cv2
    n = 0
    for page, v in sorted(d['pages'].items()):
        img = imread(os.path.join(a.folder, page))
        if img is None:
            continue
        out, used = restore_page(img, v['boxes'], v['scale'], level=a.level,
                                 mono=a.mono)
        imwrite(os.path.join(a.out, os.path.basename(page)), out,
                [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        n += 1
        if a.verbose:
            print('  %s  枠%d  かぶり a=%s b=%s'
                  % (page, len(used), [round(r.ai, 1) for r in used],
                     [round(r.bi, 1) for r in used]))
    print('%d 頁を補正しました → %s' % (n, a.out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='albumscan', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')

    d = sub.add_parser('detect', help='枠を検出して boxes.json に書く')
    d.add_argument('folder')
    d.add_argument('--model', default=None, help='学習済みモデル (.pt)')
    d.add_argument('--cache', default=None, help='輪郭マップの置き場（既定は <folder>/_cache）')
    d.add_argument('--out', default=None)
    d.add_argument('--device', default=None, help='cuda / cpu')
    d.add_argument('--dpi', type=float, default=600.0, help='スキャン解像度')
    d.add_argument('--long-side', type=int, default=1500, help='検出に使う縮小画像の長辺')
    d.add_argument('--snap-max-mm', type=float, default=4.0,
                   help='判型に寸法を合わせる上限 mm。0 で「合わせない」')
    d.add_argument('-r', '--recursive', action='store_true',
                   help='直下のフォルダをそれぞれ 1 冊として処理する')
    d.add_argument('-v', '--verbose', action='store_true')
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser('crop', help='boxes.json に従って原寸から切り出す')
    c.add_argument('folder')
    c.add_argument('--boxes', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--quality', type=int, default=95)
    c.set_defaults(func=cmd_crop)

    r = sub.add_parser('restore', help='枠の中の色を戻したページを書く')
    r.add_argument('folder')
    r.add_argument('--boxes', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--level', default='natural', choices=('light', 'natural', 'strong'))
    r.add_argument('--mono', action='store_true', help='白黒プリントのアルバム')
    r.add_argument('--quality', type=int, default=95)
    r.add_argument('-v', '--verbose', action='store_true')
    r.set_defaults(func=cmd_restore)

    o = sub.add_parser('overlay', help='枠を描いた確認画像を書く')
    o.add_argument('folder')
    o.add_argument('--boxes', required=True)
    o.add_argument('--out', required=True)
    o.set_defaults(func=cmd_overlay)

    a = ap.parse_args(argv)
    if not getattr(a, 'func', None):
        ap.print_help()
        return 1
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main())

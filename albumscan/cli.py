# -*- coding: utf-8 -*-
"""コマンドライン。

いちばん手軽なのは `run` で、1 冊分をまとめて処理して、
**人が見るべきページの一覧**まで出します。

    python -m albumscan run <アルバムのフォルダ> --model models/boundary.pt --out out

個別に動かしたいときは detect → overlay / restore / crop / report の順に。

枠は 1 冊分をまとめて見てから決めます（そのアルバムの判型を推定するため）。
**1 ページだけ渡すと判型が決まりません。**
"""
import argparse
import json
import os
import sys

import cv2

from . import quality as Q
from .album import Album, crop, overlay
from .imageio import downscale, imread, imwrite, list_pages


# ------------------------------------------------------------------ 共通
def _albums(root, recursive):
    """アルバムのフォルダを列挙する。画像が直下にあるフォルダを 1 冊とみなす。"""
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


def _load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _segmenter(a, needed):
    if not needed:
        return None
    if not a.model:
        print('--model が要ります（輪郭マップのキャッシュがありません）')
        return False
    from .segment import Segmenter
    seg = Segmenter(a.model, device=a.device)
    print('モデル: %s' % a.model)
    return seg


def _detect_one(folder, a, seg):
    """1 冊を検出して (Album, 結果) を返す。"""
    alb = Album(folder, cache_dir=a.cache, dpi=a.dpi, long_side=a.long_side)
    if not alb.pages:
        return None, None
    prog = None
    if a.verbose:
        def prog(i, n, p):
            if i % 10 == 0 or i == n:
                print('    輪郭マップ %d/%d' % (i, n), flush=True)
    alb.prepare(seg, progress=prog)
    cat, nsamp = alb.fit_sizes()
    print('%s  %d 頁  標本 %d  判型: %s'
          % (alb.name, len(alb.pages), nsamp,
             '  '.join('%.1fx%.1f×%d' % c for c in cat) or '（推定できず・検出寸法をそのまま使う）'))
    res = alb.detect(snap_max_mm=a.snap_max_mm)
    nb = sum(len(v['boxes']) for v in res.values())
    fits = [f for v in res.values() for f in v['quality']['fits']]
    print('  枠 %d 個   輪郭との一致 平均%.3f 最小%.3f'
          % (nb, sum(fits) / max(len(fits), 1), min(fits) if fits else 0))
    return alb, res


# ------------------------------------------------------------------ detect
def cmd_detect(a):
    albums = _albums(a.folder, a.recursive)
    if not albums:
        print('画像のあるフォルダが見つかりません: %s' % a.folder)
        return 1
    seg = None
    for folder in albums:
        probe = Album(folder, cache_dir=a.cache, dpi=a.dpi, long_side=a.long_side)
        if probe.pages and seg is None:
            need = probe.page_map(probe.pages[0])[0] is None
            seg = _segmenter(a, need)
            if seg is False:
                return 1
        alb, res = _detect_one(folder, a, seg)
        if alb is None:
            continue
        out = a.out or os.path.join(folder, 'boxes.json')
        if len(albums) > 1 and a.out:
            out = os.path.join(a.out, alb.name + '.json')
        alb.save(res, out)
        print('  → %s' % out)
    return 0


# ------------------------------------------------------------------ report
def _review_rows(d):
    rows = []
    for page, v in d['pages'].items():
        rep = v.get('quality') or {}
        why = v.get('review')
        if why is None:
            why = Q.needs_review(rep)
        if why:
            rows.append((Q.priority(rep), page, rep, why))
    rows.sort(key=lambda r: -r[0])
    return rows


def _write_report(d, path, album_name=None):
    rows = _review_rows(d)
    n = len(d['pages'])
    lines = []
    lines.append('# 要確認のページ%s' % (' — %s' % album_name if album_name else ''))
    lines.append('')
    lines.append('全 %d 頁のうち %d 頁（%.0f%%）。上から順に見ると効率がよい。'
                 % (n, len(rows), 100.0 * len(rows) / max(n, 1)))
    lines.append('')
    lines.append('判型: %s' % ('  '.join('%.1fx%.1fmm x%d' % tuple(c)
                                         for c in d.get('catalogue') or [])
                               or '（推定できず）'))
    lines.append('')
    if not rows:
        lines.append('要確認のページはありません。')
    for _p, page, rep, why in rows:
        lines.append('- %s  枠%d  %s' % (page, rep.get('n', 0), ' / '.join(why)))
    lines.append('')
    lines.append('## 指標の意味')
    lines.append('')
    lines.append('- fit … 枠の内側の帯と外側の帯の「プリント内側らしさ」の差。'
                 '1 に近いほど枠が紙の縁に乗っている（実データの中央値 0.45）')
    lines.append('- 未被覆 … どの枠にも入らなかったプリント画素の割合（検出漏れ）')
    lines.append('- 重なり … 枠どうしの重なり。正しい枠は重ならない')
    lines.append('')
    lines.append('**この一覧で全部の不備が拾えるわけではありません。** 実データでは、'
                 '人が直したページの約半分がここに入りました。残りは目で見るしかありません。')
    txt = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)
    return rows


def cmd_report(a):
    d = _load(a.boxes)
    rows = _review_rows(d)
    n = len(d['pages'])
    print('全 %d 頁のうち要確認 %d 頁（%.0f%%）'
          % (n, len(rows), 100.0 * len(rows) / max(n, 1)))
    for _p, page, rep, why in rows[:a.limit]:
        print('  %-28s 枠%d  %s' % (page, rep.get('n', 0), ' / '.join(why)))
    if len(rows) > a.limit:
        print('  …ほか %d 頁' % (len(rows) - a.limit))
    if a.out:
        _write_report(d, a.out)
        print('→ %s' % a.out)
    return 0


# ------------------------------------------------------------------ 出力系
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
                    [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            n += 1
    print('%d 枚を切り出しました → %s' % (n, a.out))
    return 0


def cmd_overlay(a):
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
                     [round(r.bi, 1) for r in used]), flush=True)
        elif n % 10 == 0:
            print('  %d/%d' % (n, len(d['pages'])), flush=True)
    print('%d 頁を補正しました → %s' % (n, a.out))
    return 0


# ------------------------------------------------------------------ run
def cmd_run(a):
    from .restore import restore_page
    albums = _albums(a.folder, a.recursive)
    if not albums:
        print('画像のあるフォルダが見つかりません: %s' % a.folder)
        return 1
    seg = None
    for folder in albums:
        probe = Album(folder, cache_dir=a.cache, dpi=a.dpi, long_side=a.long_side)
        if probe.pages and seg is None:
            need = probe.page_map(probe.pages[0])[0] is None
            seg = _segmenter(a, need)
            if seg is False:
                return 1
        alb, res = _detect_one(folder, a, seg)
        if alb is None:
            continue
        out = a.out if len(albums) == 1 else os.path.join(a.out, alb.name)
        os.makedirs(out, exist_ok=True)
        bpath = os.path.join(out, 'boxes.json')
        alb.save(res, bpath)
        d = _load(bpath)

        ov = os.path.join(out, 'overlay')
        rs = os.path.join(out, 'restored')
        os.makedirs(ov, exist_ok=True)
        os.makedirs(rs, exist_ok=True)
        cr = os.path.join(out, 'crops') if a.crop else None
        if cr:
            os.makedirs(cr, exist_ok=True)
        for i, page in enumerate(sorted(d['pages']), 1):
            v = d['pages'][page]
            img = imread(os.path.join(folder, page))
            if img is None:
                continue
            sm, _s = downscale(img, d.get('long_side', 1500))
            imwrite(os.path.join(ov, os.path.basename(page)),
                    overlay(sm, v['boxes'], v['scale']))
            done, _used = restore_page(img, v['boxes'], v['scale'],
                                       level=a.level, mono=a.mono)
            imwrite(os.path.join(rs, os.path.basename(page)), done,
                    [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if cr:
                stem = os.path.splitext(os.path.basename(page))[0]
                for j, b in enumerate(v['boxes'], 1):
                    c = crop(done, b, v['scale'])
                    if c is not None:
                        imwrite(os.path.join(cr, '%s_%02d.jpg' % (stem, j)), c,
                                [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            if i % 10 == 0 or i == len(d['pages']):
                print('  書き出し %d/%d' % (i, len(d['pages'])), flush=True)

        rows = _write_report(d, os.path.join(out, 'review.md'), alb.name)
        print('  要確認 %d 頁（全 %d 頁の %.0f%%）→ %s'
              % (len(rows), len(d['pages']),
                 100.0 * len(rows) / max(len(d['pages']), 1),
                 os.path.join(out, 'review.md')))
        print('  枠つき確認画像: %s' % ov)
        print('  補正後: %s' % rs)
        if cr:
            print('  切り出し: %s' % cr)
        print('  枠を直すには: python -m albumscan edit "%s" --boxes "%s"'
              % (folder, bpath))
    return 0


# ------------------------------------------------------------------ edit
def cmd_edit(a):
    from .edit import run_editor
    return run_editor(a.folder, a.boxes, only_review=a.review)


# ------------------------------------------------------------------ 引数
def _common_detect(p):
    p.add_argument('folder')
    p.add_argument('--model', default=None, help='学習済みモデル (.pt)')
    p.add_argument('--cache', default=None,
                   help='輪郭マップの置き場（既定は <folder>/_cache）')
    p.add_argument('--device', default=None, help='cuda / cpu')
    p.add_argument('--dpi', type=float, default=600.0, help='スキャン解像度')
    p.add_argument('--long-side', type=int, default=1500,
                   help='検出に使う縮小画像の長辺')
    p.add_argument('--snap-max-mm', type=float, default=4.0,
                   help='判型に寸法を合わせる上限 mm。0 で「合わせない」')
    p.add_argument('-r', '--recursive', action='store_true',
                   help='直下のフォルダをそれぞれ 1 冊として処理する')
    p.add_argument('-v', '--verbose', action='store_true')


def main(argv=None):
    from . import __version__
    ap = argparse.ArgumentParser(
        prog='albumscan', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version',
                    version='albumscan %s' % __version__)
    sub = ap.add_subparsers(dest='cmd')

    r = sub.add_parser('run', help='1 冊まとめて処理して要確認ページまで出す')
    _common_detect(r)
    r.add_argument('--out', required=True)
    r.add_argument('--level', default='natural',
                   choices=('light', 'natural', 'strong'))
    r.add_argument('--mono', action='store_true', help='白黒プリントのアルバム')
    r.add_argument('--crop', action='store_true', help='1 枚ずつ切り出しも作る')
    r.add_argument('--quality', type=int, default=95)
    r.set_defaults(func=cmd_run)

    d = sub.add_parser('detect', help='枠を検出して boxes.json に書く')
    _common_detect(d)
    d.add_argument('--out', default=None)
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser('edit', help='枠を目で見て直す（tkinter の簡易エディタ）')
    e.add_argument('folder')
    e.add_argument('--boxes', required=True)
    e.add_argument('--review', action='store_true',
                   help='要確認のページだけを順に開く')
    e.set_defaults(func=cmd_edit)

    p = sub.add_parser('report', help='要確認ページの一覧を出す')
    p.add_argument('--boxes', required=True)
    p.add_argument('--out', default=None, help='Markdown で書き出す先')
    p.add_argument('--limit', type=int, default=30)
    p.set_defaults(func=cmd_report)

    o = sub.add_parser('overlay', help='枠を描いた確認画像を書く')
    o.add_argument('folder')
    o.add_argument('--boxes', required=True)
    o.add_argument('--out', required=True)
    o.set_defaults(func=cmd_overlay)

    s = sub.add_parser('restore', help='枠の中の色を戻したページを書く')
    s.add_argument('folder')
    s.add_argument('--boxes', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--level', default='natural',
                   choices=('light', 'natural', 'strong'))
    s.add_argument('--mono', action='store_true', help='白黒プリントのアルバム')
    s.add_argument('--quality', type=int, default=95)
    s.add_argument('-v', '--verbose', action='store_true')
    s.set_defaults(func=cmd_restore)

    c = sub.add_parser('crop', help='boxes.json に従って原寸から切り出す')
    c.add_argument('folder')
    c.add_argument('--boxes', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--quality', type=int, default=95)
    c.set_defaults(func=cmd_crop)

    a = ap.parse_args(argv)
    if not getattr(a, 'func', None):
        ap.print_help()
        return 1
    return a.func(a)


if __name__ == '__main__':
    sys.exit(main())

# -*- coding: utf-8 -*-
"""枠を目で見て直すための簡易エディタ（tkinter だけで動く）。

**完全自動を目指さない**のがこのツールの立場です。実データでは 8% 前後の
ページに人の判断が要りました（写真が台紙と同じ色、傾いて貼られている、
1 枚ごとに紙が違う記念写真帳など）。そこを直す手段が無いと、自動化の
最後の 1 割で行き詰まります。

Pillow を使っていないのは、依存を増やさないため。OpenCV で PNG に符号化して
`tkinter.PhotoImage` に渡せば、それだけで表示できます。

操作:

    ← →          ページを送る
    クリック      枠を選ぶ
    ドラッグ      枠を動かす（角をつかむと大きさを変える）
    空白でドラッグ 枠を新しく作る
    Delete        選んだ枠を消す
    S             選んだ枠をこのアルバムの判型に合わせる
    Ctrl+S        保存（閉じるときにも聞かれる）
"""
import base64
import json
import os

import cv2
import numpy as np

from . import boxes as B
from .imageio import downscale, imread

DISP = 900          # 画面に出すときの長辺
HANDLE = 10         # 角をつかめる範囲（画面上の画素）


def _png_data(img):
    ok, enc = cv2.imencode('.png', img)
    if not ok:
        return None
    return base64.b64encode(enc.tobytes())


class Editor(object):
    def __init__(self, tk, folder, boxes_path, only_review=False):
        self.tk = tk
        self.folder = folder
        self.path = boxes_path
        with open(boxes_path, encoding='utf-8') as f:
            self.doc = json.load(f)
        self.catalogue = [tuple(c) for c in (self.doc.get('catalogue') or [])]
        self.pages = sorted(self.doc['pages'])
        if only_review:
            sel = [p for p in self.pages if self.doc['pages'][p].get('review')]
            if sel:
                self.pages = sel
            else:
                print('要確認のページはありません。全ページを開きます。')
        self.i = 0
        self.dirty = False
        self.sel = None
        self.drag = None
        self.photo = None

        self.root = tk.Tk()
        self.root.title('albumscan — 枠の手直し')
        top = tk.Frame(self.root)
        top.pack(fill='x')
        self.label = tk.Label(top, text='', anchor='w', justify='left')
        self.label.pack(side='left', padx=8, pady=4)
        tk.Button(top, text='◀ 前', command=lambda: self.go(-1)).pack(side='right')
        tk.Button(top, text='次 ▶', command=lambda: self.go(1)).pack(side='right')
        tk.Button(top, text='保存', command=self.save).pack(side='right', padx=6)
        self.canvas = tk.Canvas(self.root, bg='#222', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind('<Button-1>', self.on_down)
        self.canvas.bind('<B1-Motion>', self.on_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_up)
        self.root.bind('<Left>', lambda e: self.go(-1))
        self.root.bind('<Right>', lambda e: self.go(1))
        self.root.bind('<Delete>', lambda e: self.delete())
        self.root.bind('s', lambda e: self.snap())
        self.root.bind('<Control-s>', lambda e: self.save())
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.load()

    # ------------------------------------------------------------ 表示
    @property
    def page(self):
        return self.pages[self.i]

    @property
    def rec(self):
        return self.doc['pages'][self.page]

    def load(self):
        img = imread(os.path.join(self.folder, self.page))
        if img is None:
            self.label.config(text='読めません: %s' % self.page)
            return
        sm, _s = downscale(img, self.doc.get('long_side', 1500))
        self.f = float(DISP) / max(sm.shape[:2])
        disp = cv2.resize(sm, (int(sm.shape[1] * self.f), int(sm.shape[0] * self.f)),
                          interpolation=cv2.INTER_AREA)
        self.photo = self.tk.PhotoImage(data=_png_data(disp))
        self.canvas.config(width=disp.shape[1], height=disp.shape[0])
        self.sel = None
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        if self.photo is not None:
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        s = self.rec['scale']
        for i, b in enumerate(self.rec['boxes']):
            x, y, w, h = [v * self.f for v in b]
            col = '#ff3b3b' if i == self.sel else '#25d425'
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=col, width=2)
            self.canvas.create_text(
                x + 4, y + 10, anchor='w', fill=col,
                text='%.0fx%.0f' % (B.px_to_mm(b[2], s, self.doc.get('dpi', 600)),
                                    B.px_to_mm(b[3], s, self.doc.get('dpi', 600))))
            if i == self.sel:
                self.canvas.create_rectangle(x + w - HANDLE, y + h - HANDLE,
                                             x + w, y + h, outline=col, width=2)
        why = self.rec.get('review') or []
        self.label.config(
            text='%d/%d  %s   枠 %d%s%s'
                 % (self.i + 1, len(self.pages), self.page, len(self.rec['boxes']),
                    '   要確認: ' + ' / '.join(why) if why else '',
                    '   *未保存' if self.dirty else ''))

    # ------------------------------------------------------------ 操作
    def hit(self, x, y):
        """(枠の番号, 'move' or 'resize') を返す。無ければ (None, None)。"""
        for i in range(len(self.rec['boxes']) - 1, -1, -1):
            bx, by, bw, bh = [v * self.f for v in self.rec['boxes'][i]]
            if bx + bw - HANDLE <= x <= bx + bw and by + bh - HANDLE <= y <= by + bh:
                return i, 'resize'
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return i, 'move'
        return None, None

    def on_down(self, e):
        i, mode = self.hit(e.x, e.y)
        self.sel = i
        if i is None:
            self.drag = ('new', e.x, e.y, None)
        else:
            self.drag = (mode, e.x, e.y, list(self.rec['boxes'][i]))
        self.redraw()

    def on_move(self, e):
        if not self.drag:
            return
        mode, x0, y0, orig = self.drag
        dx, dy = (e.x - x0) / self.f, (e.y - y0) / self.f
        if mode == 'new':
            self.canvas.delete('tmp')
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline='#ffd000',
                                         width=2, tags='tmp')
            return
        b = list(orig)
        if mode == 'move':
            b[0] = int(round(orig[0] + dx))
            b[1] = int(round(orig[1] + dy))
        else:
            b[2] = max(8, int(round(orig[2] + dx)))
            b[3] = max(8, int(round(orig[3] + dy)))
        self.rec['boxes'][self.sel] = b
        self.dirty = True
        self.redraw()

    def on_up(self, e):
        if not self.drag:
            return
        mode, x0, y0, _o = self.drag
        self.drag = None
        if mode == 'new':
            self.canvas.delete('tmp')
            x1, y1 = min(x0, e.x), min(y0, e.y)
            w, h = abs(e.x - x0), abs(e.y - y0)
            if w > HANDLE * 2 and h > HANDLE * 2:
                self.rec['boxes'].append(
                    [int(round(x1 / self.f)), int(round(y1 / self.f)),
                     int(round(w / self.f)), int(round(h / self.f))])
                self.sel = len(self.rec['boxes']) - 1
                self.dirty = True
        self.touch()
        self.redraw()

    def delete(self):
        if self.sel is None:
            return
        del self.rec['boxes'][self.sel]
        self.sel = None
        self.dirty = True
        self.touch()
        self.redraw()

    def snap(self):
        """選んだ枠をこのアルバムの判型に合わせる（いちばん近いもの）。"""
        if self.sel is None or not self.catalogue:
            return
        s = self.rec['scale']
        px = []
        for c in self.catalogue:
            a, b = B.mm_to_px(c[0], s), B.mm_to_px(c[1], s)
            px += [(a, b), (b, a)]
        # ここでは距離の上限を掛けない（人が「これに合わせる」と言っているため）
        self.rec['boxes'][self.sel] = B.snap(
            self.rec['boxes'][self.sel], px, s, snap_max_mm=1e9)
        self.dirty = True
        self.touch()
        self.redraw()

    def touch(self):
        """人が触ったページは要確認から外し、印を付ける。"""
        self.rec['review'] = []
        self.rec['edited'] = True
        self.rec['mm'] = [[round(B.px_to_mm(b[2], self.rec['scale']), 1),
                           round(B.px_to_mm(b[3], self.rec['scale']), 1)]
                          for b in self.rec['boxes']]

    def go(self, d):
        self.i = max(0, min(len(self.pages) - 1, self.i + d))
        self.load()

    def save(self, *_a):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.doc, f, ensure_ascii=False, indent=1)
        self.dirty = False
        self.redraw()
        print('保存しました: %s' % self.path)

    def close(self):
        if self.dirty:
            from tkinter import messagebox
            if messagebox.askyesno('albumscan', '保存しますか？'):
                self.save()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def run_editor(folder, boxes_path, only_review=False):
    try:
        import tkinter as tk
    except ImportError:
        print('tkinter が入っていません（Linux では python3-tk を入れてください）。')
        return 1
    if not os.path.exists(boxes_path):
        print('boxes.json がありません: %s' % boxes_path)
        return 1
    ed = Editor(tk, folder, boxes_path, only_review=only_review)
    ed.run()
    print('枠を直したら、その boxes.json でもう一度 restore / crop を走らせてください。')
    return 0

# -*- coding: utf-8 -*-
"""albumscan — 台紙に貼られた写真アルバムのスキャンから、
プリント 1 枚ずつを切り出して退色を補正する。

公開しているのは次の 3 つ:

  albumscan.boxes    内側マップ → 写真の矩形
  albumscan.sizes    そのアルバムの判型を、アルバム自身から推定する
  albumscan.quality  正解データなしで枠の良し悪しを測る
"""
__version__ = '0.1.0'

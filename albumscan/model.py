# -*- coding: utf-8 -*-
"""プリントの輪郭を出す小型 U-Net。

大きなモデルは要らない。判定するのは「ここはプリントの縁か／内側か」という
局所的な性質で、意味理解も文脈も要らない。パラメータを小さく保つほうが、
数百ページしかない学習データでは有利になる（本体 110 万パラメータ・4.4MB）。

入力  : (B, 3, H, W) の RGB（0..1）
出力  : (B, 3, H, W) のロジット
        ch0 縦の境界 / ch1 横の境界 / ch2 プリントの内側

**枠を作るのに使うのは ch2（内側）**。境界の 2 チャンネルは学習を安定させる
補助で、単体の精度は内側ほど高くない（検証 Dice は境界 0.31/0.36 に対し
内側 0.98）。Dice は面と線で意味が違う（面の 0.98 は E判で全周 1.4mm 相当、
線は 2px ずれればほぼ 0）ので、この差だけを見て「境界が弱い」と判断しては
いけないが、**枠は面から作るほうが素直**というのが実測の結論。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=24, out_ch=3):
        super().__init__()
        b = base
        self.e1 = _block(3, b)
        self.e2 = _block(b, b * 2)
        self.e3 = _block(b * 2, b * 4)
        self.e4 = _block(b * 4, b * 8)
        self.d3 = _block(b * 8 + b * 4, b * 4)
        self.d2 = _block(b * 4 + b * 2, b * 2)
        self.d1 = _block(b * 2 + b, b)
        self.out = nn.Conv2d(b, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat(
            [F.interpolate(e4, size=e3.shape[-2:], mode='nearest'), e3], 1))
        d2 = self.d2(torch.cat(
            [F.interpolate(d3, size=e2.shape[-2:], mode='nearest'), e2], 1))
        d1 = self.d1(torch.cat(
            [F.interpolate(d2, size=e1.shape[-2:], mode='nearest'), e1], 1))
        return self.out(d1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())

# -*- coding: utf-8 -*-
"""ページ画像 → 輪郭マップ（縦境界 / 横境界 / 内側）。

推論結果は png（3ch）でキャッシュする。枠の作り直しは何度も行うが、
マップはページごとに 1 回決まれば十分で、そのたびに GPU を起こす必要は無い。
キャッシュを挟むことで、枠を作る側は torch に依存しなくなる。
"""
import os

import cv2
import numpy as np

from .imageio import imread, imwrite


def cache_path(cache_dir, rel):
    b = os.path.join(cache_dir, rel)
    return b[:b.rfind('.')] + '_map.png' if '.' in os.path.basename(rel) \
        else b + '_map.png'


def load_map(cache_dir, rel, shape=None):
    """(縦境界, 横境界, 内側) を float32 0..1 で返す。無ければ None。

    shape を渡すと、大きさの合わないマップは None にする（縮小率を変えたのに
    古いキャッシュが残っている、という取り違えを防ぐ）。
    """
    p = cache_path(cache_dir, rel)
    if not os.path.exists(p):
        return None
    im = imread(p, cv2.IMREAD_COLOR)
    if im is None or im.ndim != 3:
        return None
    if shape is not None and im.shape[:2] != tuple(shape[:2]):
        return None
    f = im.astype(np.float32) / 255.0
    return f[:, :, 0], f[:, :, 1], f[:, :, 2]


def save_map(cache_dir, rel, maps):
    p = cache_path(cache_dir, rel)
    a = np.stack([np.clip(m, 0, 1) for m in maps], axis=2)
    return imwrite(p, (a * 255).astype(np.uint8))


class Segmenter(object):
    """学習済みモデルを 1 度だけ読み、ページを順に推論する。"""

    def __init__(self, model_path, device=None, tile=512, overlap=64):
        import torch
        from .model import UNet
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = UNet(base=ck.get('base', 24)).to(self.device).eval()
        self.net.load_state_dict(ck['state'])
        self.tile, self.overlap = tile, overlap
        self.info = dict(val=ck.get('val'), dice=ck.get('dice'))

    def __call__(self, small):
        """縮小画像 (H, W, 3) → (3, H, W) float32 0..1

        タイルの継ぎ目で値が飛ばないよう、重ねて足してから重みで割る。
        """
        torch = self.torch
        H, W = small.shape[:2]
        x = torch.from_numpy(
            small.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        acc = torch.zeros((1, 3, H, W))
        wsum = torch.zeros((1, 1, H, W))
        step = self.tile - self.overlap
        ys = list(range(0, max(H - self.tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - self.tile, 0) + 1, step)) or [0]
        if ys[-1] + self.tile < H:
            ys.append(H - self.tile)
        if xs[-1] + self.tile < W:
            xs.append(W - self.tile)
        with torch.no_grad():
            for y in ys:
                for x0 in xs:
                    th, tw = min(self.tile, H - y), min(self.tile, W - x0)
                    patch = x[:, :, y:y + th, x0:x0 + tw].to(self.device)
                    # 32 の倍数でないと U-Net の連結でサイズが合わないことがある
                    ph, pw = (32 - th % 32) % 32, (32 - tw % 32) % 32
                    if ph or pw:
                        patch = torch.nn.functional.pad(
                            patch, (0, pw, 0, ph), mode='reflect')
                    with torch.amp.autocast(self.device):
                        p = torch.sigmoid(self.net(patch)).float().cpu()
                    p = p[:, :, :th, :tw]
                    acc[:, :, y:y + th, x0:x0 + tw] += p
                    wsum[:, :, y:y + th, x0:x0 + tw] += 1.0
        return (acc / wsum.clamp(min=1e-6))[0].numpy()

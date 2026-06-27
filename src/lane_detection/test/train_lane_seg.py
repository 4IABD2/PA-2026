"""Phase 2 du fine-tuning : entraine (transfer learning) un modele de
segmentation pre-entraine sur le dataset CARLA produit par generate_seg_dataset.py.

Modele : LRASPP MobileNetV3-Large (leger, pre-entraine), tete remplacee pour
3 classes : 0 = fond, 1 = ligne de voie, 2 = zone roulable.
Sortie : weights/lane_seg.pt (state_dict) -> utilise ensuite en inference.

Lancer :  python train_lane_seg.py --data ../../dataset/lane_seg_dataset --epochs 15
(GPU recommande ; ta RTX 3070 suffit largement.)
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights)

TRAIN_W, TRAIN_H = 512, 384
NUM_CLASSES = 3                      # 0 fond, 1 ligne, 2 roulable
WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), "weights", "lane_seg.pt")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
_LANE_DILATE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))    # epaissit les lignes
_LANE_BRIDGE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))   # relie les pointilles
FILL_DASHES = True            # relier les tirets en lignes CONTINUES dans le label


def _fill_dashes(lane):
    """Relie les tirets d'une meme ligne en une ligne continue (HoughLinesP, qui
    suit la direction des segments). On ne garde que les lignes obliques/verticales
    (les voies) et on exclut l'horizontal (passages pietons)."""
    out = lane.copy()
    lines = cv2.HoughLinesP(lane, 1, np.pi / 180, threshold=20,
                            minLineLength=15, maxLineGap=80)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y2 - y1) > abs(x2 - x1) * 0.6:
                cv2.line(out, (x1, y1), (x2, y2), 255, 3)
    return out


class SegDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.rgb = sorted(glob.glob(os.path.join(root, "rgb", "*.png")))
        if not self.rgb:
            raise RuntimeError(f"aucune image dans {root}/rgb (lance generate_seg_dataset.py)")

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        name = os.path.basename(self.rgb[i])
        img = cv2.cvtColor(cv2.imread(self.rgb[i]), cv2.COLOR_BGR2RGB)
        lane = cv2.imread(os.path.join(self.root, "lane", name), 0)
        drive = cv2.imread(os.path.join(self.root, "drivable", name), 0)

        img = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_LINEAR)
        lane = cv2.resize(lane, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
        drive = cv2.resize(drive, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)

        # Lignes fines/pointillees -> on les relie en lignes CONTINUES puis on
        # epaissit, pour que le modele predise des lignes pleines (comme YOLOPv2).
        if FILL_DASHES:
            lane = _fill_dashes(lane)
        lane = cv2.dilate(lane, _LANE_DILATE)
        lane = cv2.morphologyEx(lane, cv2.MORPH_CLOSE, _LANE_BRIDGE)

        target = np.zeros((TRAIN_H, TRAIN_W), np.int64)
        target[drive > 127] = 2
        target[lane > 127] = 1                    # la ligne est prioritaire sur la route

        x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        x = (x - MEAN) / STD
        return x, torch.from_numpy(target)


def build_model():
    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    low = model.classifier.low_classifier.in_channels
    high = model.classifier.high_classifier.in_channels
    model.classifier.low_classifier = nn.Conv2d(low, NUM_CLASSES, 1)
    model.classifier.high_classifier = nn.Conv2d(high, NUM_CLASSES, 1)
    return model


def main(data, epochs, batch_size, lr, out=WEIGHTS_OUT):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SegDataset(data)
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"[INFO] {len(ds)} images ({len(train_ds)} train / {len(val_ds)} val) sur {dev}")

    model = build_model().to(dev)
    # la ligne de voie est rare -> on la sur-pondere
    weight = torch.tensor([0.4, 6.0, 1.0], device=dev)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    best = 1e9
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)["out"]
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item()
        train_loss = tot / max(1, len(train_dl))

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                vtot += crit(model(x)["out"], y).item()
        val_loss = vtot / max(1, len(val_dl))
        print(f"[EPOCH {ep:02d}/{epochs}] train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out)
            print(f"  -> sauvegarde {out} (meilleur val)")

    print(f"[OK] entraine. Poids : {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../../dataset/lane_seg_dataset")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default=WEIGHTS_OUT)
    args = p.parse_args()
    main(args.data, args.epochs, args.batch_size, args.lr, args.out)

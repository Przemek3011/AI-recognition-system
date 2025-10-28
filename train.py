#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trening małego KWS (keyword spotting) dla fraz:
  - czesc_siri
  - przygotuj_obiad
  - posprzataj_pokoj
  - wynies_smieci
  - zrob_zakupy
  - wyprowadz_psa
  - poucz_sie
(+ _other jako klasa tła)

Użycie:
  python3 train.py train --data_dir data --epochs 25
  python3 train.py infer --model kws_model.pt --wav path/to/test.wav
"""

import os, sys, glob, random, math, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as T

# --------- Stałe audio / cechy ---------
SR = 16_000               # docelowa częstotliwość
WIN = int(0.025 * SR)     # 25 ms
HOP = int(0.010 * SR)     # 10 ms
N_MELS = 64
DUR = 1.0                 # długość okna (sekundy)
SAMPLES = int(DUR * SR)

# --------- Mel-spektrogram ---------
MEL = T.MelSpectrogram(
    sample_rate=SR, n_fft=1024, win_length=WIN, hop_length=HOP,
    n_mels=N_MELS, center=True, power=2.0
)
TO_DB = T.AmplitudeToDB()

def load_wav_1s(path: str) -> torch.Tensor:
    """Wczytuje wav i przycina/pad do 1 sekundy."""
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    wav = wav.squeeze(0)
    if wav.numel() < SAMPLES:
        wav = torch.cat([wav, torch.zeros(SAMPLES - wav.numel())])
    else:
        wav = wav[:SAMPLES]
    return wav

def wav_to_logmel(wav: torch.Tensor) -> torch.Tensor:
    spec = MEL(wav.unsqueeze(0))
    spec_db = TO_DB(spec).squeeze(0)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)
    return spec_db.unsqueeze(0)

# --------- Augmentacje (elastyczne pod wersję audiomentations) ---------
def build_augmentations(disable=False):
    if disable:
        return None
    try:
        from audiomentations import (
            Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, BandPassFilter
        )
    except Exception:
        print("[i] Brak audiomentations – trening bez augmentacji.")
        return None

    # Składamy listę, ale odporni na różne nazwy argumentów
    augs = []

    # Gain: min_gain_db vs min_gain_in_db
    try:
        augs.append(__build_gain(Gain))
    except Exception as e:
        print(f"[!] Gain wyłączony ({e})")

    # Szum
    try:
        augs.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5))
    except Exception as e:
        print(f"[!] AddGaussianNoise wyłączony ({e})")

    # TimeStretch
    try:
        augs.append(TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3))
    except Exception as e:
        print(f"[!] TimeStretch wyłączony ({e})")

    # PitchShift
    try:
        augs.append(PitchShift(min_semitones=-2, max_semitones=2, p=0.3))
    except Exception as e:
        print(f"[!] PitchShift wyłączony ({e})")

    # BandPassFilter: min_center_frequency vs min_center_frequency_hz
    try:
        augs.append(__build_bpf(BandPassFilter))
    except Exception as e:
        print(f"[!] BandPassFilter wyłączony ({e})")

    # Odfiltruj None
    augs = [a for a in augs if a is not None]
    if not augs:
        print("[i] Żadna augmentacja nie jest dostępna – trening bez augmentacji.")
        return None

    try:
        from audiomentations import Compose
        return Compose(augs)
    except Exception:
        return None

def __build_gain(GainCls):
    # najpierw spróbuj nowych nazw
    try:
        return GainCls(min_gain_db=-6, max_gain_db=6, p=0.6)
    except TypeError:
        # starsze nazwy
        return GainCls(min_gain_in_db=-6, max_gain_in_db=6, p=0.6)

def __build_bpf(BPFCls):
    # nowe nazwy
    try:
        return BPFCls(
            min_center_frequency=200.0,
            max_center_frequency=4000.0,
            p=0.2
        )
    except TypeError:
        # alternatywne nazwy z _hz
        return BPFCls(
            min_center_frequency_hz=200.0,
            max_center_frequency_hz=4000.0,
            p=0.2
        )

# --------- Dataset ---------
class KWSDataset(Dataset):
    def __init__(self, root: str, split: str = "train", val_ratio: float = 0.2, seed: int = 42, augment=None):
        super().__init__()
        self.items, self.labels = [], []
        self.train = (split == "train")
        self.augment = augment

        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        assert classes, f"Brak klas w {root}"
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        rng = random.Random(seed)
        for c in classes:
            files = glob.glob(os.path.join(root, c, "*.wav"))
            rng.shuffle(files)
            n = len(files)
            if n == 0:
                print(f"[WARN] Brak plików w klasie: {c}")
                continue
            cut = max(1, int((1 - val_ratio) * n))
            split_files = files[:cut] if self.train else files[cut:]
            for f in split_files:
                self.items.append(f)
                self.labels.append(self.class_to_idx[c])

        # proste wyrównanie liczebności klas (oversampling)
        if self.train and len(self.labels) > 0:
            from collections import Counter
            cnt = Counter(self.labels)
            maxc = max(cnt.values())
            oversampled = []
            for f, y in zip(self.items, self.labels):
                reps = max(1, math.floor(maxc / max(1, cnt[y])))
                oversampled += [(f, y)] * reps
            rng.shuffle(oversampled)
            self.items, self.labels = zip(*oversampled)
            self.items, self.labels = list(self.items), list(self.labels)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path = self.items[i]
        y = self.labels[i]
        wav = load_wav_1s(path)
        if self.train and self.augment is not None:
            try:
                wav = torch.tensor(self.augment(samples=wav.numpy(), sample_rate=SR))
            except Exception:
                # w razie problemów z konkretną augmentacją – użyj surowego
                pass
        feat = wav_to_logmel(wav)
        return feat, y

# --------- Model CNN ---------
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x).squeeze(-1).squeeze(-1)
        return self.head(x)

# --------- Trening ---------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augment = build_augmentations(disable=args.no_aug)
    if augment is None:
        print("[i] Trening bez augmentacji.")
    else:
        print("[i] Augmentacje audio włączone.")

    train_ds = KWSDataset(args.data_dir, "train", val_ratio=args.val_ratio, seed=args.seed, augment=augment)
    val_ds   = KWSDataset(args.data_dir, "val",   val_ratio=args.val_ratio, seed=args.seed, augment=None)

    n_classes = len(train_ds.class_to_idx)
    print(f"[i] Klasy: {list(train_ds.class_to_idx.keys())}")

    model = SmallCNN(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / max(1, len(train_ds))

        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
                tot += yb.numel()
        val_acc = correct / max(1, tot)
        print(f"Ep {ep:02d} | train_loss={train_loss:.3f} | val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "sr": SR, "n_mels": N_MELS, "win": WIN, "hop": HOP
            }, args.out)
            print(f"  ↳ zapisano model: {args.out} (best_acc={best_acc:.3f})")

# --------- Inferencja ---------
def infer(args):
    ckpt = torch.load(args.model, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = SmallCNN(len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wav = load_wav_1s(args.wav)
    feat = wav_to_logmel(wav).unsqueeze(0)
    with torch.no_grad():
        logits = model(feat)
        probs = F.softmax(logits, dim=1).squeeze(0)
    topk = torch.topk(probs, k=min(3, probs.numel()))
    print("Top przewidywania:")
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {idx_to_class[idx]}: {p:.3f}")

# --------- Parser ---------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    p_train = sub.add_parser("train", help="Trenuj model KWS")
    p_train.add_argument("--data_dir", type=str, required=True)
    p_train.add_argument("--out", type=str, default="kws_model.pt")
    p_train.add_argument("--epochs", type=int, default=25)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val_ratio", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no_aug", action="store_true", help="Wyłącz augmentacje audio")
    p_train.set_defaults(func=train)

    p_infer = sub.add_parser("infer", help="Inferencja pojedynczego pliku")
    p_infer.add_argument("--model", type=str, required=True)
    p_infer.add_argument("--wav", type=str, required=True)
    p_infer.set_defaults(func=infer)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

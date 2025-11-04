#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, random, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy.signal import fftconvolve
from tqdm import tqdm

SR = 16000  # docelowy sample rate
METER = pyln.Meter(SR)

def read_audio(path, target_sr=SR):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # przycięcie nadmiarowej ciszy na końcach (delikatnie)
    y, _ = librosa.effects.trim(y, top_db=40)
    return y.astype(np.float32), target_sr

def write_audio(path, y, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")

def loudness_normalize(y, target_lufs=-23.0):
    # zabezpieczenie: zbyt krótkie/zerowe sygnały
    if len(y) < int(0.1 * SR) or np.allclose(y, 0.0):
        return y
    loudness = METER.integrated_loudness(y)
    gain_db = target_lufs - loudness
    gain = 10 ** (gain_db / 20)
    y = y * gain
    # limiter soft: unikaj clipu
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.999:
        y = y / peak * 0.999
    return y

def change_tempo(y, min_rate=0.95, max_rate=1.05):
    rate = random.uniform(min_rate, max_rate)
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr=SR, semitones_min=-1.5, semitones_max=1.5):
    steps = random.uniform(semitones_min, semitones_max)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def gain_db(y, db_min=-3.0, db_max=3.0):
    db = random.uniform(db_min, db_max)
    return y * (10 ** (db / 20.0))

def random_noise_segment(noise_path, length):
    n, sr = sf.read(str(noise_path), always_2d=False)
    if n.ndim > 1:
        n = np.mean(n, axis=1)
    if sr != SR:
        n = librosa.resample(n, orig_sr=sr, target_sr=SR)
    if len(n) < length:
        # pętla jeśli szum krótszy
        reps = int(np.ceil(length / len(n))) + 1
        n = np.tile(n, reps)
    start = random.randint(0, len(n) - length)
    return n[start:start+length].astype(np.float32)

def add_noise_snr(y, noise_paths, snr_min_db=5, snr_max_db=20):
    if not noise_paths:
        return y
    noise_path = random.choice(noise_paths)
    n = random_noise_segment(noise_path, len(y))
    # wylicz współczynnik dla docelowego SNR
    rms_s = np.sqrt(np.mean(y**2)) + 1e-12
    rms_n_desired = rms_s / (10 ** (random.uniform(snr_min_db, snr_max_db)/20))
    rms_n = np.sqrt(np.mean(n**2)) + 1e-12
    n = n * (rms_n_desired / rms_n)
    out = y + n
    # lekkie ograniczenie clipu
    peak = np.max(np.abs(out)) + 1e-12
    if peak > 1.0:
        out = out / peak * 0.999
    return out

def apply_rir(y, rir_paths):
    if not rir_paths:
        return y
    rir_path = random.choice(rir_paths)
    rir, sr = sf.read(str(rir_path), always_2d=False)
    if rir.ndim > 1:
        rir = np.mean(rir, axis=1)
    if sr != SR:
        rir = librosa.resample(rir, orig_sr=sr, target_sr=SR)
    # normalizacja RIR, krótki ogon
    rir = rir / (np.max(np.abs(rir)) + 1e-12)
    if len(rir) > int(0.5 * SR):
        rir = rir[:int(0.5 * SR)]
    conv = fftconvolve(y, rir, mode="full")[:len(y)]
    peak = np.max(np.abs(conv)) + 1e-12
    if peak > 1.0:
        conv = conv / peak * 0.999
    return conv

def codec_roundtrip(y, sr=SR, codec="opus", bitrate="24k"):
    """Przepuszcza sygnał przez kodek (ffmpeg wymagany). Zwraca wejście, jeśli ffmpeg niedostępny."""
    try:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.wav"
            mid = Path(td) / ("mid." + ("ogg" if codec == "opus" else "amr"))
            out = Path(td) / "out.wav"
            sf.write(str(inp), y, sr, subtype="PCM_16")
            if codec == "opus":
                cmd1 = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(inp), "-c:a", "libopus", "-b:a", bitrate, str(mid)]
            elif codec == "amr":
                # amr-nb pracuje przy 8 kHz
                cmd1 = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(inp), "-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", str(mid)]
            else:
                return y
            cmd2 = ["ffmpeg", "-y", "-loglevel", "quiet", "-i", str(mid), "-ar", str(sr), "-ac", "1", str(out)]
            subprocess.check_call(cmd1)
            subprocess.check_call(cmd2)
            y2, _ = sf.read(str(out), always_2d=False)
            if y2.ndim > 1:
                y2 = np.mean(y2, axis=1)
            return y2.astype(np.float32)
    except Exception:
        return y

def augment_one(y, noises, rirs, with_codec=False):
    # 1–2 lekkie modyfikacje + (opcjonalnie) kodek
    if random.random() < 0.7:
        y = gain_db(y, -3, 3)
    if random.random() < 0.5:
        y = change_tempo(y, 0.95, 1.05)
    if random.random() < 0.5:
        y = pitch_shift(y, SR, -1.5, 1.5)
    if random.random() < 0.7 and noises:
        y = add_noise_snr(y, noises, 5, 20)
    if random.random() < 0.4 and rirs:
        y = apply_rir(y, rirs)
    if with_codec and random.random() < 0.5:
        # połowa próbek przejdzie przez opus/amr
        codec = random.choice(["opus", "amr"])
        y = codec_roundtrip(y, SR, codec=codec, bitrate=random.choice(["16k","24k","32k"]))
    y = loudness_normalize(y, -23.0)
    # zachowaj ~150 ms pustki z przodu/tyłu (dodaj lekką ramkę)
    pad = int(0.15 * SR)
    y = np.pad(y, (pad, pad))
    return y

def collect_files(root):
    root = Path(root)
    return [p for p in root.rglob("*.wav")]

def collect_assets(folder):
    if not folder:
        return []
    folder = Path(folder)
    if not folder.exists():
        return []
    files = []
    for ext in ("*.wav", "*.flac", "*.ogg"):
        files += list(folder.rglob(ext))
    return files

def main():
    ap = argparse.ArgumentParser(description="Augmentacja komend głosowych (tempo, pitch, szum, RIR, kodek).")
    ap.add_argument("--input", "-i", type=str, required=True, help="Katalog wejściowy (np. data/)")
    ap.add_argument("--output", "-o", type=str, required=True, help="Katalog wyjściowy (np. data_aug/)")
    ap.add_argument("--noises", type=str, default=None, help="Folder z szumami tła (opcjonalnie)")
    ap.add_argument("--rirs", type=str, default=None, help="Folder z RIR (opcjonalnie)")
    ap.add_argument("--num", type=int, default=5, help="Ile augmentacji na plik")
    ap.add_argument("--keep_clean", action="store_true", help="Zapisz też wersję czystą (z samą normalizacją)")
    ap.add_argument("--codec", action="store_true", help="Włącz przepuszczanie przez kodek (opus/amr)")
    ap.add_argument("--seed", type=int, default=42, help="Seed RNG")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    inp = Path(args.input)
    out = Path(args.output)
    noises = collect_assets(args.noises)
    rirs = collect_assets(args.rirs)

    files = collect_files(inp)
    if not files:
        print("Brak plików WAV w katalogu wejściowym.", file=sys.stderr)
        sys.exit(1)

    print(f"Znaleziono {len(files)} plików.")
    if noises:
        print(f"Szumy: {len(noises)}")
    if rirs:
        print(f"RIR: {len(rirs)}")

    for src in tqdm(files, desc="Augmentacja", unit="file"):
        rel = src.relative_to(inp)
        base_noext = rel.with_suffix("")  # rel path bez .wav
        # wczytaj i przygotuj
        y, _ = read_audio(src, SR)
        y = loudness_normalize(y, -23.0)
        if args.keep_clean:
            dst_clean = out / (str(base_noext) + "__orig.wav")
            write_audio(dst_clean, np.pad(y, (int(0.15*SR), int(0.15*SR))), SR)

        # augmenty
        for k in range(args.num):
            y_aug = augment_one(y, noises, rirs, with_codec=args.codec)
            dst = out / (str(base_noext) + f"__aug{k+1}.wav")
            write_audio(dst, y_aug, SR)

if __name__ == "__main__":
    main()


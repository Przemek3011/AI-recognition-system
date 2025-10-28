#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatycznie etykietuje pliki WAV z folderu out_wav/

Każdy plik ma format:
    1slowo_2slowo__jezyk.wav

Skrypt:
  • tworzy katalog 'data'
  • tworzy podfolder dla każdej etykiety (1slowo_2slowo)
  • kopiuje (lub linkuje) pliki .wav do odpowiednich folderów

Użycie:
    python3 label_by_filename.py
lub
    python3 label_by_filename.py --src out_wav --dst data --move
"""

import os
import re
import shutil
from pathlib import Path
import argparse

# --- Funkcja pomocnicza ---

def parse_label_from_filename(filename: str) -> str:
    """
    Zwraca label z nazwy pliku typu:
        "czesc_siri__pl_PL.wav" -> "czesc_siri"
        "przygotuj_obiad__en_US.wav" -> "przygotuj_obiad"
    """
    base = Path(filename).stem
    m = re.match(r"([a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+_[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9]+)__", base)
    if not m:
        return None
    label = m.group(1).lower()
    # zamień polskie znaki i spacje
    label = label.replace(" ", "_")
    return label

# --- Główna funkcja ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="out_wav", help="Źródłowy folder z plikami WAV")
    parser.add_argument("--dst", type=str, default="data", help="Folder docelowy (dataset)")
    parser.add_argument("--move", action="store_true", help="Przenoś zamiast kopiować")
    parser.add_argument("--symlink", action="store_true", help="Utwórz linki symboliczne zamiast kopiować")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"❌ Folder źródłowy nie istnieje: {src}")
        return

    dst.mkdir(exist_ok=True)
    files = sorted(src.glob("*.wav"))

    if not files:
        print(f"⚠️ Brak plików .wav w {src}")
        return

    print(f"🧠 Znaleziono {len(files)} plików WAV w {src}")
    print(f"📦 Tworzę dataset w {dst}")

    labels = {}

    for f in files:
        label = parse_label_from_filename(f.name)
        if not label:
            print(f"⚠️ Pomijam (niepoprawna nazwa): {f.name}")
            continue

        out_dir = dst / label
        out_dir.mkdir(exist_ok=True)

        target = out_dir / f.name

        if args.symlink:
            try:
                target.symlink_to(f.resolve())
            except FileExistsError:
                pass
        elif args.move:
            shutil.move(str(f), str(target))
        else:
            shutil.copy2(str(f), str(target))

        labels[label] = labels.get(label, 0) + 1

    print("\n✅ Gotowe!")
    for lbl, count in labels.items():
        print(f"  {lbl}: {count} plików")

    print(f"\n📁 Dane znajdują się w: {dst.resolve()}")

if __name__ == "__main__":
    main()


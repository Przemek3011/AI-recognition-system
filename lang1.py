# -*- coding: utf-8 -*-

"""
Użycie:
    python3 piper_all_languages.py TEKST

Opis:
    Skrypt wykrywa wszystkie modele .onnx w ~/piper_voices/
    i generuje pliki WAV z wypowiedzeniem danego tekstu (może być wiele słów)
    w każdym języku, dodając krótką ciszę na początku i końcu.
"""

import sys
import subprocess
from pathlib import Path
import re
import unicodedata
import wave
import os
import tempfile
import shutil

# --- Ustawienia ---
VOICE_DIR = Path.home() / "piper_voices"
OUT_DIR = Path("out_wav")
OUT_DIR.mkdir(exist_ok=True)

HEAD_SILENCE_MS = 50     # cisza na początku
TAIL_SILENCE_MS = 350    # cisza na końcu

# --- Funkcje pomocnicze ---

def sanitize_filename(name: str) -> str:
    """Zachowuje polskie litery i usuwa znaki niebezpieczne."""
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)

def get_all_models():
    """Zwraca listę ścieżek do wszystkich plików .onnx w VOICE_DIR."""
    return sorted(VOICE_DIR.glob("*.onnx"))

def synthesize(text: str, model_path: Path, out_path: Path):
    """Uruchamia Pipera, by zapisać plik WAV."""
    cmd = [
        "piper",
        "--model", str(model_path),
        "--output_file", str(out_path),
    ]
    try:
        # Dodajemy kropkę, by Piper nie ucinał końcówki
        subprocess.run(cmd, input=(text + " .").encode("utf-8"), check=True)
    except subprocess.CalledProcessError as e:
        print(f"✖ Błąd generowania dla {model_path.name}: {e}")

def _append_silence_bytes(n_frames: int, nchannels: int, sampwidth: int) -> bytes:
    """Zwraca bajty ciszy (zero) dla n_frames."""
    return b"\x00" * (n_frames * nchannels * sampwidth)

def pad_wav_head_tail(path: Path, head_ms=HEAD_SILENCE_MS, tail_ms=TAIL_SILENCE_MS):
    """Dodaje ciszę na początku i końcu pliku WAV."""
    if not path.exists() or path.stat().st_size == 0:
        return

    with wave.open(str(path), "rb") as rf:
        nchannels = rf.getnchannels()
        sampwidth = rf.getsampwidth()
        framerate = rf.getframerate()
        nframes = rf.getnframes()
        audio = rf.readframes(nframes)

    head_frames = int(framerate * (head_ms / 1000.0))
    tail_frames = int(framerate * (tail_ms / 1000.0))
    head_sil = _append_silence_bytes(head_frames, nchannels, sampwidth)
    tail_sil = _append_silence_bytes(tail_frames, nchannels, sampwidth)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(head_sil)
            wf.writeframes(audio)
            wf.writeframes(tail_sil)
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Główna logika ---

def main():
    if len(sys.argv) < 2:
        print("Użycie: python3 piper_all_languages.py TEKST")
        sys.exit(1)

    # Łączymy wszystkie argumenty w jeden tekst, zachowując spacje
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        print("Podaj niepusty tekst.")
        sys.exit(1)

    models = get_all_models()
    if not models:
        print(f"❌ Nie znaleziono modeli w {VOICE_DIR}")
        sys.exit(2)

    print(f"🧠 Znaleziono {len(models)} modeli w {VOICE_DIR}")
    print(f"🔊 Generuję wypowiedź: “{text}”")

    base = sanitize_filename(text)

    for model_path in models:
        lang_name = sanitize_filename(model_path.stem)
        out_file = OUT_DIR / f"{base}__{lang_name}.wav"
        synthesize(text, model_path, out_file)
        if out_file.exists() and out_file.stat().st_size > 0:
            pad_wav_head_tail(out_file)
            print(f"✔ {lang_name} -> {out_file}")
        else:
            print(f"⚠ Nie udało się zapisać: {lang_name}")

    print(f"\n✅ Gotowe! Pliki WAV w folderze: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()

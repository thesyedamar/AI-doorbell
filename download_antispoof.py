"""
download_antispoof.py — Download Pre-trained Anti-Spoof Models
==============================================================
Downloads Silent-Face Anti-Spoofing PyTorch models (.pth files)
from the official minivision-ai GitHub repository.

Run this ONCE:
    python download_antispoof.py

Then run main.py — anti-spoofing will work immediately.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg

MODELS = [
    {
        "save_as": "2.7_80x80_MiniFASNetV2.pth",
        "url":     "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
    },
    {
        "save_as": "4_0_0_80x80_MiniFASNetV1SE.pth",
        "url":     "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
    },
]

# Also need the model source code
MODEL_SRC = [
    {
        "save_as": "MiniFASNet.py",
        "url":     "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/src/model_lib/MiniFASNet.py",
    },
    {
        "save_as": "utility.py",
        "url":     "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/src/utility.py",
    },
]


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "=" * (pct // 2) + "-" * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


def download(url: str, save_path: Path) -> bool:
    if save_path.exists():
        print(f"  Already exists: {save_path.name} — skipping.")
        return True
    print(f"\nDownloading: {save_path.name}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunk_size = 8192
            downloaded = 0
            block_num  = 0
            with open(save_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    block_num  += 1
                    show_progress(block_num, chunk_size, total)
        print()
        size_kb = save_path.stat().st_size // 1024
        print(f"  Saved: {save_path} ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def main():
    print("\n" + "=" * 55)
    print("  AiDarwaza — Anti-Spoof Model Downloader")
    print("=" * 55)

    models_dir  = cfg.paths.models_dir
    antispoof_dir = models_dir / "silent_face"
    antispoof_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving models to: {antispoof_dir}")

    src_dir = cfg.paths.root / "vision" / "silent_face_src"
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving source to: {src_dir}\n")

    success = 0
    total   = len(MODELS) + len(MODEL_SRC)

    for model in MODELS:
        if download(model["url"], antispoof_dir / model["save_as"]):
            success += 1

    for src in MODEL_SRC:
        if download(src["url"], src_dir / src["save_as"]):
            success += 1

    print()
    if success == total:
        print("=" * 55)
        print("  All files downloaded!")
        print("  Run python main.py — anti-spoofing is now active.")
        print("=" * 55)
    else:
        print("=" * 55)
        print(f"  WARNING: {success}/{total} files downloaded.")
        print("  Check your internet and try again.")
        print("=" * 55)
        sys.exit(1)


if __name__ == "__main__":
    main()

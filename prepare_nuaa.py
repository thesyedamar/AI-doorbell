"""
prepare_nuaa.py — Prepare NUAA Dataset for Training
=====================================================
Organizes the downloaded NUAA dataset into the correct
folder structure for train_antispoof.py.

NUAA structure after download:
    ClientRaw/      ← real face images
    ImposterRaw/    ← spoof face images (printed photos)

Output structure:
    training/
        train/
            real/   ← 80% of ClientRaw
            spoof/  ← 80% of ImposterRaw
        test/
            real/   ← 20% of ClientRaw
            spoof/  ← 20% of ImposterRaw

Usage:
    python prepare_nuaa.py --nuaa path/to/nuaa/folder
    python prepare_nuaa.py --nuaa C:/Downloads/nuaaaa
"""

import sys
import shutil
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg


def prepare_nuaa(nuaa_root: Path, train_split: float = 0.8) -> None:
    print("\n" + "=" * 55)
    print("  AiDarwaza — NUAA Dataset Preparation")
    print("=" * 55)
    print(f"  Source : {nuaa_root}")
    print(f"  Output : {cfg.paths.root / 'training'}")
    print(f"  Split  : {int(train_split*100)}% train / {int((1-train_split)*100)}% test")
    print()

    # Find real and spoof source folders
    client_dir   = None
    imposter_dir = None

    # Search for ClientRaw and ImposterRaw
    for folder in nuaa_root.rglob("*"):
        if folder.is_dir():
            name = folder.name.lower()
            if "clientraw" in name or "client_raw" in name:
                client_dir = folder
            if "imposterraw" in name or "imposter_raw" in name:
                imposter_dir = folder

    if client_dir is None or imposter_dir is None:
        print("ERROR: Could not find ClientRaw and ImposterRaw folders.")
        print(f"Searched in: {nuaa_root}")
        print("\nPlease check the folder structure and try again.")
        print("Expected folders: ClientRaw/ and ImposterRaw/")
        sys.exit(1)

    print(f"  Real (ClientRaw)  : {client_dir}")
    print(f"  Spoof (ImposterRaw): {imposter_dir}")

    # Collect all images
    real_images  = sorted(list(client_dir.rglob("*.jpg")) +
                          list(client_dir.rglob("*.png")) +
                          list(client_dir.rglob("*.JPG")))
    spoof_images = sorted(list(imposter_dir.rglob("*.jpg")) +
                          list(imposter_dir.rglob("*.png")) +
                          list(imposter_dir.rglob("*.JPG")))

    if not real_images:
        print(f"ERROR: No images found in {client_dir}")
        sys.exit(1)
    if not spoof_images:
        print(f"ERROR: No images found in {imposter_dir}")
        sys.exit(1)

    print(f"\n  Real images  : {len(real_images):,}")
    print(f"  Spoof images : {len(spoof_images):,}")
    print(f"  Total        : {len(real_images) + len(spoof_images):,}")

    # Shuffle for random split
    random.seed(42)
    random.shuffle(real_images)
    random.shuffle(spoof_images)

    # Split
    real_train  = real_images[:int(len(real_images) * train_split)]
    real_test   = real_images[int(len(real_images) * train_split):]
    spoof_train = spoof_images[:int(len(spoof_images) * train_split)]
    spoof_test  = spoof_images[int(len(spoof_images) * train_split):]

    print(f"\n  Train real   : {len(real_train):,}")
    print(f"  Train spoof  : {len(spoof_train):,}")
    print(f"  Test real    : {len(real_test):,}")
    print(f"  Test spoof   : {len(spoof_test):,}")

    # Create output directories
    training_root = cfg.paths.root / "training"
    dirs = {
        "train_real":  training_root / "train" / "real",
        "train_spoof": training_root / "train" / "spoof",
        "test_real":   training_root / "test"  / "real",
        "test_spoof":  training_root / "test"  / "spoof",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(files: list, dest: Path, label: str) -> None:
        print(f"\n  Copying {label}...")
        for i, src in enumerate(files):
            dst = dest / f"{i:05d}_{src.name}"
            shutil.copy2(str(src), str(dst))
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(files)}")
        print(f"    Done — {len(files)} files copied.")

    copy_files(real_train,  dirs["train_real"],  "train/real")
    copy_files(spoof_train, dirs["train_spoof"], "train/spoof")
    copy_files(real_test,   dirs["test_real"],   "test/real")
    copy_files(spoof_test,  dirs["test_spoof"],  "test/spoof")

    print("\n" + "=" * 55)
    print("  Dataset prepared successfully!")
    print(f"  Location: {training_root}")
    print()
    print("  Now run:")
    print("  python training/train_antispoof.py")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare NUAA dataset for AiDarwaza training"
    )
    parser.add_argument(
        "--nuaa",
        type=str,
        required=True,
        help="Path to downloaded NUAA dataset folder",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)",
    )
    args = parser.parse_args()

    nuaa_root = Path(args.nuaa)
    if not nuaa_root.exists():
        print(f"ERROR: Path not found: {nuaa_root}")
        sys.exit(1)

    prepare_nuaa(nuaa_root, args.split)


if __name__ == "__main__":
    main()

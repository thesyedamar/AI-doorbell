"""
training/train_antispoof.py — Anti-Spoofing Model Trainer (NUAA)
=================================================================
Trains MobileNetV2 binary classifier on NUAA dataset.

NUAA dataset:
    ~3,500 real face images  (ClientRaw)
    ~5,105 spoof images      (printed photo attacks)

After running prepare_nuaa.py your structure will be:
    training/train/real/   ← real faces
    training/train/spoof/  ← spoof faces
    training/test/real/
    training/test/spoof/

Output:
    models/antispoofing_model.json
    models/antispoofing_model.h5
    models/training_report.txt
    models/training_curves.png

Usage:
    python training/train_antispoof.py
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_antispoof")

# ---------------------------------------------------------------------------
# Settings — tuned for NUAA (~8,600 images total)
# ---------------------------------------------------------------------------
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS_HEAD   = 10
EPOCHS_FINE   = 15
LR_HEAD       = 1e-3
LR_FINE       = 1e-5
UNFREEZE_FROM = 100


def count_images(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*")
               if f.suffix.lower() in (".jpg", ".jpeg", ".png"))


def build_dataset(data_dir: Path, subset: str, augment: bool = False):
    import tensorflow as tf

    subset_dir = data_dir / subset
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"Not found: {subset_dir}\n"
            f"Run: python prepare_nuaa.py --nuaa /path/to/nuaa"
        )

    real_count  = count_images(subset_dir / "real")
    spoof_count = count_images(subset_dir / "spoof")
    logger.info(
        f"{subset.upper()} | real={real_count:,} "
        f"spoof={spoof_count:,} total={real_count+spoof_count:,}"
    )

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(subset_dir),
        labels="inferred",
        label_mode="binary",
        class_names=["real", "spoof"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(subset == "train"),
        seed=42,
    )

    normalize = tf.keras.layers.Rescaling(1.0 / 255)
    dataset   = dataset.map(
        lambda x, y: (normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomBrightness(0.25),
            tf.keras.layers.RandomContrast(0.25),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ])
        dataset = dataset.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE), real_count, spoof_count


def build_model():
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x      = base(inputs, training=False)
    x      = tf.keras.layers.GlobalAveragePooling2D()(x)
    x      = tf.keras.layers.BatchNormalization()(x)
    x      = tf.keras.layers.Dense(256, activation="relu")(x)
    x      = tf.keras.layers.Dropout(0.4)(x)
    x      = tf.keras.layers.Dense(64,  activation="relu")(x)
    x      = tf.keras.layers.Dropout(0.3)(x)
    outputs= tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="antispoof_mobilenetv2")
    logger.info(f"Model built | params={model.count_params():,}")
    return model


def unfreeze_model(model):
    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[:UNFREEZE_FROM]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    logger.info(f"Unfrozen {trainable} layers from index {UNFREEZE_FROM}")
    return model


def build_callbacks(phase: str):
    import tensorflow as tf
    ckpt = cfg.paths.models_dir / f"antispoof_best_{phase}.h5"
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3,
            patience=3, min_lr=1e-8, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]


def evaluate_and_save(model, test_ds, history1, history2) -> None:
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    logger.info("\nRunning final evaluation...")

    y_true, y_pred_prob = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_prob.extend(preds.flatten())

    y_true      = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred      = (y_pred_prob >= 0.5).astype(int)

    auc    = roc_auc_score(y_true, y_pred_prob)
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=["Real", "Spoof"], digits=4)

    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    logger.info(f"\nAUC  : {auc:.4f}")
    logger.info(f"FAR  : {far:.4f}  (spoofs wrongly accepted as real)")
    logger.info(f"FRR  : {frr:.4f}  (real faces wrongly rejected)")
    logger.info(f"\nConfusion Matrix:\n"
                f"           Real  Spoof\n"
                f"Real       {tn:5d}  {fp:5d}\n"
                f"Spoof      {fn:5d}  {tp:5d}")
    logger.info(f"\n{report}")

    # Save report
    report_path = cfg.paths.models_dir / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"AiDarwaza Anti-Spoof Training Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: NUAA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"AUC  : {auc:.4f}\n")
        f.write(f"FAR  : {far:.4f}\nFRR  : {frr:.4f}\n\n")
        f.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\n\n")
        f.write(report)
    logger.info(f"Report saved: {report_path}")

    # Save curves
    try:
        import matplotlib.pyplot as plt
        acc  = history1.history["accuracy"]  + history2.history["accuracy"]
        val  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
        loss = history1.history["loss"] + history2.history["loss"]
        vloss= history1.history["val_loss"] + history2.history["val_loss"]
        pb   = len(history1.history["accuracy"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(acc, label="Train", color="royalblue")
        ax1.plot(val, label="Val",   color="darkorange")
        ax1.axvline(pb, color="gray", linestyle="--", label="Fine-tune start")
        ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(loss,  label="Train", color="royalblue")
        ax2.plot(vloss, label="Val",   color="darkorange")
        ax2.axvline(pb, color="gray", linestyle="--")
        ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.suptitle("AiDarwaza Anti-Spoofing — Training Curves (NUAA)")
        plt.tight_layout()
        curves_path = cfg.paths.models_dir / "training_curves.png"
        plt.savefig(str(curves_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Curves saved: {curves_path}")
    except ImportError:
        logger.warning("matplotlib not installed — skipping curves plot.")


def train(data_dir: Path, epochs_head: int, epochs_fine: int) -> None:
    import tensorflow as tf

    logger.info("=" * 60)
    logger.info("  AiDarwaza — Anti-Spoofing Training (NUAA Dataset)")
    logger.info("=" * 60)

    train_ds, train_real, train_spoof = build_dataset(data_dir, "train", augment=True)
    test_ds,  test_real,  test_spoof  = build_dataset(data_dir, "test",  augment=False)

    total       = train_real + train_spoof
    class_weight= {
        0: total / (2 * train_real)  if train_real  > 0 else 1.0,
        1: total / (2 * train_spoof) if train_spoof > 0 else 1.0,
    }
    logger.info(f"Class weights: real={class_weight[0]:.3f} spoof={class_weight[1]:.3f}")

    # Phase 1 — Head training
    logger.info(f"\nPhase 1 — Head training (max {epochs_head} epochs)")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    history1 = model.fit(
        train_ds, validation_data=test_ds,
        epochs=epochs_head, class_weight=class_weight,
        callbacks=build_callbacks("head"), verbose=1,
    )
    logger.info(f"Phase 1 done | best_val_acc={max(history1.history['val_accuracy']):.4f}")

    # Phase 2 — Fine-tuning
    logger.info(f"\nPhase 2 — Fine-tuning (max {epochs_fine} epochs)")
    model = unfreeze_model(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    history2 = model.fit(
        train_ds, validation_data=test_ds,
        epochs=epochs_fine, class_weight=class_weight,
        callbacks=build_callbacks("fine"), verbose=1,
    )
    logger.info(f"Phase 2 done | best_val_acc={max(history2.history['val_accuracy']):.4f}")

    # Evaluate and save report
    evaluate_and_save(model, test_ds, history1, history2)

    # Save model files
    json_path    = cfg.paths.antispoof_json
    weights_path = cfg.paths.antispoof_weights

    with open(json_path, "w") as f:
        f.write(model.to_json())
    model.save_weights(str(weights_path))

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Model → {json_path.name} + {weights_path.name}")
    logger.info("  Run python main.py — anti-spoofing is now active!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train AiDarwaza anti-spoofing on NUAA dataset"
    )
    parser.add_argument("--data",        type=str, default="training")
    parser.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    parser.add_argument("--epochs-fine", type=int, default=EPOCHS_FINE)
    args = parser.parse_args()

    data_dir = cfg.paths.root / args.data
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run: python prepare_nuaa.py --nuaa /path/to/nuaa")
        sys.exit(1)

    train(data_dir, args.epochs_head, args.epochs_fine)


if __name__ == "__main__":
    main()

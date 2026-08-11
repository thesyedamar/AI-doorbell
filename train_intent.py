"""
train_intent.py — Intent Classifier Trainer
=============================================
Trains a TF-IDF + SVM pipeline to classify visitor
intent from Urdu speech transcripts.

Intents:
    greeting    — visitor saying hello / salaam
    delivery    — parcel / courier / package
    guest       — wants to meet someone inside
    emergency   — urgent help needed
    unknown     — unclear intent

Output:
    nlp/models/intent_model.pkl

Usage:
    python train_intent.py
    python train_intent.py --eval   (show detailed evaluation)
"""

import sys
import argparse
import logging
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_intent")


# ---------------------------------------------------------------------------
# Training data — Urdu sentences per intent
# ---------------------------------------------------------------------------
# Add more examples to improve accuracy.
# Each sentence should be a realistic Urdu doorbell utterance.

TRAINING_DATA = {

    "greeting": [
        "السلام علیکم",
        "وعلیکم السلام",
        "آداب",
        "ہیلو",
        "جی السلام علیکم",
        "سلام بھائی",
        "سلام جی",
        "آداب عرض ہے",
        "سلام علیکم جی",
        "ہاں جی سلام",
        "نمستے",
        "ہیلو جی کیسے ہیں",
        "السلام علیکم کوئی ہے",
        "جی بھائی سلام",
        "ارے سلام بھائی",
        "السلام علیکم آپ کیسے ہیں",
        "سلام دوست",
        "جی آداب",
        "خیر سلامت",
        "سلام جی میں آیا ہوں",
    ],

    "delivery": [
        "پارسل ہے",
        "پارسل لایا ہوں",
        "کوریئر ہے",
        "ڈیلیوری ہے",
        "آپ کا پارسل",
        "پیکیج لایا ہوں",
        "آرڈر آیا ہے",
        "سامان لایا ہوں",
        "پارسل دینا ہے",
        "کوریئر والا ہوں",
        "ڈیلیوری کرنی ہے",
        "آن لائن آرڈر ہے",
        "پوسٹ آفس سے آیا ہوں",
        "خط ہے دستخط چاہیے",
        "لفافہ ہے",
        "سامان آیا ہے",
        "شاپنگ کا پارسل ہے",
        "دراز کا آرڈر ہے",
        "ایمیزون کا پارسل",
        "کچھ چیزیں لایا ہوں آپ کے لیے",
        "ڈبہ لایا ہوں",
        "گھر کا سامان ہے",
        "آپ نے آرڈر کیا تھا",
        "پارسل پہنچانا ہے",
        "کوریئر کمپنی سے ہوں",
    ],

    "guest": [
        "ملنا ہے",
        "صاحب سے ملنا ہے",
        "گھر میں کوئی ہے",
        "احمد بھائی ہیں",
        "میں دوست ہوں",
        "رشتہ دار ہوں",
        "گھر والوں سے ملنا ہے",
        "انکل ہیں گھر میں",
        "آنٹی کو بلائیں",
        "بھائی سے ملنا ہے",
        "کیا صاحب گھر میں ہیں",
        "جی ملنے آیا ہوں",
        "میں پڑوسی ہوں",
        "یہاں کام ہے",
        "دفتر کا کام ہے",
        "میں علی کا دوست ہوں",
        "کیا آپ کے گھر میں احمد ہیں",
        "بیگم صاحبہ سے ملنا ہے",
        "کچھ کام ہے آپ سے",
        "میں ان کا ساتھی ہوں",
        "بچوں کا استاد ہوں",
        "باہر سے آیا ہوں ملنے",
        "رشتے کی بات کرنی ہے",
        "کیا وہ گھر پر ہیں",
        "ذرا بلائیں انہیں",
    ],

    "emergency": [
        "مدد کریں",
        "ضروری کام ہے",
        "جلدی آئیں",
        "ایمرجنسی ہے",
        "حادثہ ہو گیا",
        "جلدی کریں",
        "خطرہ ہے",
        "آگ لگ گئی",
        "چور آیا ہے",
        "کوئی بیمار ہے",
        "ہسپتال جانا ہے",
        "ابھی مدد چاہیے",
        "بچاؤ بچاؤ",
        "پانی آ گیا ہے",
        "گھر میں کوئی گر گیا",
        "ڈاکٹر بلاؤ",
        "ایمبولینس چاہیے",
        "بہت تکلیف ہے",
        "فوری مدد کریں",
        "کچھ ہو گیا ہے",
        "گھبراہٹ ہو رہی ہے",
        "دیوار گر گئی",
        "گیس لیک ہو رہی ہے",
        "پولیس بلاؤ",
        "بچہ گم ہو گیا",
    ],

    "unknown": [
        "کچھ نہیں",
        "بس ایسے ہی",
        "پتہ نہیں",
        "ہاں",
        "نہیں",
        "ٹھیک ہے",
        "جی",
        "اچھا",
        "ہمم",
        "وہ",
        "کیا",
        "ذرا",
        "بس",
        "آؤں",
        "دیکھیں",
        "کچھ بات کرنی تھی",
        "وہ میں",
        "کیا آپ",
        "ارے",
        "اوہ",
    ],
}


# ---------------------------------------------------------------------------
# Build sklearn pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    """
    TF-IDF vectorizer + Linear SVM classifier.

    TF-IDF works well for Urdu because:
    - Captures word importance across documents
    - Character n-grams handle Urdu morphology variations
    - Works on Unicode text without tokenization changes

    LinearSVC is fast, accurate for text classification,
    and works well with small-to-medium datasets.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",       # Character n-grams — great for Urdu
            ngram_range=(2, 4),       # bi to 4-grams
            min_df=1,
            max_features=10000,
            sublinear_tf=True,        # Log normalization
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
            ),
            cv=3,
        )),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def prepare_data(extra_csv: str = None):
    """
    Prepare training data from built-in sentences
    and optionally from a CSV file.

    CSV format (no header):
        intent,urdu_sentence

    Returns:
        (X: list of texts, y: list of labels)
    """
    X, y = [], []

    # Built-in training data
    for intent, sentences in TRAINING_DATA.items():
        for sentence in sentences:
            X.append(sentence.strip())
            y.append(intent)

    # Optional CSV file
    if extra_csv:
        csv_path = Path(extra_csv)
        if csv_path.exists():
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        intent_label = row[0].strip()
                        sentence     = row[1].strip()
                        if intent_label and sentence:
                            X.append(sentence)
                            y.append(intent_label)
            logger.info(f"Loaded extra data from: {csv_path}")
        else:
            logger.warning(f"CSV file not found: {csv_path}")

    logger.info(f"Total training samples: {len(X)}")
    for intent in TRAINING_DATA.keys():
        count = y.count(intent)
        logger.info(f"  {intent:<12}: {count} samples")

    return X, y


def train(extra_csv: str = None, run_eval: bool = True) -> None:
    """Full training pipeline."""
    from sklearn.model_selection import cross_val_score

    logger.info("=" * 55)
    logger.info("  AiDarwaza — Intent Classifier Training")
    logger.info("=" * 55)

    # Prepare data
    X, y = prepare_data(extra_csv)

    # Build pipeline
    pipeline = build_pipeline()

    # Cross-validation (gives honest accuracy estimate)
    logger.info("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    logger.info(
        f"CV Accuracy: {cv_scores.mean():.4f} "
        f"(+/- {cv_scores.std():.4f})"
    )

    # Train on full dataset
    logger.info("\nTraining on full dataset...")
    pipeline.fit(X, y)
    logger.info("Training complete.")

    # Full evaluation
    if run_eval:
        evaluate(pipeline, X, y)

    # Save model
    model_path = cfg.paths.intent_model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(model_path))
    logger.info(f"\nModel saved: {model_path}")

    logger.info("\n" + "=" * 55)
    logger.info("  Intent model ready!")
    logger.info("  Run python main.py — intent detection now active.")
    logger.info("=" * 55)


def evaluate(pipeline, X: list, y: list) -> None:
    """Print detailed classification report."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = pipeline.predict(X)

    logger.info("\nClassification Report (training set):")
    report = classification_report(y, y_pred, digits=4)
    logger.info(f"\n{report}")

    cm = confusion_matrix(y, y_pred, labels=list(TRAINING_DATA.keys()))
    logger.info("Confusion Matrix:")
    labels = list(TRAINING_DATA.keys())
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
    logger.info(header)
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>12}" + "".join(f"{v:>12}" for v in row)
        logger.info(row_str)


def test_predictions(pipeline) -> None:
    """Quick sanity check with sample sentences."""
    test_cases = [
        ("السلام علیکم",           "greeting"),
        ("پارسل لایا ہوں",         "delivery"),
        ("صاحب سے ملنا ہے",        "guest"),
        ("مدد کریں ایمرجنسی ہے",  "emergency"),
        ("وہ بس",                  "unknown"),
    ]

    logger.info("\nQuick prediction test:")
    logger.info(f"{'Sentence':<35} {'Expected':<12} {'Predicted':<12} {'Match'}")
    logger.info("-" * 70)

    for sentence, expected in test_cases:
        predicted = pipeline.predict([sentence])[0]
        probs     = pipeline.predict_proba([sentence])[0]
        conf      = max(probs)
        match     = "✅" if predicted == expected else "❌"
        logger.info(
            f"{sentence:<35} {expected:<12} "
            f"{predicted:<12} {match} ({conf:.2f})"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train AiDarwaza intent classifier"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file with extra training data (intent,sentence)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Show detailed evaluation after training",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick prediction test after training",
    )
    args = parser.parse_args()

    train(extra_csv=args.csv, run_eval=True)

    if args.test:
        import joblib
        pipeline = joblib.load(str(cfg.paths.intent_model_path))
        test_predictions(pipeline)


if __name__ == "__main__":
    main()

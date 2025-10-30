import os
from joblib import load
import re

bundle_path = os.path.join(os.getcwd(), "best_seller_bundle.joblib")
bundle = load(bundle_path)
model = bundle["model"]

MENTION_HASHTAG_RE = re.compile(r"(@|#)([A-Za-z0-9]+)")
EMAIL_RE = re.compile(
    r"([A-Za-z0-9]+[._-])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Za-z]{2,})+"
)
URL_RE = re.compile(r"https?\S+", re.MULTILINE)
DIGIT_RE = re.compile(r"\d")
PUNCT_RE = re.compile(r"[^\w\s\']")
SPACE_RE = re.compile(r"\s+")


def clean_sentence(sent: str, lower: bool = True) -> str:
    if lower:
        sent = sent.lower()
    sent = MENTION_HASHTAG_RE.sub(" ", sent)
    sent = EMAIL_RE.sub(" ", sent)
    sent = URL_RE.sub(" ", sent)
    sent = DIGIT_RE.sub(" ", sent)
    sent = PUNCT_RE.sub(" ", sent)
    sent = SPACE_RE.sub(" ", sent).strip()
    return sent


def predict_best_seller(desc, pipeline):
    desc = clean_sentence(desc)
    prediction = pipeline.predict([desc])
    return {
        "prediction": int(prediction[0]),
        "best_seller": "Yes" if prediction == 1 else "No",
    }


def run_bot():
    while True:
        value = input("Enter the book description or (q) to quit: ")
        if value.strip().lower() == "q":
            break
        preds = predict_best_seller(value, model)

        print("=" * 50)
        print("📑 BOOK DESCRIPTION")
        print("=" * 50)
        print(value)
        print()
        print("=" * 50)
        print("🔮 BEST SELLER PREDICTION")
        print("=" * 50)
        print(f" PREDICTION              : {preds['prediction']}")
        print(f" BEST SELLER OUTCOME     : {preds['best_seller']}")
        print()


if __name__ == "__main__":
    run_bot()

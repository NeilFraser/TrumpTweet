"""
06_compare_results.py
---------------------
Loads all trained models, evaluates them on the SAME test set, and
plots a comparison. Use this for the presentation.


DEPENDENCIES:
    pip install matplotlib scikit-learn pandas joblib tensorflow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_data


def evaluate_logreg():
    import joblib
    if not os.path.exists("model_01_logreg.joblib"):
        return None
    vectorizer, model = joblib.load("model_01_logreg.joblib")
    _, X_test, _, y_test = get_data()
    y_pred = model.predict(vectorizer.transform(X_test))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)


def evaluate_keras(path: str):
    if not os.path.exists(path):
        return None
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    X_train, X_test, y_train, y_test = get_data()
 
    tokenizer = Tokenizer(num_words=20_000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=60, padding="post"
    )
    model = tf.keras.models.load_model(path, compile=False)
    y_prob = model.predict(X_test_pad, batch_size=128, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)


def evaluate_distilbert():
    path = "model_05_distilbert"
    if not os.path.isdir(path):
        return None
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _, X_test, _, y_test = get_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = X_test[i : i + 64]
            enc = tok(batch, truncation=True, padding=True,
                      max_length=64, return_tensors="pt").to(device)
            out = model(**enc).logits.argmax(dim=-1).cpu().numpy()
            preds.extend(out)
    return accuracy_score(y_test, preds), f1_score(y_test, preds)


def main():
    results = {}

    print("Evaluating LogReg...")
    r = evaluate_logreg();      results["LogReg (TF-IDF)"] = r
    print("Evaluating MLP...")
    r = evaluate_keras("model_02_mlp.keras"); results["Embedding+MLP"] = r
    print("Evaluating LSTM...")
    r = evaluate_keras("model_03_lstm.keras"); results["BiLSTM"] = r
    print("Evaluating Transformer (scratch)...")
    r = evaluate_keras("model_04_transformer.keras"); results["Transformer (scratch)"] = r
    print("Evaluating DistilBERT...")
    r = evaluate_distilbert();  results["DistilBERT (fine-tuned)"] = r

    # Print + plot
    print("\n=== Model comparison ===")
    print(f"{'Model':<28s} {'Accuracy':>10s} {'F1':>10s}")
    print("-" * 50)
    for name, r in results.items():
        if r is None:
            print(f"{name:<28s} {'(not trained)':>22s}")
        else:
            acc, f1 = r
            print(f"{name:<28s} {acc:>10.4f} {f1:>10.4f}")

    # Bar chart
    trained = {k: v for k, v in results.items() if v is not None}
    if trained:
        names = list(trained.keys())
        accs = [v[0] for v in trained.values()]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, accs)
        plt.ylim(0.5, 1.0)
        plt.ylabel("Test accuracy")
        plt.title("Real-vs-fake tweet classifier comparison")
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                     f"{acc:.3f}", ha="center", fontsize=9)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig("comparison_accuracy.png", dpi=150)
        print("\nSaved chart: comparison_accuracy.png")


if __name__ == "__main__":
    main()

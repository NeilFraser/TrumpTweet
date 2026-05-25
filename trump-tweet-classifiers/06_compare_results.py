"""
06_compare_results.py — PyTorch version
---------------------------------------
Loads all trained classifiers and evaluates them on the SAME test set,
then prints a table and saves two plots:
   - comparison_accuracy.png   bar chart of test accuracy
   - training_curves.png       loss/val_loss curves for each NN model

Run AFTER you've trained the models. Gracefully skips missing ones.
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from data_loader import get_data
from tokenizer_util import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------
def evaluate_logreg():
    import joblib
    if not os.path.exists("model_01_logreg.joblib"):
        return None
    vectorizer, model = joblib.load("model_01_logreg.joblib")
    _, X_test, _, y_test = get_data()
    y_pred = model.predict(vectorizer.transform(X_test))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred)


def _evaluate_pytorch(model_path, tok_path, model_builder, needs_lengths=False):
    """Generic loader for our PyTorch classifiers."""
    if not os.path.exists(model_path) or not os.path.exists(tok_path):
        return None
    from tokenizer_util import PAD_IDX

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    tokenizer = Tokenizer.load(tok_path)
    model = model_builder(ckpt["config"], tokenizer.vocab_len).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, X_test, _, y_test = get_data()
    max_len = ckpt["config"]["max_len"]
    ids = torch.tensor(tokenizer.encode_batch(X_test, max_len), dtype=torch.long)

    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            batch = ids[i:i+256].to(DEVICE)
            if needs_lengths:
                lengths = torch.tensor(
                    [max(1, (row != PAD_IDX).sum().item()) for row in batch],
                    dtype=torch.long,
                ).to(DEVICE)
                logits = model(batch, lengths)
            else:
                logits = model(batch)
            preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().numpy())

    return accuracy_score(y_test, preds), f1_score(y_test, preds)


def evaluate_mlp():
    from importlib import import_module
    mod = import_module("02_mlp_embedding".replace("02_", "model_mlp_"))  # placeholder
    # We need to import the model class. Easier: redefine it here.
    return _evaluate_pytorch(
        "model_02_mlp.pt", "tokenizer_02.json",
        model_builder=lambda cfg, vs: _build_mlp(cfg, vs),
    )


def _build_mlp(cfg, vocab_size):
    """Rebuild the MLP architecture from 02_mlp_embedding.py."""
    import torch.nn as nn
    from tokenizer_util import PAD_IDX

    class EmbeddingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, cfg["embed_dim"],
                                          padding_idx=PAD_IDX)
            self.fc1 = nn.Linear(cfg["embed_dim"], cfg["hidden_dim"])
            self.fc2 = nn.Linear(cfg["hidden_dim"], 1)
            self.dropout = nn.Dropout(cfg["dropout"])
            self.relu = nn.ReLU()

        def forward(self, x):
            emb = self.embedding(x)
            mask = (x != PAD_IDX).unsqueeze(-1).float()
            summed = (emb * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
            h = self.relu(self.fc1(pooled))
            h = self.dropout(h)
            return self.fc2(h).squeeze(-1)
    return EmbeddingMLP()


def evaluate_lstm():
    return _evaluate_pytorch(
        "model_03_lstm.pt", "tokenizer_03.json",
        model_builder=lambda cfg, vs: _build_lstm(cfg, vs),
        needs_lengths=True,
    )


def _build_lstm(cfg, vocab_size):
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from tokenizer_util import PAD_IDX

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, cfg["embed_dim"],
                                          padding_idx=PAD_IDX)
            self.lstm = nn.LSTM(
                input_size=cfg["embed_dim"], hidden_size=cfg["hidden"],
                num_layers=cfg["num_layers"], bidirectional=True,
                batch_first=True,
                dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
            )
            self.dropout = nn.Dropout(cfg["dropout"])
            self.classifier = nn.Linear(2 * cfg["hidden"], 1)

        def forward(self, ids, lengths):
            emb = self.embedding(ids)
            packed = pack_padded_sequence(
                emb, lengths.cpu(),
                batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            return self.classifier(self.dropout(h)).squeeze(-1)
    return BiLSTMClassifier()


def evaluate_transformer():
    return _evaluate_pytorch(
        "model_04_transformer.pt", "tokenizer_04.json",
        model_builder=lambda cfg, vs: _build_transformer(cfg, vs),
    )


def _build_transformer(cfg, vocab_size):
    import math
    import torch.nn as nn
    from tokenizer_util import PAD_IDX

    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, max_len, d_model):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                            * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, cfg["embed_dim"],
                                          padding_idx=PAD_IDX)
            self.pos_enc = SinusoidalPositionalEncoding(
                cfg["max_len"], cfg["embed_dim"])
            layer = nn.TransformerEncoderLayer(
                d_model=cfg["embed_dim"], nhead=cfg["num_heads"],
                dim_feedforward=cfg["ff_dim"], dropout=cfg["dropout"],
                batch_first=True, activation="relu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["num_blocks"])
            self.dropout = nn.Dropout(cfg["dropout"])
            self.classifier = nn.Sequential(
                nn.Linear(cfg["embed_dim"], cfg["embed_dim"]),
                nn.ReLU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(cfg["embed_dim"], 1),
            )

        def forward(self, ids):
            x = self.embedding(ids)
            x = self.pos_enc(x)
            pad_mask = (ids == PAD_IDX)
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            mask = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            return self.classifier(self.dropout(pooled)).squeeze(-1)
    return TransformerClassifier()


def evaluate_distilbert():
    path = "model_05_distilbert"
    if not os.path.isdir(path):
        return None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()

    _, X_test, _, y_test = get_data()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = X_test[i:i+64]
            enc = tok(batch, truncation=True, padding=True,
                      max_length=64, return_tensors="pt").to(DEVICE)
            out = model(**enc).logits.argmax(dim=-1).cpu().numpy()
            preds.extend(out)
    return accuracy_score(y_test, preds), f1_score(y_test, preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}\n")
    results = {}

    print("Evaluating LogReg...");           results["LogReg (TF-IDF)"]         = evaluate_logreg()
    print("Evaluating MLP...");              results["Embedding+MLP"]           = evaluate_mlp()
    print("Evaluating BiLSTM...");           results["BiLSTM"]                  = evaluate_lstm()
    print("Evaluating Transformer (scratch)..."); results["Transformer (scratch)"] = evaluate_transformer()
    print("Evaluating DistilBERT...");       results["DistilBERT (fine-tuned)"] = evaluate_distilbert()

    # Print summary
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
        print("\nSaved: comparison_accuracy.png")

    # Training curves
    histories = [
        ("Embedding+MLP", "history_02_mlp.json"),
        ("BiLSTM",        "history_03_lstm.json"),
        ("Transformer",   "history_04_transformer.json"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, path) in zip(axes, histories):
        if not os.path.exists(path):
            ax.set_title(f"{label}\n(not trained)"); ax.axis("off"); continue
        with open(path) as f:
            h = json.load(f)
        ax.plot(h["loss"], label="loss")
        ax.plot(h["val_loss"], label="val_loss")
        ax.set_title(label); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved: training_curves.png")


if __name__ == "__main__":
    main()

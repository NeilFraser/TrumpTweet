"""
tokenizer_util.py
-----------------
Simple word-level tokenizer used by all PyTorch classifiers in this project.

Why we wrote our own (instead of using torchtext / spaCy / HF):
- Tweets are short, so we don't need anything fancy
- We want to SAVE the tokenizer alongside the model and reload it
  during evaluation, which keeps eval clean and reproducible
- It's a tiny amount of code that everyone can read and understand,
  which matters for the grading criterion "explain your model"

Special tokens (always at fixed indices):
    0 = <pad>   used to right-pad all sequences to the same length
    1 = <unk>   any word not in the vocabulary at training time
"""

import json
import re
from collections import Counter


PAD_IDX = 0
UNK_IDX = 1
SPECIAL_TOKENS = ["<pad>", "<unk>"]


def simple_tokenize(text: str) -> list[str]:
    """
    Very simple tokenization: lowercase, then split on non-word characters
    while keeping punctuation as separate tokens. Good enough for tweets.

    Examples:
        "Sleepy Joe is at it again!"
        -> ['sleepy', 'joe', 'is', 'at', 'it', 'again', '!']
    """
    text = text.lower()
    # Replace URLs with a placeholder token so the model can learn that
    # "tweets containing a URL" is itself a feature, without exploding vocab
    text = re.sub(r"https?://\S+", " <url> ", text)
    # Split on word characters; keep punctuation as separate tokens
    tokens = re.findall(r"[a-z0-9<>_]+|[^\w\s]", text)
    return tokens


class Tokenizer:
    """Word-level tokenizer with save/load."""

    def __init__(self, vocab_size: int = 20_000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def fit(self, texts: list[str]) -> None:
        """Build the vocabulary from a list of training texts."""
        counter: Counter = Counter()
        for t in texts:
            counter.update(simple_tokenize(t))

        # Reserve indices 0..len(SPECIAL_TOKENS)-1 for special tokens
        self.idx2word = list(SPECIAL_TOKENS)
        for word, freq in counter.most_common():
            if freq < self.min_freq:
                break
            if len(self.idx2word) >= self.vocab_size:
                break
            self.idx2word.append(word)

        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        print(f"  Built vocabulary: {len(self.idx2word):,} tokens "
              f"(asked for max {self.vocab_size:,})")

    def encode(self, text: str, max_len: int) -> list[int]:
        """Encode a single text into a list of token IDs of length max_len.
        Pads with PAD_IDX (0) on the right; truncates if too long."""
        ids = [self.word2idx.get(w, UNK_IDX) for w in simple_tokenize(text)]
        ids = ids[:max_len]
        ids = ids + [PAD_IDX] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts: list[str], max_len: int) -> list[list[int]]:
        return [self.encode(t, max_len) for t in texts]

    @property
    def vocab_len(self) -> int:
        return len(self.idx2word)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "min_freq": self.min_freq,
                "idx2word": self.idx2word,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], min_freq=data["min_freq"])
        tok.idx2word = data["idx2word"]
        tok.word2idx = {w: i for i, w in enumerate(tok.idx2word)}
        return tok


if __name__ == "__main__":
    # Quick sanity test
    t = Tokenizer(vocab_size=100, min_freq=1)
    t.fit(["Sleepy Joe is at it again!", "The economy is GREAT!"])
    enc = t.encode("Joe is great", max_len=10)
    print("Encoded:", enc)
    t.save("/tmp/tok_test.json")
    t2 = Tokenizer.load("/tmp/tok_test.json")
    print("Reloaded vocab size:", t2.vocab_len)

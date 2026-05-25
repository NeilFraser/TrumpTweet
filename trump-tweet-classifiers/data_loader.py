"""
data_loader.py
--------------
Shared data loading utility for the real-vs-fake tweet classifier project.

What it does:
- Loads real Trump tweets from tweets.csv (filtering out retweets / deletions
  the same way Neil did when building his generator)
- Loads Neil's fake tweets generated with the 3-gram Markov model
  (this is the size his email said we should standardize on)
- Labels them: 1 = real, 0 = fake
- Returns a consistent train/test split with a fixed random seed so every
  classifier sees the exact same split, making comparisons fair
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths — edit these if your files live somewhere else.
REAL_PATH = "tweets.csv"
FAKE_PATH = "fake_tweets_3.json"

# Fixed random seed so every classifier gets the same split.
RANDOM_SEED = 42


def load_real_tweets(path: str = REAL_PATH) -> list[str]:
    """Load and clean real Trump tweets. Filter retweets + deleted, same as Neil."""
    df = pd.read_csv(path)
    df = df[df["isRetweet"] == "f"]
    df = df[df["isDeleted"] == "f"]
    # Drop near-empty tweets (just a URL or whitespace) — also matches Neil's
    # 45,454-tweet figure once filtered.
    df = df[df["text"].str.strip().str.len() > 0]
    return df["text"].astype(str).tolist()


def load_fake_tweets(path: str = FAKE_PATH) -> list[str]:
    """Load Neil's 3-gram fake tweets."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_data(test_size: float = 0.2, max_per_class: int | None = None):
    """
    Build the combined dataset with labels and split into train/test.

    Args:
        test_size: fraction held out for testing (default 20%)
        max_per_class: optional cap for quick experiments
                       (e.g. set to 5000 to train on a small subset
                        while you're debugging)

    Returns:
        X_train, X_test, y_train, y_test
        Where X is a list[str] (tweet texts) and y is a list[int]
        (1 = real, 0 = fake).
    """
    real = load_real_tweets()
    fake = load_fake_tweets()

    if max_per_class is not None:
        real = real[:max_per_class]
        fake = fake[:max_per_class]

    print(f"Loaded {len(real):,} real tweets and {len(fake):,} fake tweets.")

    # Labels: 1 for real, 0 for fake.
    X = real + fake
    y = [1] * len(real) + [0] * len(fake)

    # Stratified split keeps the class balance equal in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"Train size: {len(X_train):,}   Test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Sanity check — run "python data_loader.py" to confirm everything loads.
    X_train, X_test, y_train, y_test = get_data()
    print("\nSample real tweet:", X_train[y_train.index(1)][:120])
    print("Sample fake tweet:", X_train[y_train.index(0)][:120])

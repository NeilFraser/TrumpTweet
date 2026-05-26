import json
import random
import re
import torch
import torch.nn as nn
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import accuracy_score

# --- REPRODUCIBILITY ---
random.seed(42)
torch.manual_seed(42)

# 1. GLOBAL SETTINGS
MAX_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 100 # Fixed for GloVe 100d
GLOVE_PATH = 'glove.6B.100d.txt'

# 2. DATA LOADING (Trigram)
print("Loading Trigram datasets...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 3. PREPROCESSING
def simple_tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data_raw = data[:train_size]
val_data_raw = data[train_size : train_size + val_size]
test_data_raw = data[train_size + val_size :]

# --- Fixed Vocabulary Building Block ---
all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))

word_counts = Counter(all_words)
# First, filter the words that appear more than once
filtered_words = [word for word, count in word_counts.items() if count > 1]

# Second, map them to consecutive indices starting from 2
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1
# ---------------------------------------

def preprocess_set(data_list, vocab, max_len):
    inputs, labels = [], []
    for text, label in data_list:
        nums = [vocab.get(token, 1) for token in simple_tokenize(text)]
        nums = nums[:max_len] + [0] * max(0, max_len - len(nums))
        inputs.append(nums)
        labels.append(label)
    return torch.tensor(inputs), torch.tensor(labels)

train_inputs, train_labels = preprocess_set(train_data_raw, vocab, MAX_LEN)
val_inputs, val_labels = preprocess_set(val_data_raw, vocab, MAX_LEN)
test_inputs, test_labels = preprocess_set(test_data_raw, vocab, MAX_LEN)

# 4. GLOVE LOADING (Outside of Objective to save time)
def load_glove_embeddings(path, word_to_idx, embedding_dim):
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.randn(vocab_size, embedding_dim)
    print(f"Pre-loading GloVe from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_idx:
                vector = torch.tensor([float(x) for x in values[1:]])
                embedding_matrix[word_to_idx[word]] = vector
    return embedding_matrix

glove_matrix = load_glove_embeddings(GLOVE_PATH, vocab, EMBED_DIM)

# 5. MODEL ARCHITECTURE (Flexible for Tuning)
class TunableGloVeClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate, spatial_dr_rate, embedding_matrix):
        super(TunableGloVeClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.spatial_dropout = nn.Dropout1d(p=spatial_dr_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0, 
                            num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded).permute(0, 2, 1)
        _, (hidden, _) = self.lstm(embedded)
        # Use final hidden states of the last layer
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(out))

# 6. OPTUNA OBJECTIVE FUNCTION
def objective(trial):
    # --- Suggesting Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    spatial_dr_rate = trial.suggest_float("spatial_dr_rate", 0.1, 0.4)
    
    # Model Setup
    model = TunableGloVeClassifier(len(vocab), EMBED_DIM, hidden_dim, num_layers, dropout_rate, spatial_dr_rate, glove_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

    # Short Training for Tuning (3-5 Epochs)
    for epoch in range(4):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            preds = (model(inputs).squeeze() > 0.5).int()
            all_preds.extend(preds.tolist())
    
    accuracy = accuracy_score(val_labels.tolist(), all_preds)
    return accuracy

# 7. RUNNING THE STUDY
print("\nStarting Hyperparameter Optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20) # You can increase n_trials for better results

print("\nBest Hyperparameters found:")
print(study.best_params)

# 8. FINAL TRAINING WITH BEST PARAMS
best = study.best_params
final_model = TunableGloVeClassifier(len(vocab), EMBED_DIM, best['hidden_dim'], 
                                     best['num_layers'], best['dropout_rate'], 
                                     best['spatial_dr_rate'], glove_matrix)

# Final full training (Optional: Train for more epochs here)
print(f"\nFinal training with accuracy: {study.best_value:.4f}")
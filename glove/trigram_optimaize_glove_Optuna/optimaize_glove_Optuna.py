import json
import random
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- SETTINGS ---
random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. BEST HYPERPARAMETERS (Found by Optuna)
MAX_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 1
LR = 0.002010848471212492
DROPOUT = 0.12209395466217875
SPATIAL_DROPOUT = 0.20498604834989784
EPOCHS = 15  # Set high, Early Stopping will handle it
PATIENCE = 3 # Stop if no improvement after 3 epochs

# 2. DATA LOADING & PREPROCESSING
print("Loading Trigram datasets for Final Training...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

def simple_tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

train_size, val_size = int(0.8 * len(data)), int(0.1 * len(data))
train_data_raw, val_data_raw = data[:train_size], data[train_size:train_size+val_size]
test_data_raw = data[train_size+val_size:]

all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))
word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 1]
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"], vocab["{UNK}"] = 0, 1

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

train_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

# 3. GLOVE LOADING
def load_glove_embeddings(path, word_to_idx, embedding_dim):
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.randn(vocab_size, embedding_dim)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if values[0] in word_to_idx:
                embedding_matrix[word_to_idx[values[0]]] = torch.tensor([float(x) for x in values[1:]])
    return embedding_matrix

glove_matrix = load_glove_embeddings('glove.6B.100d.txt', vocab, EMBED_DIM)

# 4. FINAL MODEL ARCHITECTURE
class FinalGloVeClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, spatial_dr, embedding_matrix):
        super(FinalGloVeClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.spatial_dropout = nn.Dropout1d(p=spatial_dr)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, 
                            dropout=dropout if num_layers > 1 else 0, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded).permute(0, 2, 1)
        _, (hidden, _) = self.lstm(embedded)
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(out))

# 5. TRAINING WITH EARLY STOPPING
model = FinalGloVeClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, SPATIAL_DROPOUT, glove_matrix).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()
# Reduces LR when accuracy plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

best_val_loss = float('inf')
counter = 0

print(f"\nTraining on: {device}")
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs).squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            v_loss += criterion(model(inputs).squeeze(), labels.float()).item()
    
    avg_v_loss = v_loss / len(val_loader)
    scheduler.step(avg_v_loss)
    print(f"Epoch {epoch+1} -> Val Loss: {avg_v_loss:.4f}")

    # Early Stopping check
    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        torch.save(model.state_dict(), 'best_trigram_glove_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print("Early Stopping triggered!")
            break

# 6. FINAL TEST EVALUATION
model.load_state_dict(torch.load('best_trigram_glove_model.pth'))
model.eval()
with torch.no_grad():
    test_inputs = test_inputs.to(device)
    test_outputs = model(test_inputs).squeeze()
    predictions = (test_outputs > 0.5).int()
accuracy = accuracy_score(test_labels, predictions)

print(f"\n[FINAL RECORD] Trigram Accuracy: {accuracy*100:.2f}%")
import json
import random
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- REPRODUCIBILITY SEEDS ---
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 1. Hyperparameters
MAX_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# 2. Data Loading (Focus: BIGRAM)
print("Loading Bigram datasets...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_2.json', 'r', encoding='utf-8') as f: # Bigram File
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 3. Preprocessing & Vocabulary
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

all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))

word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 1]
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1

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

# 4. Advanced Architecture (Bi-LSTM + Spatial Dropout)
class TrumpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TrumpClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.spatial_dropout = nn.Dropout1d(p=0.2) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded)
        embedded = embedded.permute(0, 2, 1)
        
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        out_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(out_hidden))

# 5. Training Setup
model = TrumpClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, val_losses = [], []

# 6. Training Loop
print("Starting Training for Bigram Model...")
for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs).squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            v_loss += criterion(model(inputs).squeeze(), labels.float()).item()

    train_losses.append(t_loss / len(train_loader))
    val_losses.append(v_loss / len(val_loader))
    print(f"Epoch {epoch+1} -> Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

# --- OUTPUT 1: SAVE BIGRAM MODEL (.pth) ---
torch.save(model.state_dict(), 'trump_classifier_bigram_final.pth')
print("\n[SUCCESS] Bigram model saved as 'trump_classifier_bigram_final.pth'")

# 7. Evaluation
model.eval()
with torch.no_grad():
    predictions = (model(test_inputs).squeeze() > 0.5).int()
accuracy = accuracy_score(test_labels, predictions)
print(f"Final Bigram Test Accuracy: {accuracy*100:.2f}%")

# --- OUTPUT 2: GENERATE BIGRAM LEARNING CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
plt.plot(val_losses, label='Validation Loss', color='red', marker='s')
plt.title('Learning Curve: Bi-LSTM + Spatial Dropout (Bigram)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('final_learning_curve_bigram.png', dpi=300)
print("[SUCCESS] Bigram learning curve saved as 'final_learning_curve_bigram.png'")
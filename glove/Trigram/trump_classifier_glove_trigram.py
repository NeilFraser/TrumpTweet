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

# --- REPRODUCIBILITY SEEDS ---
# Ensuring consistent results for comparison with previous runs
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 1. HYPERPARAMETERS
# MAX_LEN=50 is kept from your most successful model to capture full context
MAX_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 100  # Must match GloVe file dimension (100d)
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# 2. DATA LOADING
print("Loading datasets...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
    fake_tweets = json.load(f)

# Labeling: 1 for Real Trump, 0 for Trigram Fake
data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 3. PREPROCESSING & VOCABULARY
def simple_tokenize(tweet):
    """Basic NLP cleaning: lowercase, URL tokens, and punctuation spacing."""
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

# Split Data (80% Train, 10% Val, 10% Test)
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data_raw = data[:train_size]
val_data_raw = data[train_size : train_size + val_size]
test_data_raw = data[train_size + val_size :]

# Building Vocabulary from training set
all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))

word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 1]
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1

def preprocess_set(data_list, vocab, max_len):
    """Converts raw text into padded numerical tensor sequences."""
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

# 4. GLOVE EMBEDDING LOADER
def load_glove_embeddings(path, word_to_idx, embedding_dim):
    """
    Parses GloVe text file and creates a weight matrix for the model.
    Words not found in GloVe are initialized with random normal distribution.
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.randn(vocab_size, embedding_dim)
    
    print(f"Loading GloVe embeddings from {path}...")
    found = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_to_idx:
                    vector = torch.tensor([float(x) for x in values[1:]])
                    embedding_matrix[word_to_idx[word]] = vector
                    found += 1
        print(f"Successfully mapped {found}/{vocab_size} words from GloVe.")
    except Exception as e:
        print(f"Error loading GloVe: {e}")
    return embedding_matrix

# 5. MODEL ARCHITECTURE (Bi-LSTM + Spatial Dropout + GloVe)
class GloVeTrumpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix):
        super(GloVeTrumpClassifier, self).__init__()
        # Initialize with Pre-trained GloVe weights
        # freeze=False allows fine-tuning GloVe vectors for Trump-specific vocabulary
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        
        # Spatial Dropout drops entire 1D feature maps instead of individual neurons
        self.spatial_dropout = nn.Dropout1d(p=0.2)
        
        # Bidirectional LSTM processes sequences from both directions
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2, num_layers=2)
        
        # Final layer: Input is hidden_dim * 2 because of Bidirectional architecture
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply Spatial Dropout on the embedding dimension
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded).permute(0, 2, 1)
        
        # LSTM output
        _, (hidden, _) = self.lstm(embedded)
        
        # Concatenate the final hidden state from forward and backward passes
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(out))

# 6. TRAINING INITIALIZATION
glove_matrix = load_glove_embeddings('glove.6B.100d.txt', vocab, EMBED_DIM)
model = GloVeTrumpClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, glove_matrix)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = [], []
print("\nStarting Training with GloVe-enhanced Bi-LSTM...")

# 7. TRAINING LOOP
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

    avg_train = t_loss / len(train_loader)
    avg_val = v_loss / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

# 8. FINAL EVALUATION
model.eval()
with torch.no_grad():
    # Convert test inputs to predictions (threshold 0.5)
    test_outputs = model(test_inputs).squeeze()
    predictions = (test_outputs > 0.5).int()

accuracy = accuracy_score(test_labels, predictions)
print(f"\nFinal Test Accuracy with GloVe-enhanced Model: {accuracy*100:.2f}%")

# 9. SAVE MODEL & RESULTS
torch.save(model.state_dict(), 'trump_classifier_glove_final.pth')
print("[SUCCESS] Model saved as 'trump_classifier_glove_final.pth'")

# Generate Learning Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.title('GloVe-enhanced Bi-LSTM Learning Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss (BCE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve_glove.png')
print("[SUCCESS] Learning curve saved as 'learning_curve_glove.png'")
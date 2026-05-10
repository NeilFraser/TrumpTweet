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
# Setting seeds to ensure the results are consistent across different runs
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Hyperparameters
# Main optimization change: MAX_LEN reduced from 50 to 35 to focus on actual content
MAX_LEN = 35 
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# 2. Data Loading (Focusing on Trigram fake data)
print("Loading datasets...")
try:
    with open('tweets.json', 'r', encoding='utf-8') as f:
        real_tweets = json.load(f)
    with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
        fake_tweets = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure data files exist in the same directory.")
    exit()

# Labeling: 1 for Real, 0 for Fake
data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 3. Preprocessing & Vocabulary Building
def simple_tokenize(tweet):
    """Basic cleaning: lowercase, URL replacement, and punctuation spacing."""
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

# Split data into Train (80%), Validation (10%), and Test (10%)
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data_raw = data[:train_size]
val_data_raw = data[train_size : train_size + val_size]
test_data_raw = data[train_size + val_size :]

# Build vocabulary from training data only
all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))

# Filter out words that appear only once to reduce noise
word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 1]
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1

def preprocess_set(data_list, vocab, max_len):
    """Converts text lists into padded numeric tensors."""
    inputs, labels = [], []
    for text, label in data_list:
        # Convert tokens to IDs; use {UNK} ID (1) if word is not in vocab
        nums = [vocab.get(token, 1) for token in simple_tokenize(text)]
        # Truncate or Pad to reach exactly max_len
        nums = nums[:max_len] + [0] * max(0, max_len - len(nums))
        inputs.append(nums)
        labels.append(label)
    return torch.tensor(inputs), torch.tensor(labels)

# Prepare numeric datasets
train_inputs, train_labels = preprocess_set(train_data_raw, vocab, MAX_LEN)
val_inputs, val_labels = preprocess_set(val_data_raw, vocab, MAX_LEN)
test_inputs, test_labels = preprocess_set(test_data_raw, vocab, MAX_LEN)

# Create DataLoaders for batching
train_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

# 4. Model Architecture (Bi-LSTM + Spatial Dropout)
class TrumpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TrumpClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Dropout1d serves as Spatial Dropout (dropping entire channels)
        self.spatial_dropout = nn.Dropout1d(p=0.2) 
        # Bidirectional LSTM with 2 layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2, num_layers=2)
        # Final layer: hidden_dim * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        
        # Spatial dropout expects (Batch, Channel, Length)
        embedded = embedded.permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded)
        embedded = embedded.permute(0, 2, 1)
        
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate the final hidden states from both directions
        out_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.sigmoid(self.fc(out_hidden))

# 5. Training Setup
model = TrumpClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM)
criterion = nn.BCELoss() # Binary Cross Entropy for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, val_losses = [], []

# 6. Training Loop
print(f"Starting Training (Bi-LSTM + Spatial Dropout) with MAX_LEN = {MAX_LEN}...")
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    t_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(inputs).squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    # Validation Phase
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

# --- OUTPUT 1: SAVE THE MODEL (.pth) ---
torch.save(model.state_dict(), 'trump_classifier_trigram_maxlen35.pth')
print("\n[SUCCESS] Model weights saved as 'trump_classifier_trigram_maxlen35.pth'")

# 7. Final Evaluation on Test Set
print("\nEvaluating on Trigram test set...")
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs).squeeze()
    predictions = (test_outputs > 0.5).int()
accuracy = accuracy_score(test_labels, predictions)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# --- OUTPUT 2: GENERATE LEARNING CURVE PNG ---
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o', linewidth=2)
plt.plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
plt.title('Learning Curve: Bi-LSTM + Spatial Dropout (MAX_LEN=35)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('learning_curve_maxlen35.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Learning curve plot saved as 'learning_curve_maxlen35.png'")

# --- OUTPUT 3: STATISTICAL REPORT TXT ---
report_content = f"""
TRUMP TWEET CLASSIFIER - OPTIMIZATION STEP 1 (Reduced MAX_LEN)
--------------------------------------------------------------
Hyperparameters:
- Max Sequence Length: {MAX_LEN} (Reduced from 50)
- Epochs: {EPOCHS}
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LEARNING_RATE}

Dataset Statistics:
- Total Samples: {len(data)}
- Real Tweets: {len(real_tweets)}
- Fake Tweets (Trigram): {len(fake_tweets)}
- Training Set Size: {len(train_data_raw)}
- Validation Set Size: {len(val_data_raw)}
- Test Set Size: {len(test_data_raw)}
- Vocabulary Size: {len(vocab)}

Training History:
"""
for i in range(EPOCHS):
    report_content += f"Epoch {i+1}: Train Loss = {train_losses[i]:.4f}, Val Loss = {val_losses[i]:.4f}\n"

report_content += f"\nFinal Performance:\n- Trigram Test Accuracy: {accuracy*100:.2f}%\n"

with open('classification_stats_maxlen35.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("[SUCCESS] Statistical report saved as 'classification_stats_maxlen35.txt'")
print("\nStep 1 Complete. Review your accuracy and let's move to Step 2 (Attention Mechanism)!")
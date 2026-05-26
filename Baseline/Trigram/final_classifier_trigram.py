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
torch.backends.cudnn.benchmark = False
# -----------------------------

# 1. Hyperparameters
MAX_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# 2. Data Loading (Updated for Trigram)
print("Loading Trigram datasets...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: # Target Trigram file
    fake_tweets = json.load(f)

# 3. Labeling and Shuffling
data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 4. Preprocessing Function
def simple_tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

# 5. Data Splitting and Vocabulary Building
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
        if len(nums) < max_len:
            nums = nums + [0] * (max_len - len(nums))
        else:
            nums = nums[:max_len]
        inputs.append(nums)
        labels.append(label)
    return torch.tensor(inputs), torch.tensor(labels)

train_inputs, train_labels = preprocess_set(train_data_raw, vocab, MAX_LEN)
val_inputs, val_labels = preprocess_set(val_data_raw, vocab, MAX_LEN)

train_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

# 6. Model Architecture (LSTM)
class TrumpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TrumpClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# 7. Training Setup
model = TrumpClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

print(f"Preparation complete. Vocab size: {len(vocab)}")
print("Starting training on Trigram data...")

# 8. Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            val_loss = criterion(outputs, labels.float())
            total_val_loss += val_loss.item()

    avg_train = total_train_loss / len(train_loader)
    avg_val = total_val_loss / len(val_loader)
    train_losses.append(avg_train)
    val_losses.append(avg_val)
    
    print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

# Save the Trigram Model
torch.save(model.state_dict(), 'trump_classifier_trigram.pth')

# 9. Final Evaluation
print("\nEvaluating on Trigram test set...")
model.eval()
test_inputs, test_labels = preprocess_set(test_data_raw, vocab, MAX_LEN)
with torch.no_grad():
    test_outputs = model(test_inputs).squeeze()
    predictions = (test_outputs > 0.5).int()

accuracy = accuracy_score(test_labels, predictions)
print(f"Final Trigram Test Accuracy: {accuracy*100:.2f}%")

# 10. Generating High-Quality Plot for Presentation
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.title('Trigram Model Learning Curve', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('learning_curve_trigram.png', dpi=300, bbox_inches='tight')

# 11. Statistical Report for Submission
report_content = f"""
TRUMP TWEET CLASSIFIER - TRIGRAM REPORT
-------------------------------------------
Dataset Statistics:
- Total Samples: {len(data)}
- Real Tweets: {len(real_tweets)}
- Fake Tweets (Trigram): {len(fake_tweets)}
- Training Set Size: {len(train_data_raw)}
- Validation Set Size: {len(val_data_raw)}
- Test Set Size: {len(test_data_raw)}

Training History:
"""
for i in range(EPOCHS):
    report_content += f"Epoch {i+1}: Train Loss = {train_losses[i]:.4f}, Val Loss = {val_losses[i]:.4f}\n"

report_content += f"\nFinal Performance:\n- Trigram Test Accuracy: {accuracy*100:.2f}%\n"

with open('classification_stats_trigram.txt', 'w', encoding='utf-8') as f:
    f.write(report_content)
print("\nAll files saved. Ready for analysis.")
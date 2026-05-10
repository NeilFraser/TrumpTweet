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
# Ensuring consistent results across runs
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Hyperparameters
# Reverting to the original report's length (50) as Attention handles padding effectively 
MAX_LEN = 50 
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# 2. Data Loading (Trigram focus)
print("Loading datasets...")
try:
    with open('tweets.json', 'r', encoding='utf-8') as f:
        real_tweets = json.load(f)
    with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
        fake_tweets = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure data files exist.")
    exit()

# Labeling: 1 for Real, 0 for Fake
data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

# 3. Preprocessing & Vocabulary
def simple_tokenize(tweet):
    """Basic cleaning: lowercase, URL tokenization, and punctuation handling."""
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data_raw = data[:train_size]
val_data_raw = data[train_size : train_size + val_size]
test_data_raw = data[train_size + val_size :]

# Build vocab from training set
all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))

word_counts = Counter(all_words)
filtered_words = [word for word, count in word_counts.items() if count > 1]
vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1

def preprocess_set(data_list, vocab, max_len):
    """Converts text into padded sequences of numerical IDs."""
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

# 4. Step 2 Model: Bi-LSTM + Spatial Dropout + Attention Mechanism
# This architecture was suggested in the technical report's future work 
class AttentionTrumpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(AttentionTrumpClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.spatial_dropout = nn.Dropout1d(p=0.2)
        # Bidirectional LSTM processes text from both ends 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True, dropout=0.2, num_layers=2)
        
        # --- Attention Layers ---
        # Learns to assign importance scores to each word hidden state
        self.attention_fc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)
        
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        embedded = self.spatial_dropout(embedded).permute(0, 2, 1)
        
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        
        # --- Attention Calculation ---
        # 1. Compute 'energy' for each token
        energy = torch.tanh(self.attention_fc(lstm_out))
        # 2. Compute raw scores
        attention_scores = self.v(energy) # (batch, seq_len, 1)
        # 3. Normalize scores into weights that sum to 1
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # 4. Multiply each word's hidden state by its weight (Weighted Sum)
        # Effectively focuses on keywords like 'Fake' or 'Great' while ignoring padding
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        return self.sigmoid(self.fc(context_vector))

# 5. Training Setup
model = AttentionTrumpClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses, val_losses = [], []

# 6. Training Loop
print(f"Starting Training (Step 2: Attention) with MAX_LEN = {MAX_LEN}...")
for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
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

# --- OUTPUT 1: SAVE STEP 2 MODEL ---
torch.save(model.state_dict(), 'trump_classifier_attention_maxlen50.pth')

# 7. Evaluation
model.eval()
with torch.no_grad():
    predictions = (model(test_inputs).squeeze() > 0.5).int()
accuracy = accuracy_score(test_labels, predictions)
print(f"\nFinal Test Accuracy (Attention + MAX_LEN=50): {accuracy*100:.2f}%")

# --- OUTPUT 2: GENERATE LEARNING CURVE ---
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='s')
plt.title(f'Step 2 Learning Curve (Attention, MAX_LEN={MAX_LEN})', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('learning_curve_step2_attention_maxlen50.png')

# --- OUTPUT 3: STATISTICAL REPORT ---
with open('stats_step2_attention_maxlen50.txt', 'w') as f:
    f.write(f"STEP 2: ATTENTION MECHANISM REPORT\n")
    f.write(f"----------------------------------\n")
    f.write(f"Max Sequence Length: {MAX_LEN}\n")
    f.write(f"Final Test Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Epochs: {EPOCHS}\n")

print("\n[SUCCESS] Model saved as 'trump_classifier_attention_maxlen50.pth'")
print("[SUCCESS] Learning curve and stats report generated.")
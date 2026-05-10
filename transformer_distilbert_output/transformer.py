import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW  # اصلاح شده: استفاده از نسخه رسمی PyTorch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm  # برای مشاهده درصد پیشرفت کار

# --- REPRODUCIBILITY ---
random.seed(42)
torch.manual_seed(42)

# --- انتخاب هوشمند سخت‌افزار (Apple Silicon) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Running on: Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Running on: NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️ Running on: CPU (This will be slow)")

# 1. HYPERPARAMETERS
MAX_LEN = 64
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
MODEL_NAME = 'distilbert-base-uncased'

# 2. DATA LOADING
print("Loading Trigram datasets...")
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_3.json', 'r', encoding='utf-8') as f: 
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.shuffle(data)

train_split = int(0.8 * len(data))
train_data = data[:train_split]
test_data = data[train_split:]

# 3. TRANSFORMER DATASET CLASS (Optimized)
class TrumpTransformerDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # استفاده از فراخوانی مستقیم برای جلوگیری از خطاهای احتمالی نسخه پایتون
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. PREPARE DATALOADERS
print("Initializing Tokenizer and Dataloaders...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = TrumpTransformerDataset(train_data, tokenizer, MAX_LEN)
test_dataset = TrumpTransformerDataset(test_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 5. INITIALIZE MODEL
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)

# 6. TRAINING LOOP (With Progress Bar)
train_losses = []
print("\nStarting Transformer Fine-tuning...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    # افزودن نوار پیشرفت برای هر اپوک
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Summary Epoch {epoch+1} -> Avg Loss: {avg_loss:.4f}")

# 7. EVALUATION
print("\nEvaluating on Test Set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✅ Final Transformer Accuracy: {accuracy*100:.2f}%")

# 8. SAVING OUTPUTS
output_dir = './transformer_distilbert_output'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, marker='o')
plt.title('Training Loss')
plt.savefig('learning_curve_transformer.png')
print(f"Results saved in {output_dir} and learning_curve_transformer.png")
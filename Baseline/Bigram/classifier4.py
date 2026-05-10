import json
import random
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# --- بخش‌های قبلی (بارگذاری و توکنایز) ---
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_2.json', 'r', encoding='utf-8') as f:
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.seed(42)
random.shuffle(data)

def simple_tokenize(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet)
    return tweet.split()

train_size = int(0.8 * len(data))
train_data_raw = data[:train_size]
val_data_raw = data[train_size : train_size + int(0.1 * len(data))]

all_words = []
for text, label in train_data_raw:
    all_words.extend(simple_tokenize(text))
word_counts = Counter(all_words)
vocab = {word: i + 2 for i, (word, count) in enumerate(word_counts.items()) if count > 1}
vocab["{PAD}"] = 0
vocab["{UNK}"] = 1

# --- مرحله ۴: تبدیل به تنسورهای پایتورچ و پدینگ ---

MAX_LEN = 50 # حداکثر طول هر توییت (ابرپارامتر که باید تنظیم شود)

def preprocess_set(data_list, vocab, max_len):
    inputs = []
    labels = []
    for text, label in data_list:
        # تبدیل به عدد
        nums = [vocab.get(token, 1) for token in simple_tokenize(text)]
        # پدینگ (یکسان‌سازی طول)
        if len(nums) < max_len:
            nums = nums + [0] * (max_len - len(nums))
        else:
            nums = nums[:max_len]
        inputs.append(nums)
        labels.append(label)
    
    # تبدیل به تنسور (فرمت مخصوص پایتورچ)
    return torch.tensor(inputs), torch.tensor(labels)

# آماده‌سازی داده‌های آموزش و ارزیابی
train_inputs, train_labels = preprocess_set(train_data_raw, vocab, MAX_LEN)
val_inputs, val_labels = preprocess_set(val_data_raw, vocab, MAX_LEN)

# ساخت بسته‌های داده (Batching)
BATCH_SIZE = 32 # هر بار ۳۲ توییت بررسی شود

train_loader = DataLoader(TensorDataset(train_inputs, train_labels), shuffle=True, batch_size=BATCH_SIZE)
val_loader = DataLoader(TensorDataset(val_inputs, val_labels), batch_size=BATCH_SIZE)

print(f"تنسور آموزش ساخته شد: {train_inputs.shape}")
print(f"تعداد دسته‌ها (Batches) در آموزش: {len(train_loader)}")
print(f"آماده‌سازی برای پایتورچ با موفقیت انجام شد.")
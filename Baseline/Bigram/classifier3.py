import json
import random
import re
from collections import Counter

# ۱. بارگذاری و برچسب‌گذاری (مانند قبل)
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_2.json', 'r', encoding='utf-8') as f:
    fake_tweets = json.load(f)

data = [(t, 1) for t in real_tweets] + [(t, 0) for t in fake_tweets]
random.seed(42)
random.shuffle(data)

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

# ۲. تابع تکه‌تکه کردن متن (Tokenization) - مشابه generator.py
def simple_tokenize(tweet):
    tweet = tweet.lower() # کوچک کردن حروف برای یکسانی
    tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet) # جایگزینی لینک‌ها
    tweet = re.sub(r'([ \.!?,;:])', r' \1 ', tweet) # جدا کردن علائم نگارشی
    return tweet.split()

# ۳. ساخت فرهنگ لغت از داده‌های آموزش
all_words = []
for text, label in train_data:
    all_words.extend(simple_tokenize(text))

word_counts = Counter(all_words)
# فقط کلماتی را نگه می‌داریم که حداقل ۲ بار تکرار شده‌اند تا کلمات غلط املایی حذف شوند
vocab = {word: i + 2 for i, (word, count) in enumerate(word_counts.items()) if count > 1}
vocab["{PAD}"] = 0  # برای پر کردن جملات کوتاه
vocab["{UNK}"] = 1  # برای کلمات ناشناخته

# ۴. تبدیل یک توییت نمونه به اعداد
def text_to_numbers(text, vocab):
    tokens = simple_tokenize(text)
    return [vocab.get(token, 1) for token in tokens] # ۱ کد کلمه ناشناخته است

print(f"تعداد کلمات منحصر به فرد در فرهنگ لغت: {len(vocab)}")

sample_tweet = train_data[0][0]
numeric_tweet = text_to_numbers(sample_tweet, vocab)

print("\n--- تست تبدیل ---")
print(f"متن اصلی: {sample_tweet}")
print(f"لیست اعداد: {numeric_tweet}")
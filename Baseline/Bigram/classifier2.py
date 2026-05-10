import json
import random

# ۱. بارگذاری داده‌ها (مانند قبل)
with open('tweets.json', 'r', encoding='utf-8') as f:
    real_tweets = json.load(f)
with open('fake_tweets_2.json', 'r', encoding='utf-8') as f:
    fake_tweets = json.load(f)

# ۲. برچسب‌گذاری: واقعی = 1، جعلی = 0
# ما لیستی از جفت‌های (متن، برچسب) می‌سازیم
data = []
for t in real_tweets:
    data.append((t, 1))
for t in fake_tweets:
    data.append((t, 0))

# ۳. مخلوط کردن داده‌ها (Shuffle)
# خیلی مهم است که داده‌ها کاملاً قاطی شوند تا مدل ترتیب را حفظ نکند
random.seed(42) # برای اینکه هر بار نتایج یکسان بگیریم
random.shuffle(data)

# ۴. تقسیم‌بندی داده‌ها (مثلاً ۸۰٪ آموزش، ۱۰٪ ارزیابی، ۱۰٪ تست)
total_count = len(data)
train_size = int(0.8 * total_count)
val_size = int(0.1 * total_count)

train_data = data[:train_size]
val_data = data[train_size : train_size + val_size]
test_data = data[train_size + val_size :]

print(f"کل داده‌ها: {total_count}")
print(f"تعداد داده‌های آموزش (Train): {len(train_data)}")
print(f"تعداد داده‌های ارزیابی (Validation): {len(val_data)}")
print(f"تعداد داده‌های تست (Test): {len(test_data)}")

# نمایش یک نمونه از داده‌های برچسب‌گذاری شده
sample_text, sample_label = train_data[0]
print(f"\nنمونه تصادفی از بخش آموزش:")
print(f"متن: {sample_text}")
print(f"برچسب: {sample_label} (یعنی {'واقعی' if sample_label==1 else 'جعلی'})")
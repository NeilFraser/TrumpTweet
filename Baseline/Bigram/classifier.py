import json

try:
    with open('tweets.json', 'r', encoding='utf-8') as f:
        real_tweets = json.load(f)
    
    with open('fake_tweets_2.json', 'r', encoding='utf-8') as f:
        fake_tweets = json.load(f)

    print(f"تعداد توییت‌های واقعی: {len(real_tweets)}")
    print(f"تعداد توییت‌های جعلی (Bigram): {len(fake_tweets)}")

    print("\n--- نمونه واقعی ---")
    print(real_tweets[0])
    print("\n--- نمونه جعلی ---")
    print(fake_tweets[0])

except FileNotFoundError as e:
    print(f"خطا: فایل مورد نظر پیدا نشد. لطفا نام فایل را بررسی کنید: {e}")
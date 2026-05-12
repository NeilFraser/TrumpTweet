import json
import random
import re

# Creates a Markov chain database of token frequencies from the input tweets,
# and then uses it to generate new fake tweets.
# The n-gram size can be adjusted by changing the SIZE constant.

SIZE = 4  # The n-gram size to use for the database.
INPUT_FILE = 'tweets.json'
OUTPUT_FILE = 'fake_tweets_%d.json' % SIZE

tweets = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))

def tokenize(tweet):
  # Replace URLs with a special token.
  tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
  tweet = re.sub(r'\n', '{CR}', tweet)
  # Wrap punctuation in special tokens, so that they get treated as separate tokens.
  tweet = re.sub(r'([ \.!?,;:])', r'{\1}', tweet)
  tweet = '{START}' + tweet + '{END}'
  # Inject null characters around the special tokens, so that we can split on them.
  tweet = re.sub(r'{', '\0{', tweet)
  tweet = re.sub(r'}', '}\0', tweet)
  tweet = re.sub(r'\0+', '\0', tweet)  # No repeated null characters.
  # Split on the null character.
  tokens = tweet.split('\0')
  # Remove empty tokens.
  tokens = [token for token in tokens if token]
  return tokens


def detokenize(tokens):
  # Remove the special tokens.
  tokens = [token for token in tokens if token not in ['{START}', '{END}']]
  # Join the tokens together.
  tweet = ''.join(tokens)
  # Replace the special tokens with their original values.
  tweet = re.sub(r'\{URL\}', randomUrl(), tweet)
  tweet = re.sub(r'\{CR\}', '\n', tweet)
  tweet = re.sub(r'\{([ \.!?,;:“”&])\}', r'\1', tweet)
  return tweet


def randomUrl():
  # E.g. https://t.co/3fs1oPVnAx
  return 'https://t.co/' + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=10))


# Tokenize the tweets.
tokenized_tweets = [tokenize(tweet) for tweet in tweets]

# Build a database of token frequencies, with n-gram of the specified size.
def build_database(tokenized_tweets, size):
  data = {}
  for tweet in tokenized_tweets:
    grams = []
    for token in tweet:
      if len(grams) > 0:
        key = tuple(grams)  # Use a tuple of the previous tokens as the key.
        if key not in data:
          data[key] = {'\0': 0}  # Initialize with zero total.
        if token not in data[key]:
          data[key][token] = 0
        data[key][token] += 1  # Increment single entry.
        data[key]['\0'] += 1  # Increment running total.
      grams.append(token)
      if len(grams) > size:
        grams.pop(0)
  return data


# Dump the database to a file, so that we can inspect it and use it in other programs.
#with open('data.json', 'w', encoding='utf-8') as f:
#  json.dump(data, f, indent=2, ensure_ascii=False)

def generate_text(data, size):
  text = []  # Start generating text.
  grams = []
  for i in range(256):  # Don't generate more than 256 tokens.
    if len(grams) == 0:
      grams.append('{START}')  # Start with the START token.
    key = tuple(grams)
    data_grams = data.get(key)
    n = random.randint(0, data_grams['\0'])
    for (k, v) in data_grams.items():  # Choose a weighted random entry.
      if k == '\0': continue  # Skip the running total field.
      n = n - v
      if n <= 0:
        text.append(k)
        grams.append(k)
        if k == '{END}':
          return text
        if len(grams) > size:
          grams.pop(0)
        break
  return text

data = build_database(tokenized_tweets, SIZE - 1)

fake_tweets = []
for i in range(len(tweets)):
  text = generate_text(data, SIZE - 1)
  fake_tweets.append(detokenize(text))

#print(detokenize(text))

json.dump(fake_tweets, open(OUTPUT_FILE, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

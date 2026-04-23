import json
import random
import re

from regex import WORD


tweets = json.load(open('tweets.json', 'r', encoding='utf-8'))

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


data = {}  # Grow the database on the training text.
for tweet in tokenized_tweets:
  prev_token = None
  for token in tweet:
    if prev_token != None:
      if prev_token not in data:
        data[prev_token] = {'\0': 0}  # Initialize with zero total.
      if token not in data[prev_token]:
        data[prev_token][token] = 0
      data[prev_token][token] += 1  # Increment single entry.
      data[prev_token]['\0'] += 1  # Increment running total.
    prev_token = token

# Dump the database to a file, so that we can inspect it and use it in other programs.
#with open('data.json', 'w', encoding='utf-8') as f:
#  json.dump(data, f, indent=2, ensure_ascii=False)

text = []  # Start generating text.
c = None
for i in range(1000):
  if c not in data:  # First iteration, or if the last entry is unique.
    c = '{START}'  # Start with the START token.
  n = random.randint(0, data[c]['\0'])
  for (k, v) in data[c].items():  # Choose a weighted random entry.
    if k == '\0': continue  # Skip the running total field.
    n = n - v
    if n <= 0:
      text.append(k)
      c = k
      break

print(detokenize(text))

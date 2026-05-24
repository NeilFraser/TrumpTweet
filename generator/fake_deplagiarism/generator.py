import json
import random
import re
import SequenceFinder

# Creates a Markov chain database of token frequencies from the input tweets,
# and then uses it to generate new fake tweets.
# The n-gram size can be adjusted by changing the SIZE constant.

SIZE = 4  # The n-gram size to use for the database.
INPUT_FILE = '../tokenized_tweets.json'
OUTPUT_FILE = 'fake_tweets_%d.json' % SIZE


# Reassemble tokens back into a tweet.
# "{START}", "Make", "{ }", "America", "{ }", "great", "{!}", "{END}" ->
# "Make America great!"
def detokenize(tokens):
  # Remove the special tokens.
  tokens = [token for token in tokens if token not in ['{START}', '{END}']]
  # Join the tokens together.
  tweet = ''.join(tokens)
  # Replace the special tokens with their original values.
  tweet = re.sub(r'\{URL\}', randomUrl(), tweet)
  tweet = re.sub(r'\{CR\}', '\n', tweet)
  tweet = re.sub(r'\{([ \.!?,;:“”&]|""")\}', r'\1', tweet)
  return tweet


# Create a plausible (but fake) Twitter URL.
# E.g. https://t.co/3fs1oPVnAx
def randomUrl():
  return 'https://t.co/' + ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=10))


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


tokenized_tweets = json.load(open(INPUT_FILE, 'r', encoding='utf-8'))

print("Building database...")
data = build_database(tokenized_tweets, SIZE - 1)

print("Populating plagiarism finder...")
sequence_finder = SequenceFinder.SequenceLCSFinder(tokenized_tweets)

print("Generating %d fake tweets..." % len(tokenized_tweets))
fake_tweets = []
good_count = 0
bad_count = 0
while len(fake_tweets) < len(tokenized_tweets):
  tokens = generate_text(data, SIZE - 1)
  commonality = sequence_finder.find_longest_common_substring(tokens)
  if commonality['length'] < len(tokens) * 0.33:
    if len(fake_tweets) % 1000 == 0:
      print("...%d..." % len(fake_tweets))
    text = detokenize(tokens)
    fake_tweets.append(text)
    good_count += 1
  else:
    bad_count += 1

print("Percentage of rejected plagiarized tweets: %.2f%%" % (bad_count / (good_count + bad_count) * 100))

print("Saving fake tweets to %s..." % OUTPUT_FILE)
json.dump(fake_tweets, open(OUTPUT_FILE, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

print("Done.")

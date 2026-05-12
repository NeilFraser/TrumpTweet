from difflib import SequenceMatcher
import json
import re


# Compare the fake tweets with the real tweets by looking for the longest common substring.
# This takes a wile to run, so we only sample a subset of the fake tweets.
# The result shows a linear relationship between the n-gram size and the amount of plagiarism.

SAMPLE_SIZE = 100  # Number of fake tweets to sample for comparison.

def plagiarism(size):
  print ("Comparing fake tweets with real tweets using n-gram size %d..." % size)
  FAKE_FILE = 'fake_tweets_%d.json' % size
  REAL_FILE = 'tweets.json'
  fake_tweets = json.load(open(FAKE_FILE, 'r', encoding='utf-8'))
  real_tweets = json.load(open(REAL_FILE, 'r', encoding='utf-8'))

  # Convert all URLs to a common token.
  for i in range(len(fake_tweets)):
    fake_tweets[i] = re.sub(r'https?://t\.co/\w+', '{URL}', fake_tweets[i])
  for i in range(len(real_tweets)):
    real_tweets[i] = re.sub(r'https?://t\.co/\w+', '{URL}', real_tweets[i])

  total = 0
  buckets = [0] * 10
  for fake_tweet in fake_tweets[:SAMPLE_SIZE]:
    longest = 0
    for real_tweet in real_tweets:
      match = SequenceMatcher(None, fake_tweet, real_tweet).find_longest_match()
      if match.size > longest:
        longest = match.size
        if longest == len(fake_tweet):
          break  # Found a perfect match.  Stop checking.
    percentage = longest / len(fake_tweet)
    total += percentage
    # Increment the appropriate bucket
    bucket_index = int(percentage * 10)
    if bucket_index >= 10:
      bucket_index = 9
    buckets[bucket_index] += 1

  print("Average longest common substring length as percentage of fake tweet length: %.2f%%" % (total / SAMPLE_SIZE * 100))
  print("Distribution of longest common substrings:")
  print("Range,Count")
  for i, count in enumerate(buckets):
    print("%d-%d%%,%d" % (i * 10, (i + 1) * 10 - 1, count))


for size in range(3, 12):
  plagiarism(size)

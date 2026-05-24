from difflib import SequenceMatcher
import json
import re
import time

# Compare real tweets against each other to get a baseline for how much
# commonality there is between real tweets.  This is not plagiarism,
# but it gives us a baseline for how much commonality we might expect to see.


SAMPLE_SIZE = 100  # Number of real tweets to sample for comparison.

REAL_FILE = 'tweets.json'
real_tweets = json.load(open(REAL_FILE, 'r', encoding='utf-8'))

# Convert all URLs to a common token.
for i in range(len(real_tweets)):
  real_tweets[i] = re.sub(r'https?://t\.co/\w+', '{URL}', real_tweets[i])

test_tweets = real_tweets[:SAMPLE_SIZE]
corpus_tweets = real_tweets[SAMPLE_SIZE:]

start = time.time()
total = 0
count = 0
for test_tweet in test_tweets:
  longest = 0
  for corpus_tweet in corpus_tweets:
    match = SequenceMatcher(None, test_tweet, corpus_tweet).find_longest_match()
    if match.size > longest:
      longest = match.size
      if longest == len(test_tweet):
        break  # Found a perfect match.  Stop checking.
  total += longest / len(test_tweet)
  count += 1
  print("%d Tweet result: %.2f%%" % (count, longest / len(test_tweet) * 100))
elapsed = time.time() - start
print("Elapsed time: %f" % elapsed)

print("Average longest common substring length as percentage of test tweet length: %.2f%%" % (total / SAMPLE_SIZE * 100))

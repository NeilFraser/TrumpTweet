import json
import re

# Tokenize the tweets into parts.

# Split a tweet into token strings.
# "Make America great!" ->
# "{START}", "Make", "{ }", "America", "{ }", "great", "{!}", "{END}"
def tokenize(tweet):
  # Replace URLs with a special token.
  tweet = re.sub(r'https?://t\.co/\w+', '{URL}', tweet)
  tweet = re.sub(r'"""', '{"""}', tweet)
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


REAL_FILE = 'tweets.json'
TOKENIZED_FILE = 'tokenized_tweets.json'

print("Reading tweets...")
real_tweets = json.load(open(REAL_FILE, 'r', encoding='utf-8'))

print("Tokenizing %d tweets..." % len(real_tweets))
tokenized_tweets = [tokenize(tweet) for tweet in real_tweets]

print("Saving tokenized tweets...")
json.dump(tokenized_tweets, open(TOKENIZED_FILE, 'w', encoding='utf-8'), ensure_ascii=False)

print("Done.")

import re

DATASET = 'tweets.csv'

# from_CSV
# Takes one line of CSV code, and returns an array of elements.
# e.g. from_CSV('ab,"cd","e,f","h""i"') -> ['ab', 'cd', 'e,f', 'h"i']
def from_CSV(line):
    line = line.rstrip("\n\r")
    row = []
    quotes = 0
    value = ''
    tokens = tokenise(line, ',"')
    for token in tokens:
        if token == ',':
            if quotes == 0 or quotes == 2:
                row.append(value)
                value = ''
                quotes = 0
            elif quotes == 1:
                value = value + token
        elif token == '"':
            if quotes == 0:
                if value:
                    # Quotes in an unquoted value...
                    # This is an illegal CSV line; feel free to raise an error here.
                    if value.strip() == '':
                        # Let's be nice, and strip leading spaces before a quoted string.
                        value = ''
                        quotes = 1
                    else:
                        # Let's be nice, and quietly add the rogue quote to the value.
                        value = value + token
                else:
                    quotes = 1
            elif quotes == 1:
                quotes = 2
            elif quotes == 2:
                value = value + token
                quotes = 1
        else:
            value = value + token
    if quotes == 1:
        # Quotes didn't end.  The line must continue.
        # Signal that the next line needs to be appended and reprocessed.
        return None
    row.append(value)
    return row


# tokenise
# Break the first string into pieces based on the chars of the second string.
# Returns a list of the words and the dividing chars.
# e.g. tokenise('Hello World', ' o') -> ['Hell', 'o', ' ', 'W', 'o', 'rld']
def tokenise(raw, chars):
    tokens = []
    while raw != '':
        firstIndex = len(raw) + 1
        for x in range(len(chars)):
            thisIndex = raw.find(chars[x])
            if 0 <= thisIndex and thisIndex < firstIndex:
                firstIndex = thisIndex
        if firstIndex > len(raw):
            tokens.append(raw)
            raw = ''
        elif firstIndex == 0:
            tokens.append(raw[0])
            raw = raw[1:]
        else:
            tokens.append(raw[:firstIndex])
            tokens.append(raw[firstIndex])
            raw = raw[firstIndex+1:]
    return tokens

# Read the entire dataset from disk.
with open(DATASET, 'r', encoding='utf-8') as f:
    text_data = f.readlines()

# Parse as CSV.
tweets = []
line = ''
for i in range(len(text_data)):
    line = line + text_data[i]
    tweet = from_CSV(line)
    if tweet is not None:
        tweets.append(tweet)
        line = ''

header = tweets[0]
tweets = tweets[1:]

print("Number of tweets in dataset: %d" % len(tweets))

# Extract only the text of tweets that aren't retweets or just URLs.
tweet_texts = []
text_index = header.index('text')
isRetweet_index = header.index('isRetweet')
for i in range(len(tweets)):
    if tweets[i][isRetweet_index] == 'f':
        if not re.match(r'^\s*https?://t\.co/\w+\s*$', tweets[i][text_index]):
            # Decode HTML entities.
            # Sometimes the dataset has &amp; and sometimes &amp, and so on.
            # Sometimes there's an unencoded &.
            # This is a bit hacky, but it seems to work for the dataset.
            line = tweets[i][text_index]
            line = line.replace('&lt;', '<').replace('&lt', '<')
            line = line.replace('&gt;', '>').replace('&gt', '>')
            line = line.replace('&amp;', '&').replace('&amp', '&')
            tweet_texts.append(line)
print("Number of tweets after filtering out retweets and just URLs: %d" % len(tweet_texts))

print("First 20 tweets:")
for i in range(20):
    print("%d: %s" % (i+1, tweet_texts[i]))

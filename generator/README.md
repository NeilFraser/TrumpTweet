# Tweet Generator

## tweets.csv

This is the raw datafile with 56,571 real tweets from Trump.
Data source: https://www.kaggle.com/datasets/codebreaker619/donald-trump-tweets-dataset

## tweets.py

The raw datafile contains retweets (tweets written by other people and simply
reshared by Trump).  These are not actually written by Trump.  There are also
tweets from Trump that are just URLs.  Both of these types of tweets need to be
discarded.  Additionally, data cleaning is required to handle inconsistent
encoding of special characters.  This python script parses the tweets.csv,
filters the tweets, cleans the data, and saves the results to tweets.json.

## tweets.json

This is a cleaned dataset with 45,454 real tweets from Trump.

## commonality.py

This script takes a sample of the first 100 tweets and compares each one to the
remaining 45,345 tweets, computing the longest common substring.  This determines
that the average tweet from Trump shares 36% of its content with another of
his tweets.  Checking one tweet against the whole dataset takes 8 seconds.

## tokenizer.py

This script reads tweets.json, tokenizes it, saves the result as
tokenized_tweets.json.

## tokenized_tweets.json

This is a list of all tweets, tokenized.  For example one tweet might be:
["{START}", "Make", "{ }", "America", "{ }", "Great", "{!}", "{END}"]

## fake

This directory has a file `generator.py` which creates 45k fake tweets with
a specified n-gram length, and saves it to a JSON file.

The script `plagiarism.py` computes how original the fake tweets of each n-gram
length are, by finding the longest common substring between each fake tweet and
every original tweet.

## fake_deplagiarism

This directory has a file `generator.py` which creates 45k fake tweets with
a specified n-gram length, and saves it to a JSON file.  The difference is that
each fake tweet is checked for plagiarism, and discarded if it matches a real
tweet by more than 33%.

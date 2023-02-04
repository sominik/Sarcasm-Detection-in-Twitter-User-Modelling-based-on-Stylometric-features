import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.stem.snowball import SnowballStemmer

import csv
import ast

urls = ['http', 'https', 'ftp']


def extract_tweets_text(csv_path, text_index):
    tweets_text = []
    with open(csv_path, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if lines[text_index] != '':
                tweets_text.append(lines[text_index])
    del tweets_text[0]  # bc it is the header
    return tweets_text


def normalization(tweet):
    words = word_tokenize(tweet)
    # delete one-letter words, numbers, punctuations, tags and urls and make them lower case
    words = [word.lower() for word in words if word not in urls and word.isalpha() and len(word) > 1]
    return words


def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in words if w not in stop_words]
    return filtered_sentence


def spellchecking(tweet):
    result = []
    check = Speller(lang='en')
    for word in tweet:
        result.append(check(word))
    return result


def stemming(tweet):
    snow_stemmer = SnowballStemmer(language='english')
    try:
        stem_words = []
        for w in tweet:
            x = snow_stemmer.stem(w)
            stem_words.append(x)
    except Exception as e:
        print(e)
        return tweet
    return stem_words


def clean_tweets(tweets):
    clean_text_tweets = []
    normal_tweets = []
    spell_tweets = []
    no_stop_word_tweets = []
    stem_tweets = []
    index = 0
    for tweet in tweets:
        print(tweet)
        print(index)
        normal_tweets.append(normalization(tweet))
        spell_tweets.append(spellchecking(normal_tweets[index]))
        no_stop_word_tweets.append(remove_stop_words(spell_tweets[index]))
        stem_tweets.append(stemming(no_stop_word_tweets[index]))
        clean_text_tweets.append(stem_tweets[index])
        index += 1
    return clean_text_tweets


def pre_process_tweets(file_path):
    file = pd.read_csv(file_path)
    text_tweets = file['tweet_text']
    clean_text_tweets = clean_tweets(text_tweets)
    file["clean_tweet_text"] = clean_text_tweets
    return file


sarcastic_users_tweets_clean = pre_process_tweets('../csv_files/SPIRS-sarcastic-users-tweets.csv')
sarcastic_users_tweets_clean.to_csv('csv_files/SPIRS-sarcastic-users-tweets_clean.csv')


non_sarcastic_users_tweets_clean = pre_process_tweets('../csv_files/SPIRS-non-sarcastic-users-tweets.csv')
non_sarcastic_users_tweets_clean.to_csv('csv_files/SPIRS-non_sarcastic-users-tweets_clean.csv')


# #####################################################################################  clean files:


# file1 = pd.read_csv('csv_files/SPIRS-sarcastic-users-recent-tweets.csv')
# file2 = pd.read_csv('csv_files/SPIRS-sarcastic-users-recent-tweets-2.csv')
# del file1['Unnamed: 0']
# del file2['Unnamed: 0']
#
# merge_file = file1.append(file2)
#
# merge_file.to_csv('csv_files/SPIRS-sarcastic-users-recent-tweets-.csv')

# dataset = pd.read_csv('csv_files/SPIRS-non-sarcastic-users-recent-tweets.csv')
# del dataset['Unnamed: 0']
# del dataset['name']
# del dataset['username']
# dataset['label'] = 'non_sarcastic'
# del file1['Unnamed: 0.1']
# dataset.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
#
# for row in dataset:
#  print(row)
#
# dataset.to_csv('csv_files/non-sarcastic-users-recent-tweets.csv', index=False)

# file1['label'] = 'non_sarcastic'
# del file1['Unnamed: 0.1']
# file1.rename(columns={'Unnamed: 0': 'index'}, inplace=Tru
# file1.to_csv('csv_files/non_sarcastic.csv', index=False)

# a = pd.read_csv('csv_files/a.csv')
# b = pd.read_csv('csv_files/b.csv')
#
# o = a.append(b)
# print(o)

# Unnamed: 0.2
# Unnamed: 0.1
# Unnamed: 0

# file1 = pd.read_csv('csv_files/tweets_dataset')

# del file1['Unnamed: 0.1']
# del file1['Unnamed: 0.2']

# for row in dataset:
#  print(row)
#
# file1.rename(columns = {'Unnamed: 0':'index'}, inplace = True)
#
# file1.to_csv('csv_files/sarcastic.csv', index=False)

# #############################################################################################


def clean_tweets_array(tweets_array):
    clean_text_tweets = []
    normal_tweets = []
    spell_tweets = []
    no_stop_word_tweets = []
    stem_tweets = []
    index = 0
    for tweet in tweets_array:
        normal_tweets.append(normalization(tweet))
        spell_tweets.append(spellchecking(normal_tweets[index]))
        no_stop_word_tweets.append(remove_stop_words(spell_tweets[index]))
        stem_tweets.append(stemming(no_stop_word_tweets[index]))
        clean_text_tweets.append(stem_tweets[index])
        index += 1
    return clean_text_tweets


def pre_process_tweets_arrays(file_path):
    file = pd.read_csv(file_path)
    text_tweets = file['tweets_text']
    clean_tweets_arrays = []
    index = 0
    for tweets_array_str in text_tweets:
        print(index)
        tweets_array = ast.literal_eval(tweets_array_str)
        clean_tweets_arrays.append(clean_tweets_array(tweets_array))
        index += 1
    file["clean_tweet_text"] = clean_tweets_arrays
    return file


file1 = pre_process_tweets_arrays('../csv_files/sarcastic-users-recent-tweets.csv')
file1.to_csv('csv_files/sarcastic-users-recent-tweets-clean.csv')

file2 = pre_process_tweets_arrays('../csv_files/non-sarcastic-users-recent-tweets.csv')
file2.to_csv('csv_files/non-sarcastic-users-recent-tweets-clean.csv')




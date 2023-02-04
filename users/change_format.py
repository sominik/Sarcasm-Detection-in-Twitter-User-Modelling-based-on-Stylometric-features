import numpy as np
import pandas as pd
import csv


def reshape(csv_path, destination_csv_file):
    recent_tweets_list = {'user_id': [], 'recent_tweets': []}
    with open(csv_path, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            recent_tweets_list['user_id'].append('user_id:' + lines[1])
            reshaped_tweets = lines[8].replace('[], ', '').replace('], [', ' <END> ').replace('[[', '', 1).replace(']]', '', 1)\
                .replace(',', '').replace('\'', '')
            recent_tweets_list['recent_tweets'].append(reshaped_tweets)
        del recent_tweets_list['user_id'][0]  # bc of headers
        del recent_tweets_list['recent_tweets'][0]  # bc of headers
    recent_tweets_list_df = pd.DataFrame(recent_tweets_list)
    print(recent_tweets_list_df)
    recent_tweets_list_df.to_csv(destination_csv_file, index=False, quoting=csv.QUOTE_ALL, header=False)


def merge2file(file1, file2):
    file1_df = pd.read_csv(file1)
    file2_df = pd.read_csv(file2)
    final_df = file1_df.append(file2_df)
    final_df.to_csv('csv_files/all-users-recent-tweets-clean-reshaped.csv', index=False, quoting=csv.QUOTE_ALL,
                    header=False)
    print(final_df)


# reshape('csv_files/non-sarcastic-users-recent-tweets-clean.csv',
#         'csv_files/non-sarcastic-users-recent-tweets-clean-reshaped.csv')

reshape('../csv_files/sarcastic-users-recent-tweets-clean.csv',
        'csv_files/sarcastic-users-recent-tweets-clean-reshaped-quoting.csv')


# merge2file('csv_files/sarcastic-users-recent-tweets-clean-reshaped.csv',
#            'csv_files/non-sarcastic-users-recent-tweets-clean-reshaped.csv')



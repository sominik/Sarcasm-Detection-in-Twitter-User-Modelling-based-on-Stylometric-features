import numpy as np
import pandas as pd
import csv
import random


def cleanFiles(tweets_users_file, recent_tweets_file):
    tweets_users_data = np.asarray(pd.read_csv(tweets_users_file, header=None))
    recent_tweets_data = np.asarray(pd.read_csv(recent_tweets_file, header=None))

    print("tweets counts before cleaning: ", len(tweets_users_data))
    print("users  counts before cleaning: ", len(recent_tweets_data))

    not_found_user_indices = []
    for i in range(len(tweets_users_data)):
        line = tweets_users_data[i]
        user_id = "user_id:" + str(line[3])
        found_indices = np.where(recent_tweets_data[:, 0] == user_id)

        if len(found_indices[0]) == 0:
            not_found_user_indices.append(i)
        else:
            found_in_self_indices = np.where(tweets_users_data[:, 3] == line[3])

            if len(found_in_self_indices[0]) != len(found_indices[0]) \
                    and len(found_in_self_indices[0]) > 0 \
                    and len(found_indices[0]) > 0:
                if len(found_in_self_indices[0]) > len(found_indices[0]):
                    index = found_indices[0][0]
                    recent_tweets_data = np.append(recent_tweets_data, [recent_tweets_data[index]], axis=0)
                else:
                    index = found_indices[0][1]
                    recent_tweets_data = np.delete(recent_tweets_data, [index], axis=0)
    print("not found users counts: ", len(not_found_user_indices))
    tweets_users_data = np.delete(tweets_users_data, not_found_user_indices, axis=0)
    print("tweets counts after cleaning: ", len(tweets_users_data))

    print("users  counts after cleaning 1: ", len(recent_tweets_data))

    not_found_tweet_indices = []
    for i in range(len(recent_tweets_data)):
        line = recent_tweets_data[i]
        user_id = int(line[0].split('user_id:')[1])
        found_indices = np.where(tweets_users_data[:, 3] == user_id)
        if len(found_indices[0]) == 0:
            not_found_tweet_indices.append(i)

    print("not found tweets counts: ", len(not_found_tweet_indices))
    recent_tweets_data = np.delete(recent_tweets_data, not_found_tweet_indices, axis=0)
    print("users  counts after cleaning 2: ", len(recent_tweets_data))
    tweets_users_data = np.delete(tweets_users_data, 0, axis=1)
    print(len(tweets_users_data))
    pd.DataFrame(tweets_users_data).to_csv("../csv_files/clean_datasets/tweets_dataset.csv", index=False, header=False)
    pd.DataFrame(recent_tweets_data).to_csv("../csv_files/clean_datasets/users_recent_tweets_dataset.csv", index=False,
                                            quoting=csv.QUOTE_ALL, header=False)


# tweets_users_file = "./csv_files/tweets_users_dataset.csv"
# recent_tweets_file = './csv_files/recent-tweets-clean-reshaped.csv'
# cleanFiles(tweets_users_file, recent_tweets_file)

def split_tweets_with_label(tweets_dataset_array):
    sarcastic_tweets = []
    non_sarcastic_tweets = []
    for line in tweets_dataset_array:
        if line[6] == 'sarcastic':
            sarcastic_tweets.append(line)
        elif line[6] == 'non_sarcastic':
            non_sarcastic_tweets.append(line)
    sarcastic_tweets_np_arr = np.asarray(sarcastic_tweets)
    non_sarcastic_tweets_np_arr = np.asarray(non_sarcastic_tweets)
    print("sarcastic tweets counts: ", len(sarcastic_tweets))
    print("non sarcastic tweets counts: ", len(non_sarcastic_tweets))
    return sarcastic_tweets_np_arr, non_sarcastic_tweets_np_arr


def create_train_test_tweets(tweets_dataset):
    tweets_dataset_array = np.asarray(pd.read_csv(tweets_dataset, header=None))
    print("tweets_dataset_array: ", len(tweets_dataset_array))
    sarcastic_tweets, non_sarcastic_tweets = split_tweets_with_label(tweets_dataset_array)
    number_of_sarcastic_rows = sarcastic_tweets.shape[0]
    sarcastic_indices = np.array(range(number_of_sarcastic_rows))
    number_of_non_sarcastic_rows = non_sarcastic_tweets.shape[0]
    non_sarcastic_indices = np.array(range(number_of_non_sarcastic_rows))

    # create Test tweets dataset:
    test_sarcastic_indices = np.random.choice(number_of_sarcastic_rows,
                                         size=1376, replace=False)
    test_non_sarcastic_indices = np.random.choice(number_of_non_sarcastic_rows,
                                             size=2019, replace=False)
    test_tweets_sarcastic = sarcastic_tweets[test_sarcastic_indices, :]
    test_tweets_non_sarcastic = non_sarcastic_tweets[test_non_sarcastic_indices, :]
    test_tweets = np.concatenate([test_tweets_sarcastic, test_tweets_non_sarcastic])
    print("test dataset size: ", len(test_tweets))

    # create Train tweets dataset:
    train_sarcastic_indices = np.setdiff1d(sarcastic_indices, test_sarcastic_indices)
    train_non_sarcastic_indices = np.setdiff1d(non_sarcastic_indices, test_non_sarcastic_indices)
    train_tweets_sarcastic = sarcastic_tweets[train_sarcastic_indices, :]
    train_tweets_non_sarcastic = non_sarcastic_tweets[train_non_sarcastic_indices, :]
    train_tweets =np.concatenate([train_tweets_sarcastic, train_tweets_non_sarcastic])
    print("train dataset size: ", len(train_tweets))
    pd.DataFrame(test_tweets).to_csv("../csv_files/clean_datasets/test_tweets_dataset.csv", index=False, header=False)
    pd.DataFrame(train_tweets).to_csv("../csv_files/clean_datasets/train_tweets_dataset.csv", index=False,
                                      header=False)


# create_train_test_tweets("./csv_files/clean_datasets/tweets_dataset.csv")
# test_tweets_dataset_array = np.asarray(pd.read_csv("./csv_files/clean_datasets/test_tweets_dataset.csv", header=None))
# train_tweets_dataset_array = np.asarray(pd.read_csv("./csv_files/clean_datasets/train_tweets_dataset.csv", header=None))
# print("/////////////////// TEST: ")
# split_tweets_with_label(test_tweets_dataset_array)
# print("/////////////////// TRAIN: ")
# split_tweets_with_label(train_tweets_dataset_array)

def create_train_test_users(users_recent_tweets_dataset, train_tweets_dataset,
                            test_tweets_dataset):
    users_recent_tweets_array = np.asarray(pd.read_csv(users_recent_tweets_dataset, header=None))
    train_tweets_array = np.asarray(pd.read_csv(train_tweets_dataset, header=None))
    test_tweets_array = np.asarray(pd.read_csv(test_tweets_dataset, header=None))

    print("train_tweets_array length: ", len(train_tweets_array))
    print("test_tweets_array length: ", len(test_tweets_array))

    train_recent_tweets_array = []
    test_recent_tweets_array = []

    # for line in users_recent_tweets_array:
    #     user_id = int(line[0].split("user_id:")[1])
    #     train_index_list = np.where(train_tweets_array[:, 2] == user_id)
    #     if len(train_index_list[0]) > 0:
    #         train_recent_tweets_array.append(line)
    #     test_index_list = np.where(test_tweets_array[:, 2] == user_id)
    #     if len(test_index_list[0]) > 0:
    #         test_recent_tweets_array.append(line)

    for line in train_tweets_array:
        user_id = "user_id:" + str(line[2])
        train_index_list = np.where(users_recent_tweets_array[:, 0] == user_id)
        if len(train_index_list[0]) > 0:
            index = train_index_list[0][0]
            train_recent_tweets_array.append(users_recent_tweets_array[index, :])

    for line in test_tweets_array:
        user_id = "user_id:" + str(line[2])
        test_index_list = np.where(users_recent_tweets_array[:, 0] == user_id)
        if len(test_index_list[0]) > 0:
            index = test_index_list[0][0]
            test_recent_tweets_array.append(users_recent_tweets_array[index, :])

    print("recent_tweets_train length:", len(train_recent_tweets_array))
    print("recent_tweets_test length:", len(test_recent_tweets_array))

    train_recent_tweets_array = np.asarray(train_recent_tweets_array)
    test_recent_tweets_array = np.asarray(test_recent_tweets_array)
    pd.DataFrame(train_recent_tweets_array).to_csv("../csv_files/clean_datasets/train_users_recent_tweets.csv", index=False, quoting=csv.QUOTE_ALL, header=False)
    pd.DataFrame(test_recent_tweets_array).to_csv("../csv_files/clean_datasets/test_users_recent_tweets.csv", index=False, quoting=csv.QUOTE_ALL, header=False)


create_train_test_users("./csv_files/clean_datasets/users_recent_tweets_dataset.csv",
                        "./csv_files/clean_datasets/train_tweets_dataset.csv",
                        "./csv_files/clean_datasets/test_tweets_dataset.csv")

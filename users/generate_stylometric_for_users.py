#!/usr/bin/env python
import pandas as pd
import numpy as np
import csv
import gensim, os

doc2vec = gensim.models.Doc2Vec.load('./models/users_stylometric.model')
data = np.asarray(pd.read_csv("../csv_files/clean_datasets/users_recent_tweets_dataset.csv", header=None))
DIM = 300
VECTOR_SIZE = 200

directory = "../user_embeddings"
if not os.path.exists(directory):
	os.makedirs(directory)
file = open(directory+"/users_stylometric.csv", 'w')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)

# Inferring paragraphVec vectors for each user
doc2vec_vectors = []
for i in range(data.shape[0]):
	doc2vec_vectors.append(doc2vec.infer_vector([str(data[i][1])]))

print("length of vectors: ", len(doc2vec_vectors))
vectors = np.asarray(doc2vec_vectors)

users = data[:, 0]
print("length of users: ", len(users))
print("vectors: ", vectors)

for i in range(len(users)):
	ls = []
	ls.append(users[i])
	v = [0]*VECTOR_SIZE
	for j in range(len(vectors[i])):
		v[j] = vectors[i][j]
	ls.append(v)
	wr.writerow(ls)

# smartsense_assignment_siddhesh


run train.py and pass the arguments as 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='random_forest', help='name of the model to be trained {logistic_regression, random_forest, svm, knn}')
parser.add_argument('--data_path', type=str, default='./data/deceptive-opinion.csv', help='path to the data')
parser.add_argument('--embedding_method', type=str, default='word2vec', help='method to be used to create the embeddings {word2vec, glove, tfidf}')

args = parser.parse_args()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

import 

# In[0]: get the arugments from the command line
# 1. model_name: name of the model to be trained {logistic_regression, random_forest, svm, knn}
# 2. data_path: path to the data
# 3. embedding_method: method to be used to create the embeddings {word2vec, glove, tfidf}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='random_forest', help='name of the model to be trained {logistic_regression, random_forest, svm, knn}')
parser.add_argument('--data_path', type=str, default='./data/deceptive-opinion.csv', help='path to the data')
parser.add_argument('--embedding_method', type=str, default='word2vec', help='method to be used to create the embeddings {word2vec, glove, tfidf}')

args = parser.parse_args()

model_name = args.model_name
data_path = args.data_path
embedding_method = args.embedding_method

assert model_name in ['logistic_regression', 'random_forest', 'svm', 'knn'], "Invalid model name"
assert embedding_method in ['word2vec', 'glove', 'tfidf'], "Invalid embedding method"

if not os.path.exists(data_path):
    print("The data path you provided does not exist")
    raise Exception("Invalid data path")


#In[1]: Data processing - tokenization and lemmatization

data = pd.read_csv('./data/deceptive-opinion.csv')
# drop the columns that are not required
data.drop(['hotel', 'polarity', 'source'], axis=1, inplace=True)
data['deceptive']   = data['deceptive'].map({'truthful': 0, 'deceptive': 1})

'''
# Tokenize the data 

1. split the data to get the words
2. remove the stop words
3. remove the punctuation
4. remove the numbers
5. lammatize the words (preferring lemmatization over stemming as it gives the root word)

'''

lemmatizer = WordNetLemmatizer() # lemmatizing the words
def apply_tokenization(text):

    text = text.lower() # converting the text to lower case

    words = nltk.word_tokenize(text)   # splitting the text into words

    stop_words = set(stopwords.words('english')) # getting the stop words in english

    tokens = [word for word in words if word.isalpha()] # removing the punctuations from the text or getting only alpha-words

    non_stop_words = [word for word in tokens if not word in stop_words] # removing the stop words from the text 

    lemmatized_words = [lemmatizer.lemmatize(word) for word in non_stop_words] # lemmatizing the words

    return " ".join(lemmatized_words) # joining the words to form a sentence


data['tokenized_data'] = data['text'].apply(apply_tokenization) # applying the function to the text column


# In[2]: Creating embeddings

# get the embedding for the sentence
def get_embedding(sentence):
    # get the tokens from the sentence
    tokens = nltk.word_tokenize(sentence)
    # get the embedding for each word in the sentence
    embeddings = [model.wv[word] for word in tokens]
    # get the mean of the embeddings
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


#_____________________________ Glove _________________________________
if embedding_method.lower().startswith('tf'):
    # create embeddings for the data using the tfidf model from sklearn

    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
    # create the tfidf model
    tfidf = TfidfVectorizer()
    y = data['deceptive']
    x = data['tokenized_data']
    # fit the model on the tokenized data
    embeddings = tfidf.fit_transform(x)
    # get the vocabulary of the model
    words = tfidf.get_feature_names_out()
    print("Vocab size", len(words))
    # save the model
    pickle.dump(tfidf, open('./model/tfidf.model', 'wb'))
    # add a column to the data to store the embeddings
    data['text_embedding'] = None
    for i in range(embeddings.shape[0]):
        data['text_embedding'][i] = embeddings[i].toarray()[0]
    
    # data.drop(['text', 'tokenized_data'], axis=1, inplace=True)

#_____________________________ Glove _________________________________
if embedding_method.lower() == 'glove':
    # create embeddings for the data using the glove model from gensim

    # current there is no implementation of glove as We need to download the glove embedding vectors
    # which is be too time comsuming and complext for this assignment-3
    print("Glove is not implemented yet")
    print("Instead using word2vec")
    embedding_method = 'word2vec'

#_____________________________ Word2Vec _________________________________
if embedding_method.lower() == 'word2vec':
    # create embeddings for the data using the word2vec model from gensim

    from gensim.models import Word2Vec

    # create a list of sentences from the tokenized data
    sentences = [nltk.word_tokenize(sentence) for sentence in data['tokenized_data']]
    # create the word2vec model
    model = Word2Vec(sentences, min_count=1, vector_size=100, sg=0)
    # get the vocabulary of the model
    words = model.wv.key_to_index.keys()
    # save the model
    model.save('./model/word2vec.model')
    print("Vocab size", len(words))

    data['text_embedding'] = data['tokenized_data'].apply(get_embedding)
    data.drop(['text', 'tokenized_data'], axis=1, inplace=True)



def get_scores(y, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print("Accuracy: ", accuracy_score(y, y_pred))
    print("Precision: ", precision_score(y, y_pred))
    print("Recall: ", recall_score(y, y_pred))
    print("F1: ", f1_score(y, y_pred))


# ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

# plot the confusion matrix

def plot_cm(y_test, y_pred):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
    disp.plot()
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
    plt.show()



# split the data into train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)
print(train.shape, test.shape)

# get the X_train, y_train, X_test, y_test
X_train = np.array(train['text_embedding'].tolist())
y_train = np.array(train['deceptive'].tolist())

X_test = np.array(test['text_embedding'].tolist())
y_test = np.array(test['deceptive'].tolist())

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)




from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score







from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier




if model_name  == 'logistic_regression':
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression()

    lr_model.fit(X_train, y_train)

    # get the predictions

    y_pred = lr_model.predict(X_test)

    # get the scores
    print("----------- Train Scores ------------")
    get_scores(y_train, lr_model.predict(X_train))
    print("----------- Test Scores ------------")
    get_scores(y_test, y_pred)


    plot_cm(y_test, y_pred) 

if model_name == 'random_forest':
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier()

    rf_model.fit(X_train, y_train)

    # get the predictions

    y_pred = rf_model.predict(X_test)

    # get the scores
    print("----------- Train Scores ------------")
    get_scores(y_train, rf_model.predict(X_train))
    print("----------- Test Scores ------------")
    get_scores(y_test, y_pred)

    plot_cm(y_test, y_pred)

if model_name == 'svm':
    from sklearn.svm import SVC

    svm_model = SVC()

    svm_model.fit(X_train, y_train)

    # get the predictions

    y_pred = svm_model.predict(X_test)

    # get the scores
    print("----------- Train Scores ------------")
    get_scores(y_train, svm_model.predict(X_train))
    print("----------- Test Scores ------------")
    get_scores(y_test, y_pred)

    plot_cm(y_test, y_pred)

if model_name == 'knn':
    from sklearn.neighbors import KNeighborsClassifier

    knn_model = KNeighborsClassifier()

    knn_model.fit(X_train, y_train)

    # get the predictions

    y_pred = knn_model.predict(X_test)

    # get the scores
    print("----------- Train Scores ------------")
    get_scores(y_train, knn_model.predict(X_train))
    print("----------- Test Scores ------------")
    get_scores(y_test, y_pred)

    plot_cm(y_test, y_pred)


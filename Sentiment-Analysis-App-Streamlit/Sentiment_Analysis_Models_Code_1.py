import pandas as pd
# import spacy
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle

amazon_df = pd.read_csv(r"C:\Users\manan\Documents\Sentiment-Analysis-App-Streamlit\Sentiment-Analysis-App-Streamlit\data\amazon_cells_labelled.txt",
                        delimiter='\t', header=None, names=['review', 'sentiment'])

imdb_df = pd.read_csv(r"C:\Users\manan\Documents\Sentiment-Analysis-App-Streamlit\Sentiment-Analysis-App-Streamlit\data\imdb_labelled.txt",
                      delimiter='\t', header=None, names=['review', 'sentiment'])

yelp_df = pd.read_csv(r"C:\Users\manan\Documents\Sentiment-Analysis-App-Streamlit\Sentiment-Analysis-App-Streamlit\data\yelp_labelled.txt", delimiter='\t',
                      header=None, names=['review', 'sentiment'])

data = pd.concat([amazon_df, yelp_df, imdb_df])

data.reset_index(drop='True', inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.3)


# data_review_list = data['review'].to_list()

# nltk.download('stopwords')

# sp = spacy.load('en_core_web_sm')

# all_stopwords = sp.Defaults.stop_words

all_stopwords = ['a', 'about', 'after', 'all', 'also', 'always', 'am', 'an', 'and', 'any', 'are', 'at', 'be', 'been', 'being', 'but', 'by', 'came', 'can', 'cant', 'come',
                 'could', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'else', 'for', 'from', 'get', 'give', 'goes', 'going', 'had', 'happen',
                 'has', 'have', 'having', 'how', 'i', 'if', 'ill', 'im', 'in', 'into', 'is', 'isnt', 'it', 'its', 'ive', 'just', 'keep', 'let', 'like', 'made', 'make',
                 'many', 'may', 'me', 'mean', 'more', 'most', 'much', 'no', 'not', 'now', 'of', 'only', 'or', 'our', 'really', 'say', 'see', 'some', 'something',
                 'take', 'tell', 'than', 'that', 'the', 'their', 'them', 'then', 'they', 'thing', 'this', 'to', 'try', 'up', 'us', 'use', 'used',  'uses', 'very',
                 'want', 'was', 'way', 'we',  'what', 'when', 'where', 'which', 'who', 'why',  'will',  'with',  'without', 'wont', 'you', 'your']


def text_cleaning(message):
    punc_removed = [char for char in message if char not in string.punctuation]
    punc_removed_join = ''.join(punc_removed)
    punc_removed_join_clean = [word.lower() for word in punc_removed_join.split(
    ) if word.lower() not in all_stopwords]
    punc_removed_join_clean = ' '.join(punc_removed_join_clean)
    return punc_removed_join_clean


X_train_clean = []
X_test_clean = []

for review in X_train:
    cleaned_review = text_cleaning(review)
    X_train_clean.append(cleaned_review)

X_train = X_train_clean

vectorizer_train = CountVectorizer(max_features=2000)
data_vectorizer_train = vectorizer_train.fit_transform(X_train)
X_train = data_vectorizer_train

X_train = X_train.toarray()
#X_train.reset_index(inplace=True, drop=True)
Y_train.reset_index(inplace=True, drop=True)

vectorizer_test = CountVectorizer(max_features=2000)
data_vectorizer_test = vectorizer_test.fit_transform(X_test)
X_test = data_vectorizer_test

X_test = X_test.toarray()
#X_test.reset_index(inplace=True, drop=True)
Y_test.reset_index(inplace=True, drop=True)

log_reg = LogisticRegression().fit(X_train, Y_train)

# save
with open('sa_lr_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

Y_predicted_lr = log_reg.predict(X_test)

print(classification_report(Y_test, Y_predicted_lr))

Lsvc = LinearSVC()
Lsvc_reg = Lsvc.fit(X_train, Y_train)

Y_predicted_svc = Lsvc_reg.predict(X_test)

with open('sa_svc_model.pkl', 'wb') as f:
    pickle.dump(Lsvc_reg, f)

print(classification_report(Y_test, Y_predicted_svc))

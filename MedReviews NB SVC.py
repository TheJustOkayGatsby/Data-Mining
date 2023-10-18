import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import scipy

# Reading dataset as dataframe
df = pd.read_csv("MedReviews.csv", encoding = "ISO-8859-1") # You can also use "utf-8"
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window

# Converting structured categorical features to numerical features
df['Rating'] = df['Rating'].map({'High':1, 'Low':0})

# Converting unstructured 'Review' column to a TF-IDF matrix
def cleaner(review): # Cleaning reviews
    soup = BeautifulSoup(review, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", souped) # substituting @mentions, urls, etc with whitespace
    re2 = re.sub("[^A-Za-z]+"," ", re1) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

df['cleaned_review'] = df.Review.apply(cleaner)
df = df[df['cleaned_review'].map(len) > 0] # removing rows with cleaned reviews of length 0
print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
print(df[['Review','cleaned_review']].head())
df['cleaned_review'] = [" ".join(row) for row in df['cleaned_review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_review']
Y = df['Rating'] # target column
tfidf = TfidfVectorizer(min_df=.00086, ngram_range=(1,3)) # min_df=.00086 means that each ngram (unigram, bigram, & trigram) must be present in at least 20 documents for it to be considered as a token (23305*.00086=20). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
print("The created tokens: \n", tfidf.get_feature_names_out())
print("Shape of tfidf matrix: ", data_tfidf.shape)
print(type(data_tfidf))

# Concatenate Tf-Idf matrix with medicine and condition columns
df1 = pd.DataFrame(data_tfidf.toarray(), columns = tfidf.get_feature_names_out()) #Converts Tf-Idf matrix to a dataframe
df2 = df[['Medicine', 'Condition']]
column_list = ['Medicine', 'Condition']
df3 = pd.get_dummies(df2,columns = column_list)
X = pd.concat([df1,df3], axis=1)
print(type(X))
print(X.shape)
print(X.info())

print("Implementing SVC...")
# Implementing Support Vector Classifier
model = LinearSVC() # kernel = 'linear' and C = 1

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
smote = SMOTE(random_state = 101)
for train_index, test_index in kf.split(data_tfidf,Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    X_train,Y_train = smote.fit_resample(X_train,Y_train) # Balancing training data
    model.fit(X_train, Y_train) # Fitting SVC
    Y_pred = model.predict(X_test)
    score = metrics.precision_score(Y_test, Y_pred) # Calculating precision
    print("Cross-validation precison: ", score)
    scores.append(score) # appending cross-validation precision for each iteration
mean_precision = np.mean(scores)
print("Mean cross-validation precision: ", mean_precision)

print("Implementing NBC.....")
# Implementing Naive Bayes Classifier
nbc_clf = MultinomialNB()

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf,Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    X_train,Y_train = smote.fit_resample(X_train,Y_train) # Balancing training data
    nbc_clf.fit(X_train, Y_train) # Fitting NBC
    Y_pred = nbc_clf.predict(X_test)
    score = metrics.precision_score(Y_test, Y_pred) # Calculating precision
    print("Cross-validation precision: ", score)
    scores.append(score) # appending cross-validation precision for each iteration
nbc_mean_precision = np.mean(scores)
print("Mean cross-validation precision: ", nbc_mean_precision)

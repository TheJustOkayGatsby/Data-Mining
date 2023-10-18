import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Reading dataset as dataframe
df = pd.read_csv("News.csv")
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window
df['Category'] = df['Category'].map({'Sport':1, 'Sci/Tech':0})

# Cleaning Summaries
def cleaner(summary):
    soup = BeautifulSoup(summary, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
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

df['cleaned_summary'] = df.Summary.apply(cleaner)
df = df[df['cleaned_summary'].map(len) > 0] # removing rows with cleaned summaries of length 0
print("Printing top 5 rows of dataframe showing original and cleaned summaries....")
print(df[['Summary','cleaned_summary']].head())
df['cleaned_summary'] = [" ".join(row) for row in df['cleaned_summary'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_summary']
Y = df['Category'] # target column
tfidf = TfidfVectorizer(min_df=.0005, ngram_range=(1,3)) # min_df=.0005 means that each ngram (unigram, bigram, & trigram) must be present in at least 30 documents for it to be considered as a token (60000*.0005=30). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
print("Shape of tfidf matrix: ", data_tfidf.shape)

# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(data_tfidf)

category = list(df['Category'])
news = list(df['Summary'])

data_ = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=df['Category'], colorscale='Rainbow', opacity=0.5),
                                text=[f'Category: {a}<br>News: {b}' for a,b in list(zip(category, news))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 1400, height = 1400,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data_, layout=layout)
fig.show()
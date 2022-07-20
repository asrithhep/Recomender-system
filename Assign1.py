import pandas as pd
import re
import csv
import numpy as np
from csv import writer
from csv import reader
import random
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('data_set.csv')
X = df['ArticleID'].values
Y = df['news'].values
Y = Y.astype(str)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def clean_tokenize(document):
	document = re.sub('[^\w_\s-]', ' ',document)
	stop_free = " ".join([i for i in document.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	tokens = nltk.word_tokenize(document)             
	cleaned_article = ' '.join([stemmer.stem(item) for item in tokens])  
	return(cleaned_article)
    
cleaned_articles = list(map(clean_tokenize, Y))
tfidf_matrix = TfidfVectorizer(stop_words='english', min_df=2)
article_tfidf_matrix = tfidf_matrix.fit_transform(cleaned_articles)
#print(article_tfidf_matrix)
#print(pd.DataFrame(article_tfidf_matrix.todense(),columns=tfidf_matrix.get_feature_names()))
print(article_tfidf_matrix.toarray())

kmeans = KMeans(n_clusters=10).fit(article_tfidf_matrix)
#print(pd.Series(kmeans.labels_).value_counts())
y = kmeans.labels_          
y1 = y.reshape(y.shape[0],1)
print(y1)
a = pd.Series(kmeans.labels_).value_counts()

def add_column_in_csv(input_file, output_file, transform_row):
	""" Append a column in existing csv using csv.reader / csv.writer classes"""
	with open(input_file, 'r') as read_obj, \
	open(output_file, 'w', newline='') as write_obj:
		csv_reader = reader(read_obj)
		csv_writer = writer(write_obj)
		for row in csv_reader:
			transform_row(row, csv_reader.line_num)
			csv_writer.writerow(row)
          
z = np.array(["Article rank"]) 
A = np.vstack((z,y1))
add_column_in_csv('data_set.csv', 'output_data.csv', lambda row, line_num: row.append(A[line_num - 1,0]))
data1 = pd.read_csv('output_data.csv')
data1.sort_values("Article rank",axis =0,ascending = True,inplace = True, na_position ='last')
d1 = data1['news'].values
d2 = data1['Article rank'].values
d3 = data1['ArticleID']
for j in range(10):
	list1 = np.where(j==d2)
	index1 = np.random.choice(list1[0].shape[0], 1, replace=False)
	index2 = list1[0][index1,]
	print(j,index2)     #index2 is the articleID and j is the article rank
				

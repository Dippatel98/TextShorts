from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re
import math


#function to find distance b/w points
def distance(a,b):
	vect=(a-b)**2
	dis=0
	for a in vect:
		dis+=a
	return math.sqrt(dis)

#read text file and separate lines, then remove special symbols and from line containing only words
lineAry=[]
with open("data.txt") as file:
	for line in file:
		processedLine=""
		for word in re.findall(r"[\w']+", line):
			processedLine=processedLine+word
		lineAry.append(processedLine)

print(lineAry)


#count vectorize lines with allowed ngram upto 2
vectorizer = CountVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(lineAry)

print(X.toArray())

#remove trivial (un-interesting) words
X=TfidfTransformer().fit_transform(X)

print(X.toArray())

#cluster the document
k= KMeans(n_clusters=min(len(lineAry),4), random_state=0).fit_transform(X)

#mean points
centers=k.cluster_centers_

#find points closest to the means
minIndex={}
min={}
i=0
for val in k.labels_:
	d=distance(X[i],centers[val])
	if !(val in minIndex):
		min[val]=d
		minIndex[val]=i
	else :
		
		if d<min[val]:
			min[val]=d
			minIndex[val]=i		
	i=i+1


for key in minIndex:
	index=a[key]
	print(lineAry[index])
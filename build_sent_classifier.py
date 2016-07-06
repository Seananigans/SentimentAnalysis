import pandas as pd
import nltk, pickle, random, re, sqlite3, time
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
		
	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	
	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes*1./len(votes)
		return conf
		

# try:
# 	save_features = open("this_is_not_here.pickle", "rb")
# 	save_features = open("features.pickle", "rb")
# 	feature_sets = pickle.load(save_features)
# 	save_features.close()
# except IOError, e:
# 	print str(e)
conn = sqlite3.connect('knowledgeBase.db')
conn.text_factory= str
c = conn.cursor()

# query = c.execute("SELECT * FROM imdb_reviews")
df = pd.read_sql("SELECT * FROM imdb_reviews", conn)
df["sentiment"] = 1.0
df.ix[df.value<6,"sentiment"] = -1.0
documents = [tuple((nltk.word_tokenize(x[0]),x[1])) for x in df[['review','sentiment']].values ]

random.shuffle(documents)

all_words = []
for doc in documents:
	[all_words.append(word.lower()) for word in doc[0]]

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(doc):
	words = set(doc)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	
	return features

feature_sets = [(find_features(rev), sent) for rev,sent in documents]

# 	save_features = open("features.pickle", "wb")
# 	pickle.dump(feature_sets, save_features)
# 	save_features.close()

doc_rows = len(feature_sets)
train_rows = int(doc_rows*0.6)
cross_rows = train_rows + int(doc_rows*0.2)

training = feature_sets[:train_rows]
cross_val = feature_sets[train_rows:cross_rows]
testing = feature_sets[cross_rows:]

print "Length training",len(training)
print "Length cross validation",len(cross_val)
print "Length testing",len(testing)
print "Length of training, cross val, and testing",len(training)+len(cross_val)+len(testing)
print "Number of documents",doc_rows

"""SIMPLE NAIVE BAYES"""
try:
	with open("naive_bayes.pickle","rb") as classifier_f:
		classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("naive_bayes.pickle","wb") as classifier_f:
		classifier = nltk.NaiveBayesClassifier.train(training)
		pickle.dump(classifier, classifier_f)
# print "Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(classifier,cross_val)*100.
# classifier.show_most_informative_features(15)

"""MULTINOMIAL NAIVE BAYES"""
try:
	with open("MNB_classifier.pickle","rb") as classifier_f:
		MNB_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("MNB_classifier.pickle","wb") as classifier_f:
		MNB_classifier = SklearnClassifier(MultinomialNB())
		MNB_classifier.train(training)
		pickle.dump(MNB_classifier, classifier_f)
# print "Multinomial Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(MNB_classifier,cross_val)*100.

"""BERNOULLI NAIVE BAYES"""
try:
	with open("BNB_classifier.pickle","rb") as classifier_f:
		BNB_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("BNB_classifier.pickle","wb") as classifier_f:
		BNB_classifier = SklearnClassifier(BernoulliNB())
		BNB_classifier.train(training)
		pickle.dump(BNB_classifier, classifier_f)
# print "Bernoulli Naive Bayes Algorithm Accuracy Percent:", nltk.classify.accuracy(BNB_classifier,cross_val)*100.


"""LOGISTIC REGRESSION"""
try:
	with open("LogReg_classifier.pickle","rb") as classifier_f:
		LogReg_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("LogReg_classifier.pickle","wb") as classifier_f:
		LogReg_classifier = SklearnClassifier(LogisticRegression())
		LogReg_classifier.train(training)
		pickle.dump(LogReg_classifier, classifier_f)
# print "Logistic Regression Algorithm Accuracy Percent:", nltk.classify.accuracy(LogReg_classifier,cross_val)*100.

"""STOCHASTIC GRADIENT DESCENT"""
try:
	with open("SGD_classifier.pickle","rb") as classifier_f:
		SGD_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("SGD_classifier.pickle","wb") as classifier_f:
		SGD_classifier = SklearnClassifier(SGDClassifier())
		SGD_classifier.train(training)
		pickle.dump(SGD_classifier, classifier_f)
# print "SGD Algorithm Accuracy Percent:", nltk.classify.accuracy(SGD_classifier,cross_val)*100.


"""SUPPORT VECTOR CLASSIFIER"""
try:
	with open("SVC_classifier.pickle","rb") as classifier_f:
		SVC_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("SVC_classifier.pickle","wb") as classifier_f:
		SVC_classifier = SklearnClassifier(SVC())
		SVC_classifier.train(training)
		pickle.dump(SVC_classifier, classifier_f)
# print "SUPPORT VECTOR CLASSIFIER Algorithm Accuracy Percent:", nltk.classify.accuracy(SVC_classifier,cross_val)*100.


"""LINEAR SUPPORT VECTOR CLASSIFIER"""
try:
	with open("LSVC_classifier.pickle","rb") as classifier_f:
		LSVC_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	with open("LSVC_classifier.pickle","wb") as classifier_f:
		LSVC_classifier = SklearnClassifier(LinearSVC())
		LSVC_classifier.train(training)
		pickle.dump(LSVC_classifier, classifier_f)
# print "LINEAR SUPPORT VECTOR CLASSIFIER Algorithm Accuracy Percent:", nltk.classify.accuracy(LSVC_classifier,cross_val)*100.

"""Voted Classifier"""
try:
	with open("Vote_classifier.pickle","rb") as classifier_f:
		Vote_classifier = pickle.load(classifier_f)
except IOError, e:
	print str(e)
	print "here"
	with open("Vote_classifier.pickle","wb") as classifier_f:
		Vote_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogReg_classifier, SGD_classifier, SVC_classifier, LSVC_classifier)
		pickle.dump(Vote_classifier, classifier_f)
print "Vote Classifier Algorithm Accuracy Percent:", nltk.classify.accuracy(Vote_classifier,cross_val)*100.

